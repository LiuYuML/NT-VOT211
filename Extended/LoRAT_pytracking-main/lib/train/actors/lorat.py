from . import BaseActor
import torch
import numpy as np
from lib.utils.bbox.rasterize import bbox_rasterize, bbox_rasterize_torch
from lib.utils.iou_loss import bbox_overlaps
import matplotlib.pyplot as plt
import os
from datetime import datetime
from matplotlib.patches import Rectangle
import copy



def save_plot_with_timestamp(folder_path, extension='png'):
    """
    Saves the current plot with a filename based on the current timestamp.

    Parameters:
    - folder_path: The path to the folder where the plot will be saved.
    - extension: The file extension (default is 'png').

    Returns:
    - The full path of the saved file.
    """
    # Get the current timestamp
    now = datetime.now()
    timestamp_str = now.strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]  # Milliseconds

    # Create the file name
    file_name = f'{timestamp_str}.{extension}'

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the plot to the specified folder
    full_path = os.path.join(folder_path, file_name)
    

    return full_path



class LoRATActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # print(data.keys()) 
        # # odict_keys(['dataset', 'z_cropped_images', 
        # 'z_cropped_bboxes', 'x_cropped_images', 
        # 'x_cropped_bboxes', 'z_feat_mask', 
        # 'valid', 'epoch', 'settings'])
        # 
        # currently only support 1 template and 1 search region
        assert len(data['z_cropped_images']) == 1
        assert len(data['x_cropped_images']) == 1

        template_img = data['z_cropped_images'][0].view(-1, *data['z_cropped_images'].shape[2:])  # (batch, 3, 224, 224)
        search_img = data['x_cropped_images'][0].view(-1, *data['x_cropped_images'].shape[2:])  # (batch, 3, 224, 224)
        z_feat_mask = data['z_feat_mask'][0].view(-1, *data['z_feat_mask'].shape[2:])  # (batch, 8, 8)
        # -----------------vis--------------------------
        #print(search_img.shape)
        _,_,img_height, img_width = search_img.shape
        block_size = 14
        class_list = [1]*int(int((img_height)// block_size)*int((img_height)// block_size))
        #print(len(class_list))
        # -----------------vis--------------------------
        class_list = []
        _,_,img_height, img_width = search_img.shape

        #target_loc = info['gt_bbox']
        # 坐标框的边界
        target_loc = data['x_cropped_bboxes']
        #print(target_loc.shape)
        target_loc_ = copy.deepcopy(target_loc)
        x, y, xx, yy = target_loc_[0,0].cpu().numpy()
        x, y, xx, yy = x*img_width, y*img_height, xx*img_width, yy*img_height
        #print("leoyu",x,y,xx,yy)
        target_x1, target_y1 = x, y
        target_x2, target_y2 = xx, yy
        w = int(xx-x)
        h = int(yy-y)
        target_block_x1 = int((target_x1 + block_size - 1) // block_size)
        target_block_y1 = int((target_y1 + block_size - 1) // block_size)
        target_block_x2 = int((target_x2)// block_size-1)
        target_block_y2 = int((target_y2)// block_size-1)

        # 创建一个与图像大小相同的全零数组，用于存储分类结果
        class_array = np.zeros((img_height // block_size, img_width // block_size), dtype=int)

        # 将完全包含在目标区域内的小块区域标记为1
        class_array[target_block_y1:target_block_y2 + 1, target_block_x1:target_block_x2 + 1] = 1

        # 将与目标区域相交但不完全包含的小块区域标记为-1
        # 计算边界区域
        
        left_edge = np.arange(target_block_y1-1, target_block_y2 + 2)
        right_edge = np.arange(target_block_y1-1, target_block_y2 + 2)
        top_edge = np.arange(target_block_x1-1, target_block_x2 + 2)
        bottom_edge = np.arange(target_block_x1-1, target_block_x2 + 2)
        
        
        left_edge = np.clip(left_edge, 0, int((img_height)// block_size-1))
        right_edge = np.clip(right_edge, 0, int((img_height)// block_size-1))
        top_edge = np.clip(top_edge, 0, int((img_width)// block_size-1))
        bottom_edge = np.clip(bottom_edge, 0, int((img_width)// block_size-1))
        
        #print(target_block_y2-1,target_block_x2,left_edge,right_edge,top_edge,bottom_edge)
        # 标记左侧边界
        if target_block_x1 > 0:
            class_array[left_edge, target_block_x1 - 1] = -1
        # 标记右侧边界
        if target_block_x2 < (img_width // block_size) - 1:
            class_array[right_edge, target_block_x2 + 1] = -1
        # 标记顶部边界
        if target_block_y1 > 0:
            class_array[target_block_y1 - 1, top_edge] = -1
        # 标记底部边界
        if target_block_y2 < (img_height // block_size) - 1:
            class_array[target_block_y2 + 1, bottom_edge] = -1

        # 将class_array展平为一维数组，得到最终的class_list
        class_list = class_array.flatten().tolist()
        #print("odtrack:",len(class_list)) # 729的token length
        check_results = False
        if check_results:
            class_array = np.array(class_list).reshape(img_height // block_size, img_width // block_size)

            # 可视化结果
            plt.figure(figsize=(10, 10))
            x_vis = search_img[0]
            image_np = x_vis.cpu().numpy()
            image_np = image_np/image_np.max()

            # 调整形状为 (H, W, C)
            image_np = image_np.transpose(1, 2, 0)
            img = image_np
            # 展示原图
            plt.imshow(img)

            # 绘制目标框
            plt.gca().add_patch(Rectangle((target_x1, target_y1), w, h, edgecolor='red', facecolor='none', linewidth=2))

            # 叠加分类结果
            for i in range(0, img_height, block_size):
                for j in range(0, img_width, block_size):
                    block_x1, block_y1 = j, i
                    block_x2, block_y2 = j + block_size, i + block_size
                    idx = (i // block_size) * (img_width // block_size) + (j // block_size)
                    if class_list[idx] == 1:
                        plt.gca().add_patch(Rectangle((block_x1, block_y1), block_size, block_size, edgecolor='green', facecolor='none', linewidth=1))
                    elif class_list[idx] == -1:
                        plt.gca().add_patch(Rectangle((block_x1, block_y1), block_size, block_size, edgecolor='blue', facecolor='none', linewidth=1))

            plt.title('Original Image with Target Box and Block Classification')
            folder_path = '/root/LY/aaaaaaaaaaaaaaaaaaaaaaaaaaaa/ltmp'

            # Save the plot with a timestamp-based filename
            saved_file_path = save_plot_with_timestamp(folder_path)
            plt.savefig(saved_file_path)
            print(f'Plot saved to: {saved_file_path}')
        # giant_378
        # -----------------vis--------------------------
        
        
        
        out_dict = self.net(z=template_img,
                            x=search_img,
                            z_feat_mask=z_feat_mask,class_list=class_list)

        return out_dict

    def positive_sample_assignment(self, bbox: torch.Tensor, response_map_size, search_region_size):
        '''

        :param bbox: (4,), in (xyxy) format
        :param response_map_size: (2,), response map size
        :param search_region_size: (2,), input search region size
        :return:
        '''
        scale = response_map_size / search_region_size
        indices = torch.arange(0, response_map_size * response_map_size, dtype=torch.int64, device=bbox.device)
        indices = indices.reshape(response_map_size, response_map_size)
        scaled_bbox = bbox.clone()
        scaled_bbox[::2] = scaled_bbox[::2] * scale
        scaled_bbox[1::2] = scaled_bbox[1::2] * scale
        rasterized_scaled_bbox = bbox_rasterize_torch(scaled_bbox, dtype=torch.int64)
        positive_sample_indices = indices[rasterized_scaled_bbox[1]: rasterized_scaled_bbox[3],
                                  rasterized_scaled_bbox[0]: rasterized_scaled_bbox[2]].flatten()
        assert len(positive_sample_indices) > 0, (f'bbox is too small.\n'
                                                  f'scale:\n{scale}\n'
                                                  f'bbox:\n{bbox}\n'
                                                  f'rasterized_scaled_bbox\n{rasterized_scaled_bbox}\n'
                                                  f'scaled_bbox:\n{scaled_bbox}')
        return positive_sample_indices

    def compute_losses(self, pred_dict, gt_dict, return_status=True):

        predicted_score_map = pred_dict['score_map'].to(torch.float)   # [B,H,W]
        predicted_bboxes = pred_dict['boxes'].to(torch.float)  # [B,H,W,4]
        groundtruth_bboxes = gt_dict['x_cropped_bboxes'][0]  # [0,1] xyxy  [B,4]
        resmax = pred_dict["delta"]
        
        N, H, W = predicted_score_map.shape
        search_region_size = self.cfg.DATA.SEARCH.SIZE

        collated_batch_ids = []
        collated_positive_sample_indices = []
        num_positive_samples = 0
        for batch_idx in range(len(groundtruth_bboxes)):
            positive_sample_indices = self.positive_sample_assignment(groundtruth_bboxes[batch_idx]*search_region_size,
                                                                      H, search_region_size)

            collated_batch_ids.append(torch.full((len(positive_sample_indices),), batch_idx, dtype=torch.long))
            collated_positive_sample_indices.append(positive_sample_indices.to(torch.long))
            num_positive_samples += len(positive_sample_indices)
        num_positive_samples = torch.as_tensor((num_positive_samples,), dtype=torch.float, device=predicted_score_map.device)
        if num_positive_samples > 0:
            positive_sample_batch_dim_index = torch.cat(collated_batch_ids)
            positive_sample_feature_map_dim_index = torch.cat(collated_positive_sample_indices)

        has_positive_samples = positive_sample_batch_dim_index is not None

        if has_positive_samples:
            predicted_bboxes = predicted_bboxes.view(N, H * W, 4)
            predicted_bboxes = predicted_bboxes[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index] # [Np, 4]
            groundtruth_bboxes = groundtruth_bboxes[positive_sample_batch_dim_index]  # [Np, 4]

        with torch.no_grad():
            groundtruth_response_map = torch.zeros((N, H * W),  dtype=torch.float32, device=predicted_score_map.device)
            if self.cfg.TRAIN.IOU_AWARE_CLASSIFICATION_SCORE:
                ious = bbox_overlaps(groundtruth_bboxes, predicted_bboxes, is_aligned=True)
                groundtruth_response_map.index_put_(
                    (positive_sample_batch_dim_index, positive_sample_feature_map_dim_index),
                    ious)
            else:
                groundtruth_response_map[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index] = 1.

        cls_loss = self.objective['bce'](predicted_score_map.view(N, -1), groundtruth_response_map).sum() / num_positive_samples

        if has_positive_samples:
            reg_loss = self.objective['giou'](predicted_bboxes, groundtruth_bboxes).sum() / num_positive_samples
        else:
            reg_loss = predicted_bboxes.mean() * 0
        if resmax is not None:
            lossextra  = 1/torch.mean(torch.abs(resmax)).item()/23.6775
            loss = self.loss_weight['giou'] * reg_loss + self.loss_weight['bce'] * cls_loss/3.2 + lossextra*(self.loss_weight['bce']+self.loss_weight['giou'])*0.5
        else:
            loss = self.loss_weight['giou'] * reg_loss + self.loss_weight['bce'] * cls_loss
        if return_status:
            # status for log
            mean_iou = ious.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": reg_loss.item(),
                      "Loss/bce": cls_loss.item(),
                      "Resmax":lossextra,
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
