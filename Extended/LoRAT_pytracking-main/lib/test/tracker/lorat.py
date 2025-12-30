from timm.layers import to_2tuple

from lib.models.lorat import build_lorat
from lib.test.tracker.basetracker import BaseTracker
import torch

# for debug
import cv2
import numpy as np

from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.siamfc_cropping import get_siamfc_cropping_params, get_foreground_bounding_box
from lib.train.data.siamfc_cropping import apply_siamfc_cropping, reverse_siamfc_cropping_params, apply_siamfc_cropping_to_boxes
from lib.utils.bbox.utility.image import bbox_clip_to_image_boundary_, is_bbox_intersecting_image
from lib.utils.bbox.validity import bbox_is_valid
from lib.utils.bbox.format import bbox_xyxy_to_cxcywh, bbox_xyxy_to_xywh, bbox_cxcywh_to_xyxy

from contextlib import nullcontext
from functools import partial
import torchvision.transforms as transforms
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


# TODO:
# 1. 构建feat_mask_z来适应输入
# 2. 构建inference_mode和amp
# 3. crop的时候应该要用template_image_mean
# 4. 后处理的window_penalty
# 5. 预测在search上之后要map回到原图

class LoRAT(BaseTracker):
    def __init__(self, params, dataset_name):
        super(LoRAT, self).__init__(params)
        network = build_lorat(params.cfg, training=False)
        self.params.checkpoint = "/root/autodl-tmp/train/lorat/giant_378/LoRATBaseline_DINOv2_ep0011.pth.tar"
        ckpt = torch.load(self.params.checkpoint, map_location='cpu')
        if 'net' in ckpt:
            ckpt = ckpt['net']
        network.load_state_dict(ckpt, strict=True)  # , map_location='cpu'
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.device = self.preprocessor.mean.device
        self._window_penalty_ratio = params.window_penalty

        self.stride = self.cfg.MODEL.BACKBONE.STRIDE if 'patch8' not in self.cfg.MODEL.BACKBONE.TYPE else 16
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.stride
        self.feat_sz_z = self.cfg.TEST.TEMPLATE_SIZE // self.stride

        # for debug
        self.debug = 0 # params.debug
        self.use_visdom = 0  # params.debug
        print(f"Tracking: debug: {self.debug} | use_visdom: {self.use_visdom}")
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                # if not os.path.exists(self.save_dir):
                #     os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = None
        self.image_normalize_transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def initialize(self, image, info: dict):
        self.last_pred = None
        init_gt_bbox = np.asarray(info['init_bbox'], dtype=np.float64)
        init_gt_bbox[..., 2:] += init_gt_bbox[..., :2]  # xywh -> xyxy
        self.cached_bbox = init_gt_bbox.copy()


        z = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        template_curation_parameter = get_siamfc_cropping_params(init_gt_bbox, self.params.template_factor, np.array(to_2tuple(self.params.template_size)))

        z_curated, z_image_mean, template_curation_parameter = apply_siamfc_cropping(
            z.to(torch.float32), np.array(to_2tuple(self.params.template_size)), template_curation_parameter, "bilinear", False)

        z_curated.div_(255.)
        z_curated = self.image_normalize_transform(z_curated).unsqueeze(0)
        self.template_image_mean = z_image_mean

        with torch.inference_mode():
            self.z_dict1 = z_curated

        # construct "z_feat_mask"
        template_mask = torch.full((self.feat_sz_z, self.feat_sz_z), 0, dtype=torch.long)
        template_cropped_bbox = get_foreground_bounding_box(init_gt_bbox, template_curation_parameter, to_2tuple(self.stride))
        assert bbox_is_valid(template_cropped_bbox)  # xyxy format
        template_cropped_bbox = torch.from_numpy(template_cropped_bbox)
        template_mask[template_cropped_bbox[1]: template_cropped_bbox[3], template_cropped_bbox[0]: template_cropped_bbox[2]] = 1
        self.z_feat_mask = template_mask.unsqueeze(0).to(self.device)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        image_size = np.array((W, H), dtype=np.int32)
        self.frame_id += 1

        x = torch.from_numpy(image).permute(2, 0, 1).to(self.device)

        cropping_params = get_siamfc_cropping_params(self._adjust_bbox_size(self.cached_bbox, np.array(to_2tuple(10.0), dtype=float)),
                                                     self.params.search_factor, np.asarray(to_2tuple(self.cfg.TEST.SEARCH_SIZE)))

        x_curated, _, x_cropping_params = apply_siamfc_cropping(x.to(torch.float32), np.array(to_2tuple(self.cfg.TEST.SEARCH_SIZE)),
                                                cropping_params, "bilinear", False, self.template_image_mean)

        # resize_factor = adjusted_cropping_params[0][0]

        x_curated = x_curated.unsqueeze(0) / 255.
        x_curated = self.image_normalize_transform(x_curated)

        amp_autocast_fn = self.get_torch_amp_autocast_fn(self.device.type, True, torch.float16)
        
        # -----------------vis--------------------------
        _,_,img_height, img_width = x_curated.shape
        block_size = 14
        class_list = [1]*int(int((img_height)// block_size)*int((img_height)// block_size))
        #print(len(class_list))
        if self.last_pred is not None:
            # -----------------vis--------------------------
            class_list = []
            _,_,img_height, img_width = x_curated.shape

            #target_loc = info['gt_bbox']
            # 坐标框的边界
            target_loc = self.last_pred
            x, y, xx, yy = target_loc
            target_x1, target_y1 = x, y
            target_x2, target_y2 = xx, yy
            w = int(xx-x)
            h = int(yy-y)
            
            

            # 计算目标区域在block坐标系中的范围
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

            # -----------------vis--------------------------
            
            
        with torch.inference_mode(), amp_autocast_fn():
            out_dict = self.network.forward(
                z=self.z_dict1, x=x_curated, z_feat_mask=self.z_feat_mask,class_list=class_list)

        # add hann windows
        pred_outputs = self.postprocessing_boxwithscoremap(output=out_dict)
        pred_boxes = pred_outputs['box'].view(4).cpu().to(torch.float64).numpy()
        self.last_pred = pred_boxes
        confidence = pred_outputs['confidence'].item()
        pred_boxes_on_full_search_image = apply_siamfc_cropping_to_boxes(
            pred_boxes, reverse_siamfc_cropping_params(x_cropping_params))
        bbox_clip_to_image_boundary_(pred_boxes_on_full_search_image, image_size)
        self.update_cropping_params(pred_boxes_on_full_search_image, image_size)
        self.state = bbox_xyxy_to_xywh(pred_boxes_on_full_search_image).tolist()

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                cv2.imshow('debug', image_BGR)
                cv2.waitKey(0)
                # cv2.destroyAllWindows()

        return {"target_bbox": self.state, "confidence": confidence}

    def update_cropping_params(self, predicted_bbox: np.ndarray, image_size: np.ndarray):
        assert image_size[0] > 0 and image_size[1] > 0

        if not bbox_is_valid(predicted_bbox):
            return
        if not is_bbox_intersecting_image(predicted_bbox, image_size):
            return
        self.cached_bbox = predicted_bbox

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

    def postprocessing_boxwithscoremap(self, output):
        self._scale_factor = torch.tensor((self.params.search_size, self.params.search_size), dtype=torch.float).to(self.device)
        self._window = torch.flatten(torch.outer(torch.hann_window(self.feat_sz, periodic=False),
                                                 torch.hann_window(self.feat_sz, periodic=False))).to(self.device)

        # shape: (N, H, W), (N, H, W, 4)
        predicted_score_map = output['score_map'].detach().float().sigmoid()
        predicted_bbox = output['boxes'].detach().float()

        N, H, W = predicted_score_map.shape
        predicted_score_map = predicted_score_map.view(N, H * W)

        # window penalty
        score_map_with_penalty = predicted_score_map * (1 - self._window_penalty_ratio) + \
                                 self._window.view(1, H * W) * self._window_penalty_ratio
        _, best_idx = torch.max(score_map_with_penalty, 1, keepdim=True)
        confidence_score = torch.gather(predicted_score_map, 1, best_idx)

        confidence_score = confidence_score.squeeze(1)
        predicted_bbox = predicted_bbox.view(N, H * W, 4)
        bounding_box = torch.gather(predicted_bbox, 1, best_idx.view(N, 1, 1).expand(-1, -1, 4)).squeeze(1)
        bounding_box = (bounding_box.view(N, 2, 2) * self._scale_factor.view(1, 1, 2)).view(N, 4)
        return {'box': bounding_box, 'confidence': confidence_score}


    def get_torch_amp_autocast_fn(self, device_type: str, enabled: bool, dtype: torch.dtype):
        # to be removed in the future, once they are supported
        if device_type == 'mps' and enabled:
            print('Auto mixed precision is disabled. reason: Auto mixed precision is not supported on MPS.', flush=True)
            enabled = False
        if device_type == 'cpu' and dtype == torch.float16 and enabled:
            dtype = torch.bfloat16
            print(f'Warning: CPU does not support float16, use bfloat16 instead', flush=True)
        return partial(torch.amp.autocast, device_type=device_type, enabled=enabled, dtype=dtype) if enabled else nullcontext

    def _adjust_bbox_size(self, bounding_box: np.ndarray, min_wh: np.ndarray):
        bounding_box = bbox_xyxy_to_cxcywh(bounding_box)
        bounding_box[2:4] = np.maximum(bounding_box[2:4], min_wh)
        return bbox_cxcywh_to_xyxy(bounding_box)

def get_tracker_class():
    return LoRAT