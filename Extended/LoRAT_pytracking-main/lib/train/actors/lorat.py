from . import BaseActor
import torch
import numpy as np
from lib.utils.bbox.rasterize import bbox_rasterize, bbox_rasterize_torch
from lib.utils.iou_loss import bbox_overlaps

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
        # currently only support 1 template and 1 search region
        assert len(data['z_cropped_images']) == 1
        assert len(data['x_cropped_images']) == 1

        template_img = data['z_cropped_images'][0].view(-1, *data['z_cropped_images'].shape[2:])  # (batch, 3, 224, 224)
        search_img = data['x_cropped_images'][0].view(-1, *data['x_cropped_images'].shape[2:])  # (batch, 3, 224, 224)
        z_feat_mask = data['z_feat_mask'][0].view(-1, *data['z_feat_mask'].shape[2:])  # (batch, 8, 8)

        out_dict = self.net(z=template_img,
                            x=search_img,
                            z_feat_mask=z_feat_mask)

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

        loss = self.loss_weight['giou'] * reg_loss + self.loss_weight['bce'] * cls_loss

        if return_status:
            # status for log
            mean_iou = ious.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": reg_loss.item(),
                      "Loss/bce": cls_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
