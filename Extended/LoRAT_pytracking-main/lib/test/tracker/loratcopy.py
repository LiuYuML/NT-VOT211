from timm.layers import to_2tuple

from lib.models.lorat import build_lorat
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import box_xyxy_to_cxcywh
# for debug
import cv2
import numpy as np

from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.siamfc_cropping import get_siamfc_cropping_params, get_foreground_bounding_box
from lib.utils.box_ops import clip_box, bbox_is_valid

from contextlib import nullcontext
from functools import partial


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
        # network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
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
        self.debug = params.debug
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
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        self.template_image_mean = np.mean(image, axis=(0, 1))
        init_gt_bbox = np.asarray(info['init_bbox'], dtype=np.float64)
        init_gt_bbox[..., 2:] += init_gt_bbox[..., :2]  # xywh -> xyxy
        curation_parameter = get_siamfc_cropping_params(init_gt_bbox, self.params.template_factor, np.array(to_2tuple(self.params.template_size)))

        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size, padding_value=self.template_image_mean)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.inference_mode():
            self.z_dict1 = template

        # construct "z_feat_mask"
        template_mask = torch.full((self.feat_sz_z, self.feat_sz_z), 0, dtype=torch.long)
        template_cropped_bbox = get_foreground_bounding_box(init_gt_bbox, curation_parameter, to_2tuple(self.stride))
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
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size, padding_value=self.template_image_mean)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        amp_autocast_fn = self.get_torch_amp_autocast_fn(self.device.type, True, torch.float16)

        with torch.inference_mode(), amp_autocast_fn():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                z=self.z_dict1.tensors, x=x_dict.tensors, z_feat_mask=self.z_feat_mask)

        # add hann windows
        pred_outputs = self.postprocessing_boxwithscoremap(output=out_dict)
        pred_boxes = box_xyxy_to_cxcywh(pred_outputs['box'].view(-1, 4))
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                cv2.imshow('debug', image_BGR)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

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


def get_tracker_class():
    return LoRAT
