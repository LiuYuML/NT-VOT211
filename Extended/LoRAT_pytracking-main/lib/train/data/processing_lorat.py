import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import numpy as np
from dataclasses import dataclass, field
from timm.layers import to_2tuple
from typing import Tuple,Sequence
from lib.train.data.siamfc_cropping import prepare_siamfc_cropping_with_augmentation, apply_siamfc_cropping, apply_siamfc_cropping_to_boxes
from lib.utils.bbox.format import bbox_xywh_to_xyxy
from lib.utils.bbox.validity import bbox_is_valid
from lib.utils.bbox.utility.image import bbox_clip_to_image_boundary_
from lib.utils.bbox.rasterize import bbox_rasterize

@dataclass(frozen=True)
class SiamFCCroppingParameter:
    output_size: np.ndarray
    area_factor: float
    scale_jitter_factor: float = 0.
    translation_jitter_factor: float = 0.
    output_min_object_size_in_pixel: np.ndarray = field(default_factory=lambda: np.array((0., 0.)))  # (width, height)
    output_min_object_size_in_ratio: float = 0.  # (width, height)
    output_max_object_size_in_pixel: np.ndarray = field(default_factory=lambda: np.array((float("inf"), float("inf"))))  # (width, height)
    output_max_object_size_in_ratio: float = 1.  # (width, height)
    interpolation_mode: str = 'bilinear'
    interpolation_align_corners: bool = False

def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x



class LoRATProcessing():
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, min_object_size,
                 aug_transform=None, norm_mean_and_std=None, mode='pair', stride=14, settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.min_object_size = min_object_size
        self.aug_transform = aug_transform
        self.norm_mean_and_std = norm_mean_and_std
        self.mode = mode
        self.stride = stride
        self.settings = settings

    def _build_siamfc_cropping_parameter(self, siamfc_cropping_config: dict, output_size: Tuple[int, int],
                                         interpolation_mode: str,
                                         interpolation_align_corners: bool) -> SiamFCCroppingParameter:
        area_factor = siamfc_cropping_config['area_factor']
        scale_jitter_factor = siamfc_cropping_config.get('scale_jitter', 0.)
        translation_jitter_factor = siamfc_cropping_config.get('translation_jitter', 0.)
        output_min_object_size_in_pixel = np.array(to_2tuple(siamfc_cropping_config.get('min_object_size', (0., 0.))))
        output_max_object_size_in_pixel = np.array(to_2tuple(siamfc_cropping_config.get('max_object_size', (float("inf"), float("inf")))))
        output_min_object_size_in_ratio = siamfc_cropping_config.get('min_object_ratio', 0.)
        output_max_object_size_in_ratio = siamfc_cropping_config.get('max_object_ratio', 1.)
        return SiamFCCroppingParameter(np.array(output_size), area_factor, scale_jitter_factor, translation_jitter_factor,
                                       output_min_object_size_in_pixel, output_min_object_size_in_ratio,
                                       output_max_object_size_in_pixel, output_max_object_size_in_ratio,
                                       interpolation_mode, interpolation_align_corners)

    def do_jitter_prepare(self, siamfc_cropping_parameter, xyxy_anno, rng_engine):
        cropping_parameters = []
        is_success_flags = []

        for num_idx in range(len(xyxy_anno)):
            cropping_parameter, is_success = prepare_siamfc_cropping_with_augmentation(
                xyxy_anno[num_idx], siamfc_cropping_parameter.area_factor,
                siamfc_cropping_parameter.output_size, siamfc_cropping_parameter.scale_jitter_factor,
                siamfc_cropping_parameter.translation_jitter_factor, rng_engine,
                siamfc_cropping_parameter.output_min_object_size_in_pixel,
                siamfc_cropping_parameter.output_max_object_size_in_pixel,
                siamfc_cropping_parameter.output_min_object_size_in_ratio,
                siamfc_cropping_parameter.output_max_object_size_in_ratio)
            cropping_parameters.append(cropping_parameter)
            is_success_flags.append(is_success)

        return cropping_parameters, is_success_flags

    def do_cropping_according_parameter(self, image, xyxy_bbox, output_size, cropping_parameter,
                                        interpolation_mode='bilinear', align_corners=False, normalized=True):

        image_cropped, image_mean, real_cropping_parameter = apply_siamfc_cropping(
            image, output_size, cropping_parameter,
            interpolation_mode=interpolation_mode,
            align_corners=align_corners
        )
        if normalized:
            image_cropped.div_(255.)

        bbox_cropped = apply_siamfc_cropping_to_boxes(xyxy_bbox, real_cropping_parameter)
        return image_cropped, bbox_cropped

    def _bbox_clip_to_image_boundary_(self, bboxes: Sequence[np.ndarray], images: Sequence[torch.Tensor]):
        new_images, new_bboxes = [], []
        for image, bbox in zip(images, bboxes):
            h, w = image.shape[-2:]
            bbox_clip_to_image_boundary_(bbox, np.array((w, h)))
            assert bbox_is_valid(bbox), f'bbox:\n{bbox}\nimage_size:\n{image.shape}'
            new_images.append(image)
            new_bboxes.append(bbox)
        return new_bboxes, new_images

    def create_z_feat_map_mask(self, xyxy_bbox, stride, template_feat_size):
        mask = np.full((template_feat_size[1], template_feat_size[0]), 0, dtype=np.int64)
        z_cropped_bbox = xyxy_bbox.copy()
        z_cropped_bbox[0] /= stride[0]
        z_cropped_bbox[1] /= stride[1]
        z_cropped_bbox[2] /= stride[0]
        z_cropped_bbox[3] /= stride[1]
        z_cropped_bbox = bbox_rasterize(z_cropped_bbox, dtype=np.int64)
        mask[z_cropped_bbox[1]:z_cropped_bbox[3], z_cropped_bbox[0]:z_cropped_bbox[2]] = 1
        return mask


    def __call__(self, data: TensorDict):
        # data = TensorDict({'template_images': template_tensor_frames,
        #                    'template_anno': template_numpy_anno,
        #                    'search_images': search_tensor_frames,
        #                    'search_anno': search_numpy_anno,
        #                    'dataset': dataset.get_name()})

        num_template = len(data["template_images"])
        num_search = len(data["search_images"])
        ## prepare cropping parameters
        template_siamfc_cropping_config = {
            'area_factor': self.search_area_factor["template"],
            'scale_jitter': self.scale_jitter_factor["template"],
            'translation_jitter': self.center_jitter_factor["template"],
            'min_object_size': self.min_object_size["template"],
        }
        template_siamfc_cropping_parameter = self._build_siamfc_cropping_parameter(
            template_siamfc_cropping_config, to_2tuple(self.output_sz["template"]), 'bilinear', False)

        search_siamfc_cropping_config = {
            'area_factor': self.search_area_factor["search"],
            'scale_jitter': self.scale_jitter_factor["search"],
            'translation_jitter': self.center_jitter_factor["search"],
            'min_object_size': self.min_object_size["search"],
        }
        search_siamfc_cropping_parameter = self._build_siamfc_cropping_parameter(
            search_siamfc_cropping_config, to_2tuple(self.output_sz["search"]), 'bilinear', False)

        rng_engine = np.random.default_rng()

        xyxy_template_anno = [bbox_xywh_to_xyxy(bb) for bb in data["template_anno"]]
        xyxy_search_anno = [bbox_xywh_to_xyxy(bb) for bb in data["search_anno"]]

        z_cropping_parameters, z_success_flags = self.do_jitter_prepare(template_siamfc_cropping_parameter, xyxy_template_anno, rng_engine)
        x_cropping_parameters, x_success_flags = self.do_jitter_prepare(search_siamfc_cropping_parameter, xyxy_search_anno, rng_engine)
        assert all(z_success_flags), "fail to prepare z cropping parameters!"
        if not all(x_success_flags):
            data["valid"] = False
            return data

        ## do cropping
        data["z_cropped_images"] = []
        data["z_cropped_bboxes"] = []
        for num_idx in range(len(z_cropping_parameters)):
            z_cropped_image, z_cropped_bbox = self.do_cropping_according_parameter(
                data["template_images"][num_idx], xyxy_template_anno[num_idx],
                template_siamfc_cropping_parameter.output_size, z_cropping_parameters[num_idx])
            data["z_cropped_images"].append(z_cropped_image)
            data["z_cropped_bboxes"].append(z_cropped_bbox)
        del data["template_images"]
        del data["template_anno"]

        data["x_cropped_images"] = []
        data["x_cropped_bboxes"] = []
        for num_idx in range(len(x_cropping_parameters)):
            x_cropped_image, x_cropped_bbox = self.do_cropping_according_parameter(
                data["search_images"][num_idx], xyxy_search_anno[num_idx],
                search_siamfc_cropping_parameter.output_size, x_cropping_parameters[num_idx])
            data["x_cropped_images"].append(x_cropped_image)
            data["x_cropped_bboxes"].append(x_cropped_bbox)
        del data["search_images"]
        del data["search_anno"]

        # do augmentation(like: 'horizontal_flip', 'color_jitter', 'DeiT_3_aug')
        if self.aug_transform is not None:
            augmented_images, augmented_bboxes = self.aug_transform(data["z_cropped_images"]+data["x_cropped_images"],
                                                                    data["z_cropped_bboxes"]+data["x_cropped_bboxes"],
                                                                    rng_engine)
            data["z_cropped_images"], data["x_cropped_images"] = augmented_images[:num_template], augmented_images[num_template:]
            data["z_cropped_bboxes"], data["x_cropped_bboxes"] = augmented_bboxes[:num_template], augmented_bboxes[num_template:]

        # clip_by_boundary and mean/std normalization
        data['z_cropped_bboxes'], data['z_cropped_images'] = self._bbox_clip_to_image_boundary_(data['z_cropped_bboxes'], data['z_cropped_images'])
        data['x_cropped_bboxes'], data['x_cropped_images'] = self._bbox_clip_to_image_boundary_(data['x_cropped_bboxes'], data['x_cropped_images'])
        image_normalize_transform_ = transforms.Normalize(mean=self.norm_mean_and_std[0], std=self.norm_mean_and_std[1], inplace=True)
        for z in data['z_cropped_images']:
            for three_channel_img in torch.split(z, 3, dim=0):
                image_normalize_transform_(three_channel_img)
        for x in data['x_cropped_images']:
            for three_channel_img in torch.split(x, 3, dim=0):
                image_normalize_transform_(three_channel_img)

        # construct another input: z_feat_mask
        data["z_feat_mask"] = []
        template_feat_size = to_2tuple(self.output_sz['template']//self.stride)
        for i in range(num_template):
            data["z_feat_mask"].append(torch.from_numpy(self.create_z_feat_map_mask(data['z_cropped_bboxes'][i], to_2tuple(self.stride), template_feat_size)))

        # create targets(like: num_positive_samples, positive_sample_batch_dim_indices, positive_sample_map_dim_indices, boxes)
        for i in range(num_search):
            x_cropped_bbox = data['x_cropped_bboxes'][i]
            normalized_bbox = x_cropped_bbox.copy()
            normalized_bbox[::2] = x_cropped_bbox[::2] / self.output_sz['search']
            normalized_bbox[1::2] = x_cropped_bbox[1::2] / self.output_sz['search']
            normalized_bbox = normalized_bbox.astype(np.float32)
            data['x_cropped_bboxes'][i] = torch.from_numpy(normalized_bbox)

        data['valid'] = True
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
