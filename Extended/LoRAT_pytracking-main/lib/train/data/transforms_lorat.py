

import numbers
from typing import Sequence, Tuple, Union
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from lib.utils.bbox.flip import bbox_horizontal_flip
from PIL import ImageFilter, ImageOps, Image


class HorizontalFlipAugmentation():
    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, images: Sequence[torch.Tensor], bboxes: Sequence[np.ndarray], rng_engine: np.random.Generator) -> Tuple[Sequence[torch.Tensor], Sequence[np.ndarray]]:
        if self.probability > 0 and rng_engine.random() < self.probability:
            # assert all(len(img.shape) == 3 for img in images)  # CHW
            images = tuple(F.hflip(img) for img in images)  # F.hflip is valid for multiple channels(3/6/9/18)
            all_image_w = tuple(img.shape[-1] for img in images)
            bboxes = tuple(bbox_horizontal_flip(bbox, w) for bbox, w in zip(bboxes, all_image_w))
        return images, bboxes

#######################################################################################################################################

def _check_input(value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError(f"If {name} is a single number, it must be non negative.")
        value = [center - float(value), center + float(value)]
        if clip_first_on_zero:
            value[0] = max(value[0], 0.0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        value = [float(value[0]), float(value[1])]
    else:
        raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

    if not bound[0] <= value[0] <= value[1] <= bound[1]:
        raise ValueError(f"{name} values should be between {bound}, but got {value}.")

    # if value is 0 or (1., 1.) for brightness/contrast/saturation
    # or (0., 0.) for hue, do nothing
    if value[0] == value[1] == center:
        return None
    else:
        return tuple(value)


class ColorJitter():
    def __init__(self,
                 brightness: Union[float, Tuple[float, float]] = 0,
                 contrast: Union[float, Tuple[float, float]] = 0,
                 saturation: Union[float, Tuple[float, float]] = 0,
                 hue: Union[float, Tuple[float, float]] = 0):
        self.brightness = _check_input(brightness, "brightness")
        self.contrast = _check_input(contrast, "contrast")
        self.saturation = _check_input(saturation, "saturation")
        self.hue = _check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _adjust(self, fn, factor, img_multiple_channel):
        """valid for 3 channel and multiple channel image"""
        tmp = []
        for three_channel_img in torch.split(img_multiple_channel, 3, dim=0):
            img = fn(three_channel_img, factor)
            tmp.append(img)
        img_multiple_channel = torch.cat(tmp, dim=0)
        return img_multiple_channel

    def __call__(self, images: Sequence[torch.Tensor], bboxes: Sequence[np.ndarray], rng_engine: np.random.Generator) -> Tuple[Sequence[torch.Tensor], Sequence[np.ndarray]]:
        fn_idx = rng_engine.permutation(4)

        brightness_factor = None if self.brightness is None else float(rng_engine.uniform(self.brightness[0], self.brightness[1]))
        contrast_factor = None if self.contrast is None else float(rng_engine.uniform(self.contrast[0], self.contrast[1]))
        saturation_factor = None if self.saturation is None else float(rng_engine.uniform(self.saturation[0], self.saturation[1]))
        hue_factor = None if self.hue is None else float(rng_engine.uniform(self.hue[0], self.hue[1]))


        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                images = tuple(self._adjust(F.adjust_brightness, brightness_factor, img) for img in images)
            elif fn_id == 1 and contrast_factor is not None:
                images = tuple(self._adjust(F.adjust_contrast, contrast_factor, img) for img in images)
            elif fn_id == 2 and saturation_factor is not None:
                images = tuple(self._adjust(F.adjust_saturation, saturation_factor, img) for img in images)
            elif fn_id == 3 and hue_factor is not None:
                images = tuple(self._adjust(F.adjust_hue, hue_factor, img) for img in images)
        return images, bboxes


#######################################################################################################################################


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, images: Sequence[Image.Image], rng_engine: np.random.Generator):
        if self.prob == 1 or (self.prob > 0 and rng_engine.random() <= self.prob):
            filter_ = ImageFilter.GaussianBlur(
                radius=rng_engine.uniform(self.radius_min, self.radius_max)
            )
            images = tuple(img.filter(filter_) for img in images)

        return images


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, images: Sequence[Image.Image], rng_engine: np.random.Generator):
        if self.p == 1 or (self.p > 0 and rng_engine.random() <= self.p):
            images = tuple(ImageOps.solarize(img) for img in images)
        return images


class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, images: Sequence[Image.Image], rng_engine: np.random.Generator):
        if self.p == 1 or (self.p > 0 and rng_engine.random() <= self.p):
            images = tuple(self.transf(img) for img in images)
        return images


class DeiT3Augmentation():
    def __init__(self):
        self.augmentations = ([gray_scale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)])

    def _adjust(self, img_multiple_channel):
        """valid for 3 channel and multiple channel image"""
        tmp = []
        for three_channel_img in torch.split(img_multiple_channel, 3, dim=0):
            img = F.to_pil_image(three_channel_img, mode='RGB')
            tmp.append(img)
        img_multiple_channel = torch.cat(tmp, dim=0)
        return img_multiple_channel

    def __call__(self, images: Sequence[torch.Tensor], bboxes: Sequence[np.ndarray], rng_engine: np.random.Generator) -> Tuple[Sequence[torch.Tensor], Sequence[np.ndarray]]:
        aug_index = rng_engine.choice(3)
        aug = self.augmentations[aug_index]

        tmp = []
        for img_multiple_channel in images:
            pil_images = tuple(F.to_pil_image(three_channel_img, mode='RGB') for three_channel_img in torch.split(img_multiple_channel, 3, dim=0))
            pil_images = aug(pil_images, rng_engine)
            tensor_images = tuple(F.to_tensor(pil_image) for pil_image in pil_images)
            img_multiple_channel_tensor = torch.cat(tensor_images, dim=0)
            tmp.append(img_multiple_channel_tensor)

        images = tuple(tmp)
        return images, bboxes

#######################################################################################################################################
class AugmentationPipeline:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, images, bboxes, rng_engine: np.random.Generator):
        # images: [z_cropped_images, x_cropped_images]
        # bboxes: [z_cropped_bbox, x_cropped_bbox]
        for augmentation in self.augmentations:
            augmentation_fn = augmentation[0]
            joint = augmentation[1]
            if joint:
                images, bboxes = augmentation_fn(images, bboxes, rng_engine)
                return images, bboxes
            else:
                new_images, new_bboxes = [], []
                for image, bbox in zip(images, bboxes):
                    augmented_images, augmented_bboxes = augmentation_fn([image], [bbox], rng_engine)
                    new_images.append(augmented_images[0])
                    new_bboxes.append(augmented_bboxes[0])
                return new_images, new_bboxes

