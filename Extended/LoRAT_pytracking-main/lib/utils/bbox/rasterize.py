import numpy as np
import torch

def bbox_rasterize(bbox: np.ndarray, eps: float = 1e-4, dtype=np.int32):
    assert np.issubdtype(bbox.dtype, np.floating)
    bbox = bbox_rasterize_(bbox.copy(), eps)
    return bbox.astype(dtype)


def bbox_rasterize_(bbox: np.ndarray, eps: float = 1e-4):
    assert np.issubdtype(bbox.dtype, np.floating)
    bbox[..., 2] += (1 - eps)
    bbox[..., 3] += (1 - eps)
    return np.floor(bbox, out=bbox)




def bbox_rasterize_torch(bbox: torch.Tensor, eps: float = 1e-4, dtype=torch.int32):
    assert torch.is_floating_point(bbox), "bbox must be a floating point tensor"
    bbox = bbox_rasterize_torch_(bbox.clone(), eps)
    return bbox.to(dtype)

def bbox_rasterize_torch_(bbox: torch.Tensor, eps: float = 1e-4):
    assert torch.is_floating_point(bbox), "bbox must be a floating point tensor"
    bbox[..., 2] += (1 - eps)
    bbox[..., 3] += (1 - eps)
    return torch.floor(bbox, out=bbox)


if __name__ == "__main__":
    # 创建示例 bbox 张量
    bbox = torch.tensor([
        [10.2, 20.5, 30.6, 40.7],
        [50.1, 60.9, 70.4, 80.6]
    ], dtype=torch.float32).cuda()

    # 调用 bbox_rasterize 函数
    rasterized_bbox = bbox_rasterize_torch(bbox)

    print("Original bbox:")
    print(bbox)
    print("Rasterized bbox:")
    print(rasterized_bbox)
