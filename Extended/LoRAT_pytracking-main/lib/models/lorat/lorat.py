import os
from typing import Tuple
import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from lib.models.modules.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from lib.models.modules.patch_embed import PatchEmbedNoSizeCheck
from lib.models.modules.head.mlp import MlpAnchorFreeHead
from timm.layers import to_2tuple
from lib.models.lorat.builder import build_dino_v2_backbone
import numpy as np
class LoRATBaseline_DINOv2(nn.Module):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int]):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[1]
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size

        assert isinstance(vit, DinoVisionTransformer)
        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        self.pos_embed = nn.Parameter(torch.empty(1, self.x_size[0] * self.x_size[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                           self.x_size,
                                                           vit.patch_embed.patches_resolution,
                                                           num_prefix_tokens=0, interpolate_offset=0))

        self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
        trunc_normal_(self.token_type_embed, std=.02)

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)

    def forward(self, z: torch.Tensor, x: torch.Tensor, z_feat_mask: torch.Tensor,class_list=None):
        z_feat = self._z_feat(z, z_feat_mask)
        x_feat = self._x_feat(x)
        x_feat = self._fusion(z_feat, x_feat,class_list=class_list)
        return self.head(x_feat)
        # score_map: [B, H, W]
        # boxes:     [B, H, W, 4]

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):
        z = self.patch_embed(z)
        z_W, z_H = self.z_size
        z = z + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : z_H, : z_W, :].reshape(1, z_H * z_W, self.embed_dim)
        z = z + self.token_type_embed[z_feat_mask.flatten(1)]
        return z

    def _x_feat(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = x + self.token_type_embed[2].view(1, 1, self.embed_dim)
        return x

    def _cf_perturbation(self, z_feat: torch.Tensor, x_feat: torch.Tensor):
        fusion_feat = torch.cat((z_feat, x_feat), dim=1)
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat)
        fusion_feat = self.norm(fusion_feat)
        
        return fusion_feat[:, z_feat.shape[1]:, :]
    
    def _correlation(self,fusion_feat):
        fusion_feat_T = fusion_feat.transpose(1, 2)
        result = torch.bmm(fusion_feat, fusion_feat_T)
        return result
    
    def _orthogonalize_vectors(self,x, y):
        """
        Orthogonalize vector y with respect to vector x, treating each element in the last dimension as a high-dimensional vector.
        
        Parameters:
        x (torch.Tensor): Input vector with shape (b, n, c)
        y (torch.Tensor): Input vector with shape (b, m, c)
        
        Returns:
        y_orth (torch.Tensor): Orthogonalized vector y with shape (b, m, c)
        """
        # Get the shapes of the input vectors
        b, n, c = x.shape
        _, m, _ = y.shape
        
        # Ensure b is 1
        if b != 1:
            raise ValueError("Batch size b must be 1.")
        
        # Reshape x and y to (n, c) and (m, c) for easier computation
        x = x.squeeze(0)  # Shape: (n, c)
        y = y.squeeze(0)  # Shape: (m, c)
        xn = x / torch.norm(x, dim=-1, keepdim=True)
        yn = y / torch.norm(y, dim=-1, keepdim=True)
        xn = xn.to(dtype=torch.float64)
        yn = yn.to(dtype=torch.float64)
        # Compute the projection of y onto x for each vector in y
        # Compute the dot product of y with each vector in x
        dot_product_yx = torch.matmul(yn, xn.T)  # Shape: (m, n)
        
        # Compute the dot product of x with itself
        dot_product_xx = torch.matmul(xn, xn.T)  # Shape: (n, n)
        
        # Compute the projection matrix
        projection_matrix = torch.matmul(dot_product_yx, torch.inverse(dot_product_xx))  # Shape: (m, n)
        
        # Compute the projection of y onto the subspace spanned by x
        projection = torch.matmul(projection_matrix, xn)  # Shape: (m, c)
        
        # Subtract the projection from y to get the orthogonal component
        y_orth = y - projection  # Shape: (m, c)
        
        # Reshape back to original shape (b, m, c)
        y_orth = y_orth.unsqueeze(0)  # Shape: (1, m, c)
        
        return y_orth.to(dtype=torch.float32)
    
    
    
    
    def _fusion(self, z_feat: torch.Tensor, x_feat: torch.Tensor,class_list=None):
        fusion_feat = torch.cat((z_feat, x_feat), dim=1)
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat)
        fusion_feat = self.norm(fusion_feat)
        #correlation_map = self._correlation(fusion_feat)
        #correlation_map = correlation_map[:, :z_feat.shape[1], z_feat.shape[1]:]
        #correlation_map=correlation_map.mean(dim=1)
        if class_list is not None:
            class_array = np.array(class_list)
            class_mask = class_array!=0
            class_mask_reverse=class_array==0
            tmp = fusion_feat[:, z_feat.shape[1]:, :]
            tmp[:,class_mask_reverse,:] = self._orthogonalize_vectors(tmp[:,class_mask,:],tmp[:,class_mask_reverse,:])
            fusion_feat[:, z_feat.shape[1]:, :] = tmp
            #print(correlation_map[:,class_mask].shape) #torch.Size([1, 27])
        #print("leo",correlation_map.shape) leo torch.Size([1, 729])
        return fusion_feat[:, z_feat.shape[1]:, :]

    # def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
    #     state_dict = lora_merge_state_dict(self, state_dict)
    #     return super().load_state_dict(state_dict, **kwargs)


def build_lorat(cfg, training=True):
    load_pretrained = cfg.MODEL.LOAD_PRETRAINED  # False
    backbone_build_params = {"name": cfg.MODEL.BACKBONE.TYPE, "acc": "default"}
    backbone = build_dino_v2_backbone(load_pretrained=load_pretrained, **backbone_build_params)

    stride = cfg.MODEL.BACKBONE.STRIDE
    feat_sz_z = int(cfg.DATA.TEMPLATE.SIZE / stride)
    feat_sz_x = int(cfg.DATA.SEARCH.SIZE / stride)

    model = LoRATBaseline_DINOv2(backbone, to_2tuple(feat_sz_z), to_2tuple(feat_sz_x))

    if training:
        ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", "pretrained", cfg.MODEL.PRETRAIN_FILE))
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(ckpt_path), strict=True)
        print(f"missing_keys when load LoRAT whole model: {missing_keys}")
        print(f"unexpected_keys when load LoRAT whole model: {unexpected_keys}")

    return model


if __name__ == '__main__':
    import importlib
    config_module = importlib.import_module("lib.config.%s.config" % 'lorat')
    cfg = config_module.cfg
    config_module.update_config_from_file('')
    lorat = build_lorat(cfg, training=True)
    print(lorat)