import torch
import copy


_default_config = {
    'block_chunks': 0,
    'init_values': 1.0e-05,
    'drop_path_uniform': True,
    'img_size': 518
}


def build_dino_v2_backbone(name: str, load_pretrained: bool, **kwargs):
    config = copy.deepcopy(_default_config)
    config.update(kwargs)

    if name == 'ViT-S/14':
        from . import vit_small
        model = vit_small(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'))
    elif name == 'ViT-B/14':
        from . import vit_base
        # from trackit.models.backbone.dinov2 import vit_base  # /home/lz/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth
        model = vit_base(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'))
    elif name == 'ViT-L/14':
        from . import vit_large
        model = vit_large(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'))
    elif name == 'ViT-g/14':
        from . import vit_giant2
        model = vit_giant2(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth'))
    else:
        raise NotImplementedError(f'Unknown DINO v2 model name: {name}')
    return model

if __name__ == '__main__':
    from torch import nn
    kwarg = {'name': 'ViT-B/14', 'acc': 'default'}
    dinov2 = build_dino_v2_backbone(load_pretrained=True, **kwarg)
    x = torch.randn(1, 3, 224, 224)
    # ox = dinov2(x, is_training=True)
    # print(f" ox:{ox}")
    for i_layer, block in enumerate(dinov2.blocks):
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                print(name)