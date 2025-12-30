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
        from lib.models.modules.dinov2 import vit_small
        model = vit_small(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'))
    elif name == 'ViT-B/14':
        from lib.models.modules.dinov2 import vit_base
        model = vit_base(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'))
    elif name == 'ViT-L/14':
        from lib.models.modules.dinov2 import vit_large
        model = vit_large(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'))
    elif name == 'ViT-g/14':
        from lib.models.modules.dinov2 import vit_giant2
        model = vit_giant2(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth'))


    elif name == 'ViT-S/14-BA':
        from lib.models.modules.dinov2 import vit_small_ba
        model = vit_small_ba(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'), strict=False)
    elif name == 'ViT-B/14-BA':
        from lib.models.modules.dinov2 import vit_base_ba
        model = vit_base_ba(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'), strict=False)
    elif name == 'ViT-L/14-BA':
        from lib.models.modules.dinov2 import vit_large_ba
        model = vit_large_ba(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'), strict=False)
    elif name == 'ViT-g/14-BA':
        from lib.models.modules.dinov2 import vit_giant2_ba
        model = vit_giant2_ba(patch_size=14, **config)
        if load_pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth'), strict=False)
    else:
        raise NotImplementedError(f'Unknown DINO v2 model name: {name}')
    return model

if __name__ == '__main__':
    from torch import nn
    kwarg = {'name': 'ViT-B/14', 'acc': 'default'}
    dinov2 = build_dino_v2_backbone(load_pretrained=False, **kwarg)
    print(dinov2.blocks[0])
    # x = torch.randn(1, 3, 224, 224)
    # for i_layer, block in enumerate(dinov2.blocks):
    #     for name, module in block.named_modules():
    #         if isinstance(module, nn.Linear):
    #             print(name)