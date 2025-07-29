from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer
from segm.model.resnet import backbone_cnn
from segm.model.vit import VisionTransformer
from segm.model.add_encoder import AddVisionTransformer
from segm.model.utils import checkpoint_filter_fn
from segm.model.decoder import DecoderLinear
from segm.model.decoder import MaskTransformer
from segm.model.segmenter import Segmenter
import segm.utils.torch as ptu


@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model

def create_resnet(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    model_cfg.pop("normalization")
    model = backbone_cnn(backbone, **model_cfg)
    return model

def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )
    model_cfg.pop("painter_depth")
    model = VisionTransformer(**model_cfg)
    if backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
    elif "deit" in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        load_custom_pretrained(model, default_cfg)

    return model

def create_vit_fintune(model_cfg, add_encoder_cfg):
    name = add_encoder_cfg.pop("name")
    n_cls = add_encoder_cfg.pop("n_cls")
    dim = model_cfg["d_model"]
    add_encoder_cfg["d_model"] = dim
    add_encoder_cfg["d_encoder"] = dim
    add_encoder_cfg["n_layers"] = 6
    n_heads = dim // 64
    add_encoder_cfg["n_heads"] = n_heads
    add_encoder_cfg["d_ff"] = 4 * dim
    add_encoder_cfg["drop_path_rate"] = model_cfg["drop_path_rate"]
    add_encoder_cfg["dropout"] = model_cfg["dropout"]
    # model = AddVisionTransformer(
    #     d_encoder = 384,
    #     n_layers = 6,
    #     n_heads = 6,
    #     d_model = 384,
    #     d_ff = 4*384,
    #     drop_path_rate = 0.1,
    #     dropout = 0.0,
    # )
    model = AddVisionTransformer(
        **add_encoder_cfg
    )
    return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


# def create_segmenter(model_cfg):
#     model_cfg = model_cfg.copy()
#     decoder_cfg = model_cfg.pop("decoder")
#     add_encoder_cfg = decoder_cfg.copy()
#     decoder_cfg["n_cls"] = model_cfg["n_cls"]
#     if 'resnet' in model_cfg['backbone']:
#         encoder = create_resnet(model_cfg)
#     else:
#         encoder = create_vit(model_cfg)
#     add_encoder = create_vit_fintune(model_cfg, add_encoder_cfg) ###### zjw add 8.22.10.27
#     decoder = create_decoder(encoder, decoder_cfg)
#     model = Segmenter(encoder, add_encoder, decoder, n_cls=model_cfg["n_cls"])

#     return model

def create_vit_cor_fintune(model_cfg, add_encoder_cfg):
    add_encoder_cfg = add_encoder_cfg.copy()
    name = add_encoder_cfg.pop("name")
    n_cls = add_encoder_cfg.pop("n_cls")
    dim = model_cfg["d_model"]
    add_encoder_cfg["d_model"] = dim
    add_encoder_cfg["d_encoder"] = dim
    add_encoder_cfg["n_layers"] = 3
    n_heads = dim // 64
    add_encoder_cfg["n_heads"] = n_heads
    add_encoder_cfg["d_ff"] = 4 * dim
    add_encoder_cfg["drop_path_rate"] = model_cfg["drop_path_rate"]
    add_encoder_cfg["dropout"] = model_cfg["dropout"]
    
    model = AddVisionTransformer(
        **add_encoder_cfg
    )
    return model

def create_vit_seg_fintune(model_cfg, add_encoder_cfg):
    add_encoder_cfg = add_encoder_cfg.copy()
    name = add_encoder_cfg.pop("name")
    n_cls = add_encoder_cfg.pop("n_cls")
    dim = model_cfg["d_model"]
    add_encoder_cfg["d_model"] = dim
    add_encoder_cfg["d_encoder"] = dim
    add_encoder_cfg["n_layers"] = 3
    n_heads = dim // 64
    add_encoder_cfg["n_heads"] = n_heads
    add_encoder_cfg["d_ff"] = 4 * dim
    add_encoder_cfg["drop_path_rate"] = model_cfg["drop_path_rate"]
    add_encoder_cfg["dropout"] = model_cfg["dropout"]
    
    model = AddVisionTransformer(
        **add_encoder_cfg
    )
    return model

def create_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    add_encoder_cfg = decoder_cfg.copy()
    decoder_cfg["n_cls"] = model_cfg["n_cls"]
    if 'resnet' in model_cfg['backbone']:
        encoder = create_resnet(model_cfg)
    else:
        encoder = create_vit(model_cfg)
    #add_encoder = create_vit_fintune(model_cfg, add_encoder_cfg)
    add_cor_encoder = create_vit_cor_fintune(model_cfg, add_encoder_cfg) ###### zjw add 8.22.10.27
    add_seg_encoder = create_vit_seg_fintune(model_cfg, add_encoder_cfg) ###### zjw add 8.22.10.27
    decoder = create_decoder(encoder, decoder_cfg)
    model = Segmenter(encoder, add_cor_encoder, add_seg_encoder, decoder, n_cls=model_cfg["n_cls"], painter_depth = model_cfg["painter_depth"])
    #model = Segmenter(encoder, add_encoder, decoder, n_cls=model_cfg["n_cls"])
    return model


def load_model(model_path, painter_depth):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]
    net_kwargs["painter_depth"] = painter_depth
    model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location="cpu")
    checkpoint = data["model"]
    # new_checkpoint = {}
    # for k, v in checkpoint.items():
    #     if 'add_seg' not in k and 'add_cor' not in k:
    #         #print(k)
    #         #exit()
    #         new_checkpoint[k] = v

    model.load_state_dict(checkpoint, strict=False) ####

    return model, variant
