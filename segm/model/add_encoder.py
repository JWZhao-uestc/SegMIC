import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_

from segm.model.blocks import Block, FeedForward
from segm.model.utils import init_weights


class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x


class AddVisionTransformer(nn.Module):
    def __init__(
        self,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.patch_scale = 16

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.proj_dec = nn.Linear(d_encoder, d_model)
        self.added_encoder_norm = nn.LayerNorm(d_model)
        self.apply(init_weights)
        #trunc_normal_(self.cls_emb, std=0.02)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {"cls_emb"}

    def forward(self, x): 
        #print("x",x.shape)
        for blk in self.blocks:
            x = blk(x)
        x = self.added_encoder_norm(x)
                                  
        return x

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
