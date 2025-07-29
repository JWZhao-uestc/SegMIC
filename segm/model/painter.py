from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath, trunc_normal_
from .common import LayerNorm2d
import math

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
            )
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x

class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).permute(0,2,3,1)
        return x

class Painter(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(
             self,
             img_size=(1024,512),
             patch_size=16,
             embed_dim=384,
             depth=4,
             num_heads=6,
             mlp_ratio=4.,
             qkv_bias=True,
             drop_path_rate=0.,
             norm_layer=nn.LayerNorm,
             act_layer=nn.GELU,
             use_abs_pos=True,
             use_rel_pos=False,
             rel_pos_zero_init=True,
             window_size=0,
             window_block_indexes=(),
             residual_block_indexes=(),
             use_act_checkpoint=False,
             pretrain_img_size=224,
             pretrain_use_cls_token=True,
             out_feature="last_feat",
             loss_func="smoothl1",
             ):
        super().__init__()

        # --------------------------------------------------------------------------
        # self.patch_embed = PatchEmbed(
        #     image_size=(512,512),
        #     patch_size=patch_size,
        #     embed_dim=embed_dim,
        #     channels=1,
        # )
        
        b_channel = 16
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, b_channel, kernel_size=2, stride=2),
            LayerNorm2d(b_channel),
            act_layer(),
            nn.Conv2d(b_channel, b_channel*2, kernel_size=2, stride=2),
            LayerNorm2d(b_channel*2),
            act_layer(),
            nn.Conv2d(b_channel*2, b_channel*4, kernel_size=2, stride=2),
            LayerNorm2d(b_channel*4),
            act_layer(),
            nn.Conv2d(b_channel*4, b_channel*8, kernel_size=2, stride=2),
            LayerNorm2d(b_channel*8),
            act_layer(),
            nn.Conv2d(b_channel*8, embed_dim, kernel_size=1), 
        )



        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_x = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_y = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.loss_func = loss_func


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size[0] // patch_size, img_size[1] // patch_size),
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self.norm = norm_layer(embed_dim)

        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.segment_token_x, std=.02)
        torch.nn.init.normal_(self.segment_token_y, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def forward_encoder(self, query_f, ep_f, query_label_f, ep_label_f, bool_masked_pos):
        
        # embed patches
        batch_size, Hp, Wp, _ = ep_label_f.size()
        
        query_f = query_f.reshape(batch_size, Hp, Wp, -1)
        ep_f = ep_f.reshape(batch_size, Hp, Wp, -1)

        imgs = torch.cat((ep_f, query_f), dim=1)#[B,2Hp,Wp,c]
        tgts = torch.cat((ep_label_f, query_label_f), dim=1)#[B,2Hp,Wp,c]

        mask_token = self.mask_token.expand(batch_size, 2*Hp, Wp, -1)
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token).reshape(-1, 2*Hp, Wp, 1)
        tgts = tgts * (1 - w) + mask_token * w
        tgts = tgts + self.segment_token_y
        
        merge_idx = 0
        #print('imgs', imgs.shape)
        #print('tgts', tgts.shape)

        x = torch.cat((imgs, tgts), axis=0)
        # apply Transformer blocks
        out = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            #print('x',x.shape)
            if idx == merge_idx:
                x = (x[:x.shape[0]//2] + x[x.shape[0]//2:]) * 0.5
            # if idx in [5, 11, 17, 23]:
            #     out.append(self.norm(x))
        return self.norm(x)
    
    def forward_loss(self, query_label_pred, tgts):
        pred = query_label_pred
        target = tgts
        if self.loss_func == "l1l2":
            loss = ((pred - target).abs() + (pred - target) ** 2.) * 0.5
        elif self.loss_func == "l1":
            loss = (pred - target).abs()
        elif self.loss_func == "l2":
            loss = (pred - target) ** 2.
        elif self.loss_func == "smoothl1":
            loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
        
        return torch.mean(loss)


                                    #[1,512,512] ##[1,512,512]
    def forward(self, query_f, ep_f, query_label, ep_label, bool_masked_pos, valid=None):
        
        ep_label_f = self.mask_downscaling(ep_label) #[b,c,h,w]
        query_label_f = self.mask_downscaling(query_label) #[b,c,h,w]
        ep_label_f = ep_label_f.permute(0,2,3,1) #[b,h,w,c]
        query_label_f = query_label_f.permute(0,2,3,1) #[b,h,w,c]
        # print('ep_label_f',ep_label_f.shape)
        # print('query_label_f',query_label_f.shape)

        latent = self.forward_encoder(query_f, ep_f, query_label_f, ep_label_f, bool_masked_pos) # [B,1024x2,c]
        
        b,_,Wp,c = latent.shape
        Hp = latent.shape[1] // 2
        ep_latent = latent[:,:Hp,:,:]
        query_latent = latent[:,Hp:,:,:]
      
        query_gen_loss = 0#= self.forward_loss(query_latent, query_label_f)
        # print('query_gen_loss', query_gen_loss)

        ep_latent = ep_latent.view(b,-1,c) #[b,seq_len,c]
        ep_label_f = ep_label_f.view(b,-1,c) #[b,seq_len,c]

        query_latent = query_latent.view(b,-1,c) #[b,seq_len,c]
        query_label_f = query_label_f.view(b,-1,c) #[b,seq_len,c]
        # print('ep_latent',ep_latent.shape)
        # print('query_latent',query_latent.shape)
        # print('ep_label_f',ep_label_f.shape)
        # print('query_label_f',query_label_f.shape)
        return latent, query_latent, ep_latent, query_gen_loss

if __name__ == "__main__":

    painter = Painter()

    input1 = torch.Tensor(torch.randn(5,1024,384))
    input2 = torch.Tensor(torch.randn(5,1024,384))
    input3 = torch.Tensor(torch.randn(5,1,512,512))
    input4 = torch.Tensor(torch.randn(5,1,512,512))
    out = painter(input1, input2, input3, input4)