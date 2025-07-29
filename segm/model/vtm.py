import torch.nn as nn
from .common import LayerNorm2d
from .attention import CrossAttention

class VTMMatchingModule(nn.Module):
    def __init__(self, in_dims, out_dims, num_heads, act_layer=nn.GELU):
        super().__init__()
        self.matching = CrossAttention(in_dims, out_dims, out_dims, num_heads=num_heads)
        
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
            nn.Conv2d(b_channel*8, out_dims, kernel_size=1), 
        )
                      #b, 1024, 384
    def forward(self, query_x, ep_x, ep_gt):
        B,N,C = ep_x.size()
        Q = query_x
        K = ep_x
        V = self.mask_downscaling(ep_gt)
        V = V.view(B,C,-1).permute(0,2,1)
        O, cor_map = self.matching(Q, K, V, get_attn_map=True)
        #print("O", O.shape)
        return O, V, cor_map