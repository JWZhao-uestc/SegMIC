import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Non_local(nn.Module):
    def __init__(self, in_channels, embed_dims, out_dims, act_layer = nn.GELU) -> None:
        super(Non_local, self).__init__()
        self.geta = conv_nd(2, in_channels, embed_dims, 1, stride = 1, padding=0)
        self.theta = conv_nd(2, in_channels, embed_dims, 1, stride = 1, padding=0)
        self.alpha = conv_nd(2, in_channels, embed_dims, 1, stride = 1, padding=0)
        self.out_conv = conv_nd(2, in_channels, embed_dims, 1, stride = 1, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        #self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(1) # zjw added 7.20/00:13
        self.acti = nn.LeakyReLU(0.1, inplace=True) # zjw added 7.20/00:13
        self.init_weights()
        
    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.geta, self.theta, self.alpha]:
            nn.init.normal_(m.weight.data, std=std)
        if zeros_init:
            nn.init.normal_(self.out_conv.weight.data, 0)
        else:
            nn.init.normal_(self.out_conv.weight.data, std=std)


    def forward(self, query, ep, value): #query: b, 256, 64,64 key_value: b,256,64,64 value: b,1,32,32
        b,h,w,c = ep.shape
        query_ = query
        query = query.permute(0,3,1,2)
        ep = ep.permute(0,3,1,2)
        value = value.permute(0,3,1,2)

        # print("query", query.shape) #
        # print("ep", ep.shape)
        # print("value", value.shape)

        query_g = self.geta(query)
        query_g = query_g.view(b,c,h*w).permute(0,2,1) #[b,32*32,c]
        
        key_theta = self.theta(ep) # b, c, 32, 32
        key_theta = key_theta.view(b, c, -1) # b, c, 32*32
        
        value_phi = self.alpha(value) # b,c,32,32
        value_phi = value_phi.view(b,c,-1).permute(0,2,1) # b,32*32,c
        
        # print("query_g", query_g.shape)
        # print("key_theta", key_theta.shape)
        # print("value_phi", value_phi.shape)
        
        query_g_nrom = query_g / query_g.norm(dim=2, keepdim=True)
        key_theta_norm = key_theta / key_theta.norm(dim=1, keepdim=True)
        cor_map = torch.matmul(query_g_nrom, key_theta_norm)#.view(b, 1, h*w, h*w)# #[b,32*32,32*32]->[b,1,64,64]

        #cor_map = self.acti(self.norm(cor_map)) 1.

        cor_map = self.softmax(cor_map) #2.
        #print('cor_map', cor_map.shape)
        # att_map = ((att_map - att_map.min(1)[0].unsqueeze(1)) /
        #           (att_map.max(1)[0].unsqueeze(1) - att_map.min(1)[0].unsqueeze(1)))
        #cor_map = cor_map.squeeze(1) #[b, 32*32, 32*32]
                              # [b, 32*32, 32*32] x [b, 32*32, c]
        query_output = torch.matmul(cor_map, value_phi) # [b, 32*32, c]

        query_output = self.out_conv(query_output.view(b,h,w,c).permute(0,3,1,2))

        query_output = query_output.permute(0,2,3,1)  #[b,h,w,c]
        #residual
        query_output = query_ + query_output
        
        return query_output, value_phi, cor_map


if __name__ == "__main__":
    a = torch.autograd.Variable(torch.randn((4, 32, 32, 768)))
    b = torch.autograd.Variable(torch.randn((4, 32, 32, 768)))
    non_local = Non_local(768,768,768)
    re,_,_ = non_local(a, b, b)
    #cv2.imwrite("re.png", re[0].squeeze().detach().numpy()*255)
    print(re.shape)