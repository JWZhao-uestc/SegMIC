import timm
import torch
import math
import torch.nn as nn

class backbone_cnn(nn.Module):
    def __init__(self,
                 backbone,
                 image_size,
                 patch_size,
                 n_layers,
                 d_model,
                 n_heads,
                 n_cls,
                 dropout=0.1,
                 drop_path_rate=0.0,
                 distilled=False,
                 ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        self.d_model = d_model
        self.patch_size = patch_size
        self.return_indices = 4
        self.norm = nn.LayerNorm(d_model)

        level = int(math.log2(patch_size)-1)
        in_chns = [64,256,512,1024]
        out_chns = [int(32 * 2 ** x) for x in range(level)]+[d_model]
        self.proj = nn.ModuleList()
        for i in range(len(out_chns)):
            self.proj.append(
                nn.Sequential(nn.Conv2d(in_chns[i], out_chns[i], kernel_size=1, stride=1, padding=0),
                              nn.BatchNorm2d(out_chns[i]),
                              nn.ReLU(inplace=True))
            )

    def forward(self, gim,lim, return_features=True):
        gx = self.backbone(gim)[:self.return_indices]
        lxs = self.backbone(lim)[:self.return_indices]
        gx = self.proj[-1](gx[-1])
        gx = gx.flatten(2).transpose(1, 2)
        for i in range(len(self.proj)):
            lxs[i] = self.proj[i](lxs[i])
        lx = lxs[-1].flatten(2).transpose(1, 2)
        gx = self.norm(gx)
        lx = self.norm(lx)
        x = torch.cat([gx,lx],dim=1)
        return x,lxs[:-1]