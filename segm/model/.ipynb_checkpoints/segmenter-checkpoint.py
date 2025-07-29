import torch
import torch.nn as nn
import torch.nn.functional as F

from segm.model.utils import padding, unpadding
from timm.models.layers import trunc_normal_

class Decoder(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1,stride=1):
        super(Decoder,self).__init__()

        self.conv_combine = nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate,stride=stride),
                                          nn.BatchNorm2d(out_ch),
                                          nn.ReLU(inplace=True))

    def forward(self,x_list):
        x = self.conv_combine(torch.cat(x_list,dim=1))
        return x

class Recovery(nn.Module):
    def __init__(self,in_ch=384,out_ch=1,level=3):
        super(Recovery,self).__init__()
        chns = [int(32 * 2 ** x) for x in range(level)]
        chns = chns[::-1]+[64]
        self.reduce = nn.Sequential(nn.Conv2d(in_ch,chns[0],1),
                                    nn.BatchNorm2d(chns[0]),
                                    nn.ReLU(inplace=True)
                                    )
        self.decoders = nn.ModuleList()
        for i in range(level):
            self.decoders.append(Decoder(chns[i]*2,chns[i+1]))

        self.output = nn.Conv2d(chns[-1],out_ch,kernel_size=1)

    def forward(self,x, x_list):
        x = self.reduce(x)
        for i in range(len(self.decoders)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.decoders[i]([x,x_list[-i-1]])

        x = self.output(x)
        return x

class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.recover = Recovery(self.decoder.d_model,1)

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, gim,lim):
        gH_ori, gW_ori = gim.size(2), gim.size(3)
        lH_ori, lW_ori = lim.size(2), lim.size(3)
        gim = padding(gim, self.patch_size)
        lim = padding(lim, self.patch_size)
        gH, gW = gim.size(2), gim.size(3)
        lH, lW = lim.size(2), lim.size(3)

        x, recover_x = self.encoder(gim, lim, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        gmasks, lmasks, x = self.decoder(x, (gH, gW, lH, lW))

        gmasks = F.interpolate(gmasks, size=(gH, gW), mode='bilinear', align_corners=True)
        gmasks = unpadding(gmasks, (gH_ori, gW_ori))

        lmasks = F.interpolate(lmasks, size=(lH, lW), mode='bilinear', align_corners=True)
        lmasks = unpadding(lmasks, (lH_ori, lW_ori))

        output = self.recover(x,recover_x)

        output = F.interpolate(output, size=(lH, lW), mode='bilinear', align_corners=True)
        output = unpadding(output, (lH_ori, lW_ori))

        return gmasks, lmasks, output

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
