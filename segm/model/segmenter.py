# add_bone 0608 
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .non_local import Non_local
# from .vtm import VTMMatchingModule
from .painter import Painter
from segm.model.utils import padding, unpadding
from timm.models.layers import trunc_normal_
import cv2
import numpy as np
from .non_local import Non_local
# from .center_loss import CenterLoss

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU(inplace=True))
            lastv = hidden
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class Decoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super(Decoder, self).__init__()

        self.conv_combine = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x_list):
        x = self.conv_combine(torch.cat(x_list, dim=1))
        return x


class Recovery(nn.Module):
    def __init__(self, in_ch=384, out_ch=1, level=1):
        super(Recovery, self).__init__()
        chns = [int(32 * 2 ** x) for x in range(level)]
        chns = chns[::-1] + [64]
        self.reduce = nn.Sequential(nn.Conv2d(in_ch, chns[0], 1),
                                    nn.BatchNorm2d(chns[0]),
                                    nn.ReLU(inplace=True)
                                    )
        self.decoders = nn.ModuleList()
        for i in range(level):
            self.decoders.append(Decoder(chns[i] * 2, chns[i + 1]))

        self.output = nn.Conv2d(chns[-1], out_ch, kernel_size=1)

    def forward(self, x, x_list):
        x = self.reduce(x)
        for i in range(len(self.decoders)):
            x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
            x = self.decoders[i]([x, x_list[-i - 1]])

        x = self.output(x)
        return x


class Continue_Recovery(nn.Module):
    def __init__(self, in_ch=384, out_ch=32):
        super(Continue_Recovery, self).__init__()
        self.reduce = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True)
                                    )
        self.imnet = MLP(in_dim=32 * 9 + 32 + 4, hidden_list=[32])

        self.output = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, stride=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 1, 1),
                                    )

        self.offset_conv = nn.Conv2d(32, 8, 3, padding=1, stride=1)

    def forward(self, feat, x, coord, cell):
        feat = self.reduce(feat)
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        coord = coord.unsqueeze(0).expand(feat.shape[0], -1, -1)
        cell = cell.unsqueeze(0).expand(feat.shape[0], -1, -1)

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 2, *feat.shape[-2:])  #

        ost = self.offset_conv(x)
        b, c, h, w = x.shape
        ost = torch.sigmoid(ost.view(b, -1, h * w).permute(0, 2, 1))*2-1
        x = x.view(b, c, h * w).permute(0, 2, 1)
        oi=0

        preds = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1) + ost[:,:,oi*2:(oi+1)*2],
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                oi+=1
                # q_feat = F.interpolate(feat, scale_factor=8, mode='bilinear', align_corners=True)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, x, rel_coord], dim=-1)
                ####
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)
        preds = torch.cat(preds, dim=-1)
        preds = preds.permute(0, 2, 1).view(b, -1, h, w)
        preds = self.output(preds)
        return preds


class Segmenter(nn.Module):
    def __init__(
            self,
            encoder,
            add_cor_encoder,
            add_seg_encoder,
            decoder,
            n_cls,
            painter_depth
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.add_cor_encoder = add_cor_encoder ### zjw added 8.22.10.27
        # self.add_seg_encoder = add_seg_encoder
        self.decoder = decoder
        #self.correlation = Non_local(self.decoder.d_model, self.decoder.d_model, self.decoder.d_model) #384
        #self.correlation = VTMMatchingModule(in_dims=self.decoder.d_model, out_dims=self.decoder.d_model, num_heads=encoder.n_heads)
        self.painter = Painter( img_size=(1024,512),
                                patch_size=encoder.patch_size,
                                embed_dim=self.encoder.d_model,
                                depth=painter_depth, #4
                                num_heads=self.encoder.n_heads,
                                mlp_ratio=4
                            )
        # self.Non_local = Non_local(in_channels=self.encoder.d_model, 
        #                            embed_dims=self.encoder.d_model,
        #                            out_dims=self.encoder.d_model
        #                            )
        # self.alpha = nn.Parameter(torch.Tensor([0.5]))
        # self.recover = Recovery(self.decoder.d_model,1)
        # self.recover = Continue_Recovery(self.decoder.d_model, 32)
        # self.CenterLoss = CenterLoss(n_cls, encoder.image_size[0], encoder.patch_size)

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, gim, lim, gim_gt, lim_gt, random_mask): #gim:query, lim:example
        # prepare
        gH_ori, gW_ori = gim.size(2), gim.size(3)
        lH_ori, lW_ori = lim.size(2), lim.size(3)
        gim = padding(gim, self.patch_size)
        lim = padding(lim, self.patch_size)
        gH, gW = gim.size(2), gim.size(3)
        lH, lW = lim.size(2), lim.size(3) 
        h, w = gH // self.patch_size, gH // self.patch_size

        # base encoder x12 layers, removed norm layer
        query_x, _ = self.encoder(gim, return_features=True) #[b, 1024+num_tokens, 384]
        ep_x, _ = self.encoder(lim, return_features=True) #[b, 1024+num_tokens, 384]
        b,N,c = query_x.shape
        
        # add_encoder for correlation x3 layers, has norm layer
        query_x_cor = self.add_cor_encoder(query_x) #[b, 1024+1, c]
        ep_x_cor = self.add_cor_encoder(ep_x)

          # remove CLS/DIST tokens for correlation
        if hasattr(self.encoder, 'distilled'):
            num_extra_tokens = 1 + self.encoder.distilled
            query_x = query_x[:, num_extra_tokens:]
            ep_x = ep_x[:, num_extra_tokens:]
            query_x_cor = query_x_cor[:, num_extra_tokens:]
            ep_x_cor = ep_x_cor[:, num_extra_tokens:]
        # # correlation
        # query_mask_embed, ep_mask_embed, cor_map = self.correlation(query_x_cor, ep_x_cor, lim_gt)# [b,N,c] #[b,N,c]
        
        # painter
        latent, query_mask_embed, ep_mask_embed, gen_loss = self.painter(query_x_cor, ep_x_cor, gim_gt, lim_gt, random_mask)
        
        # region
        ###  add example feature to query feature
        # Hp = latent.shape[1] // 2
        # ep_latent = latent[:,:Hp,:,:] #[b,32,32,768]
        # #query_latent = self.alpha * (latent[:,Hp:,:,:]) + (1-self.alpha) * (ep_latent)
        # query_latent = 0.7 * (latent[:,Hp:,:,:]) + (1 - 0.7) * (ep_latent)
        # query_mask_embed = query_latent
        # latent = torch.cat((ep_latent, query_latent), 1)
        # #print(self.alpha)
        ####

        ############ Non_local
        # Hp = latent.shape[1] // 2
        # ep_latent = latent[:,:Hp,:,:] #[b,32,32,768]

        # # ep_mask_32 = lim_gt.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
        # # ep_mask_32 = cv2.resize(ep_mask_32, (32,32),) #[1,1,32,32] -> [1,32,32,1]
        # # ep_mask_32 = torch.from_numpy(ep_mask_32[None,:,:,None]).cuda()
    
        # query_latent = latent[:,Hp:,:,:]
        #                                        #b,32,32,c        b,32,32,c        
        # query_mask_embed, _, _ = self.Non_local(query_latent, ep_latent, ep_latent)
        # latent = torch.cat((ep_latent, query_mask_embed), 1)
        # ep_mask_embed = ep_latent
        ############
        #endregion
        #show mask feature
        query_mask_embed_ = torch.mean(query_mask_embed, dim=-1)
        query_mask_embed_ = query_mask_embed_.view(b,h,w).squeeze()
        ep_mask_embed_= torch.mean(ep_mask_embed, dim=-1)
        ep_mask_embed_ = ep_mask_embed_.view(b,h,w).squeeze()
        query_x_ = torch.mean(query_x, dim=-1)
        query_x_ = query_x_.view(b,h,w).squeeze()
        ep_x_ = torch.mean(ep_x, dim=-1)
        ep_x_ = ep_x_.view(b,h,w).squeeze()
        
        # add_encoder for segmentation x3 layers, has norm layer
        # query_x_seg = self.add_seg_encoder(query_x)
        # ep_x_seg = self.add_seg_encoder(ep_x)
  
          # remove CLS/DIST tokens for correlation
        # if hasattr(self.encoder, 'distilled'):
        #     num_extra_tokens = 1 + self.encoder.distilled
        #     query_x_seg = query_x_seg[:, num_extra_tokens:]
        #     ep_x_seg = ep_x_seg[:, num_extra_tokens:]
  
        # decoder
        #print(latent.view(b,-1,c).shape)
        decoder_mask = self.decoder(latent.view(b,-1,c), (2*gH, gW)) #[b,n_cls,32,32]
        
        query_masks = decoder_mask[:,:,h:,:]
        ep_masks = decoder_mask[:,:,:h,:]
        #print(query_masks.shape)
        query_masks = F.interpolate(query_masks, size=(gH, gW), mode='bilinear', align_corners=True)
        query_masks = unpadding(query_masks, (gH_ori, gW_ori))

        ep_masks = F.interpolate(ep_masks, size=(lH, lW), mode='bilinear', align_corners=True)
        ep_masks = unpadding(ep_masks, (lH_ori, lW_ori))

        return query_masks, ep_masks, query_x_, ep_x_, query_mask_embed_, ep_mask_embed_, gen_loss

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)