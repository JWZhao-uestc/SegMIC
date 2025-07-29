import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class CenterLoss(nn.Module):
    def __init__(self, num_class, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
                #[b,N,c] #[b,N]
    def forward(self, ep_mask_embed, ep_mask, query_mask_embed, query_mask):
        b,N,c = ep_mask_embed.shape
        #print(b,N,c)

        # ep = ep_mask.view(b,512,512).detach().cpu().numpy()
        # cv2.imwrite('ep_mask_512.png', ep[0]*255)

        #select negative and positive center from ep_mask_embed
        ep_mask = F.interpolate(ep_mask, (self.image_size // self.patch_size, self.image_size // self.patch_size), mode='nearest')
        ep_mask = ep_mask.view(b,-1)#,1).repeat(1, 1, c)
        
        ep_pos_center = torch.sum(torch.einsum("bNc, bN->bNc", [ep_mask_embed, ep_mask]), 1) / (torch.sum(ep_mask, -1, keepdims=True) + 0.1) 
        rev_ep_mask = 1 - ep_mask
        ep_neg_center = torch.sum(torch.einsum("bNc, bN->bNc", [ep_mask_embed, rev_ep_mask]), 1) / (torch.sum(rev_ep_mask, -1, keepdims=True) + 0.1)
        # print("ep_pos_center", ep_pos_center.shape)
        # print("ep_neg_center", ep_neg_center.shape)
        
        # show
        # ep_mask_show = torch.mean(torch.einsum("bNc, bN->bNc", [ep_mask_embed, ep_mask]), -1)
        # ep = ep_mask_show.view(b,32,32).detach().cpu().numpy()
        # cv2.imwrite('ep_mask_show.png', ep[0]*255)
        # ep_mask_show = torch.mean(torch.einsum("bNc, bN->bNc", [ep_mask_embed, rev_ep_mask]), -1)
        # ep = ep_mask_show.view(b,32,32).detach().cpu().numpy()
        # cv2.imwrite('reverse_ep_mask_show.png', ep[0]*255)
        # ep = ep_mask.view(b,32,32).detach().cpu().numpy()
        # cv2.imwrite('ep_mask.png', ep[0]*255)
        # rev_ep = rev_ep_mask.view(b,32,32).detach().cpu().numpy()
        # cv2.imwrite('rev_ep_mask.png', rev_ep[0]*255)
        
        query_mask = F.interpolate(query_mask, (self.image_size // self.patch_size, self.image_size // self.patch_size), mode='nearest')
        query_mask = query_mask.view(b,-1)
        # print("query_mask_embed", query_mask_embed.shape)
        # print("query_mask", query_mask.shape)
        b_center_loss = 0.0
        b_neg_loss =0.0
        b_pos_loss = 0.0
        for i in range(b):
            pos_index = torch.where(query_mask[i] == 1)
            # print(query_mask.shape, pos_index)
            query_pos_feature = torch.index_select(query_mask_embed[i].unsqueeze(0), dim=1, index=pos_index[0])
            #print('query_pos_feature', query_pos_feature.shape)
    
            neg_index = torch.where(query_mask[i] == 0)
            query_neg_feature = torch.index_select(query_mask_embed[i].unsqueeze(0), dim=1, index=neg_index[0])
            #print('query_neg_feature', query_neg_feature.shape)
            assert query_pos_feature.shape[1] + query_neg_feature.shape[1] == (self.image_size // self.patch_size)**2
            if torch.sum(query_mask[i]) != 0:
                neg_loss = torch.mean(torch.pow((query_neg_feature-ep_neg_center[i].view(1,1,c)), 2))
                pos_loss = torch.mean(torch.pow((query_pos_feature-ep_pos_center[i].view(1,1,c)), 2))
                center_loss = neg_loss + pos_loss
                b_center_loss += center_loss
                b_neg_loss += neg_loss
                b_pos_loss += pos_loss
            else:
                neg_loss = torch.mean(torch.pow((query_neg_feature-ep_neg_center[i].view(1,1,c)), 2))
                pos_loss = 0.0
                center_loss = neg_loss + pos_loss
                b_center_loss += center_loss
                b_neg_loss += neg_loss
                b_pos_loss += pos_loss
        #exit()
        #print('center_loss', center_loss, pos_loss, neg_loss)torch.FloatTensor([0.0]).cuda()
        if type(b_pos_loss) == 'float':
            b_pos_loss = torch.FloatTensor(b_pos_loss).cuda()

        return [b_center_loss / b, b_pos_loss / b, b_neg_loss / b] 