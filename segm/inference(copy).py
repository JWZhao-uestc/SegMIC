import click
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import segm.utils.torch as ptu
import cv2
import os
import torch
from segm.data.utils import STATS
from segm.data.ade20k import ADE20K_CATS_PATH
from segm.data.utils import dataset_cat_description, seg_to_rgb
from segm.model import hmetrics, gmetrics
from segm.model.factory import load_model
from segm.model.utils import inference_binary, inference_bce, inference_medsam
from segm.data import transform
import numpy as np
import matplotlib.pyplot as plt



def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)
def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def main(args):
    model_path = args.model_path
    input_dir = args.input_dir
    output_dir = os.path.join(args.output_dir, model_path.split('/')[-2])  
    image_size = args.im_size

    model_dir = Path(model_path).parent
    print(model_path)
    model, variant = load_model(model_path)
    model.cuda()
    model.eval()

    normalization_name = variant["dataset_kwargs"]["normalization"]
    normalization = STATS[normalization_name]
    # cat_names, cat_colors = dataset_cat_description(ADE20K_CATS_PATH)
    
    trans = transform.Compose([
                transform.Resize((image_size, image_size)),
                transform.ToTensor(),
                transform.Normalize(mean=normalization['mean'], std=normalization['std'])
            ])

    print(input_dir, args.db_name)
    input_dir = os.path.join(input_dir, args.db_name, 'Testing/images')
    print(input_dir)
    input_mask_dir = input_dir.replace('images', 'labels')
    
    mask_shuffix = os.listdir(input_mask_dir)[0].split('.')[-1]
    img_shuffix = os.listdir(input_dir)[0].split('.')[-1]
    
    input_dir = Path(input_dir)
    output_dir = os.path.join(output_dir, args.db_name)
    os.makedirs(output_dir, exist_ok=True)
    #output_dir.mkdir(exist_ok=True)
    val_seg_gt, val_seg_pred, val_g_pred = [], [], []
    list_dir = list(input_dir.iterdir())
    ep_total_nums = len(list_dir)
    print(input_dir)

    sam_dice_scores=[]
    sam_iou_scores = []
    ep_sam_dice_scores=[]
    ep_sam_iou_scores = []
    
    iou_standard={
        'JRST-Lung':0.75,
        'JRST-Lung-Left':0.50,
        'JRST-Heart':0.19,
        'CVC-ColonDB':0.735,
        'CVC-ClinicDB':0.905,
        'ETIS-laribPolypDB':0.734,
        'Kvasir-SEG':0.877,
        'isic17':0.90,
        'isic18':0.83,
        'endovis17':0.90,
        'endovis18':0.90,
    }

    for filename in tqdm(list_dir):
        # print(filename.as_posix())
        name = filename.name[:-4]
        pil_im = cv2.imread(filename.as_posix(), cv2.IMREAD_COLOR).copy()
        #cv2.imwrite('a.png', pil_im)
        gt = cv2.imread(filename.as_posix().replace('images','labels').replace(img_shuffix,mask_shuffix),cv2.IMREAD_GRAYSCALE) / 255.
        #cv2.imwrite('b.png', gt*255)
        ori_shape = pil_im.shape[:2]
        im, seg_gt = trans(pil_im, gt.copy())
        im = im.cuda().unsqueeze(0)
        seg_gt = torch.as_tensor(seg_gt, dtype=torch.float, device='cuda:0').unsqueeze(0)

        s_index = np.random.randint(0, ep_total_nums)
        ep_filename = list_dir[s_index]
        ep_pil_im = cv2.imread(ep_filename.as_posix(), cv2.IMREAD_COLOR).copy()
        #cv2.imwrite('aa.png', ep_pil_im)
        ep_gt = cv2.imread(ep_filename.as_posix().replace('images','labels').replace(img_shuffix,mask_shuffix),cv2.IMREAD_GRAYSCALE) / 255.
        #ep_gt = np.zeros((512,512))
        #cv2.imwrite('bb.png', ep_gt*255)
        ep_im, ep_seg_gt = trans(ep_pil_im, ep_gt.copy())
        ep_im = ep_im.cuda().unsqueeze(0)
        ep_seg_gt = torch.as_tensor(ep_seg_gt, dtype=torch.float, device='cuda:0').unsqueeze(0)

        logits, g_pred, cor_map, query_mask_embed_, ep_mask_embed_ = inference_medsam(
            model,
            im,
            ep_im,
            seg_gt[None,:,:,:],
            ep_seg_gt[None,:,:,:],
            window_size=variant["inference_kwargs"]["window_size"],
            window_stride=256,
        )

        seg_map = np.squeeze(logits)
        seg_map = torch.argmax(seg_map, 0).cpu().detach().numpy()
        # seg_map = (seg_map - seg_map.min()) / (seg_map.max()-seg_map.min())
        seg_map = cv2.resize(seg_map,(512,512),cv2.INTER_LINEAR)
        gt = cv2.resize(gt,(512,512),cv2.INTER_LINEAR)
        # name = filename.name[:-4]
        ##### query value
        dice = compute_dice(gt>0, seg_map>0)
        sam_dice_scores.append(dice)
        iou = jaccard(gt>0, seg_map>0)
        sam_iou_scores.append(iou)

        #####
        g_pred = np.squeeze(g_pred)
        g_pred = torch.argmax(g_pred, 0).cpu().detach().numpy()
        ep_seg_map = cv2.resize(g_pred, (512,512), cv2.INTER_LINEAR)
        ep_gt = cv2.resize(ep_gt, (512,512), cv2.INTER_LINEAR)
        ##### ep value
        ep_dice = compute_dice(ep_gt>0, ep_seg_map>0)
        ep_sam_dice_scores.append(ep_dice)
        ep_iou = jaccard(ep_gt>0, ep_seg_map>0)
        ep_sam_iou_scores.append(ep_iou)

        # generated mask embedding
        query_att_map = cv2.resize(query_mask_embed_.cpu().detach().numpy(), (512,512))
        query_img = cv2.resize(pil_im, (512,512))
        #print(query_img.shape)
        query_img_RGB = np.ones((query_img.shape[0], query_img.shape[1], query_img.shape[2]))
        query_img_RGB[:,:,0] = query_img[:,:,2]
        query_img_RGB[:,:,1] = query_img[:,:,1]
        query_img_RGB[:,:,2] = query_img[:,:,0]

        ep_att_map = cv2.resize(ep_mask_embed_.cpu().detach().numpy(), (512,512))
        ep_img = cv2.resize(ep_pil_im, (512,512))
        ep_img_RGB = np.ones((ep_img.shape[0], ep_img.shape[1], ep_img.shape[2]))
        ep_img_RGB[:,:,0] = ep_img[:,:,2]
        ep_img_RGB[:,:,1] = ep_img[:,:,1]
        ep_img_RGB[:,:,2] = ep_img[:,:,0]


        #show cor_map * mask and ori_img in 16 subplos
        #print("cor_map", cor_map.shape)
        gt_32 = cv2.resize(gt, (32,32), interpolation=0)
        gt_32 = np.reshape(gt_32, (1024,1))
        cor_map = cor_map.cpu().detach().numpy()
        cor_mask = np.matmul(cor_map, gt_32) #[12,1024,1]
        #print(cor_mask.shape, cor_map.shape, gt_32.shape)
        cor_mask = np.reshape(cor_mask, (12,32,32))
        fig_cor_mask, axes_cor_mask = plt.subplots(2, 8, figsize=(20, 5))

        axes_cor_mask[0,0].imshow(query_img_RGB.astype('uint8'))#.astype('uint8')
        axes_cor_mask[0,0].set_title('Query Image')
        axes_cor_mask[0,0].axis('off')

        axes_cor_mask[0,1].imshow(query_att_map, cmap = plt.jet(), alpha=0.3, interpolation='nearest')
        axes_cor_mask[0,1].set_title('Mask Embedding')
        axes_cor_mask[0,1].axis('off')

        axes_cor_mask[1,0].imshow(gt)
        axes_cor_mask[1,0].set_title('GT/SOTA={:.2f}'.format(iou_standard[args.db_name]))
        axes_cor_mask[1,0].axis('off')

        axes_cor_mask[1,1].imshow(seg_map)
        axes_cor_mask[1,1].set_title('Pred/IOU={:.2f}'.format(iou))
        axes_cor_mask[1,1].axis('off')
        ind = 0
        for i1 in range(2):
            for i2 in range(6):
                cor_mask_512 = cv2.resize(cor_mask[ind], (512,512), interpolation=1)
                axes_cor_mask[i1,i2+2].imshow(cor_mask_512)#.astype('uint8')
                axes_cor_mask[i1,i2+2].set_title('cor_mask')
                axes_cor_mask[i1,i2+2].axis('off')
                ind += 1
        cor_mask_output_dir = output_dir + '/' + 'cor_mask'
        if not os.path.exists(cor_mask_output_dir):
            os.makedirs(cor_mask_output_dir)
        cor_mask_save_path = cor_mask_output_dir + '/' + name + '_query.png'
        # save figure
        fig_cor_mask.savefig(cor_mask_save_path)
        plt.close(fig_cor_mask)
        #exit()

        # show ground truth and segmentation results in 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0,0].imshow(query_img_RGB.astype('uint8'))
        axes[0,0].set_title('Query Image')
        axes[0,0].axis('off')

        axes[0,1].imshow(query_att_map, cmap = plt.jet(), alpha=0.3, interpolation='nearest')
        axes[0,1].set_title('Generated Query Mask Embedding')
        axes[0,1].axis('off')

        axes[1,0].imshow(gt)
        axes[1,0].set_title('Query Image GT / SOTA={:.2f}'.format(iou_standard[args.db_name]))
        axes[1,0].axis('off')

        axes[1,1].imshow(seg_map)
        axes[1,1].set_title('Query Image Pred / IOU={:.2f}'.format(iou))
        axes[1,1].axis('off')
 
        query_save_path = output_dir + '/' + name + '_query.png'
        # save figure
        fig.savefig(query_save_path)
        # save error fig
        if args.db_name in iou_standard.keys():
            # query
            if iou_standard[args.db_name]-iou >= 0.08:
                error_output = output_dir + '/' + 'query_error'
                if not os.path.exists(error_output):
                    os.makedirs(error_output)
                error_query_save_path = error_output + '/' + name + '_query.png'
                fig.savefig(error_query_save_path)
            # ep
            if iou_standard[args.db_name]-ep_iou >= 0.04:
                error_output = output_dir + '/' + 'ep_error'
                if not os.path.exists(error_output):
                    os.makedirs(error_output)
                error_query_save_path = error_output + '/' + name + '_query.png'
                fig.savefig(error_query_save_path)
        

        # close figure
        plt.close(fig)
        ########################################
        fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
        axes1[0,0].imshow(ep_img_RGB.astype('uint8'))
        axes1[0,0].set_title('EP Image')
        axes1[0,0].axis('off')

        axes1[0,1].imshow(ep_att_map, cmap = plt.jet(), alpha=0.3, interpolation='nearest')
        axes1[0,1].set_title('Generated EP Mask Embedding')
        axes1[0,1].axis('off')

        axes1[1,0].imshow(ep_gt)
        axes1[1,0].set_title('EP Image GT / SOTA={:.2f}'.format(iou_standard[args.db_name]))
        axes1[1,0].axis('off')

        axes1[1,1].imshow(ep_seg_map)
        axes1[1,1].set_title('EP Image Pred / IOU={:.2f}'.format(ep_iou))
        axes1[1,1].axis('off')
 
        query_save_path = output_dir + '/' + name + '_ep.png'
        # save figure
        fig1.savefig(query_save_path)
        # save error_figure 
        if args.db_name in iou_standard.keys():
            if iou_standard[args.db_name]-iou >= 0.08:
                error_output = output_dir + '/' + 'query_error'
                if not os.path.exists(error_output):
                    os.makedirs(error_output)
                error_query_save_path = error_output + '/' + name + '_ep.png'
                fig1.savefig(error_query_save_path)

            if iou_standard[args.db_name]-ep_iou >= 0.04:
                error_output = output_dir + '/' + 'ep_error'
                if not os.path.exists(error_output):
                    os.makedirs(error_output)
                error_query_save_path = error_output + '/' + name + '_ep.png'
                fig1.savefig(error_query_save_path)

        # close figure
        plt.close(fig1)

    print('evalute the results...')
    #maxF, mae = hmetrics.compute_metrics(val_seg_pred, val_seg_gt)
    #gmaxF, gmae = hmetrics.compute_metrics(val_g_pred, val_seg_gt)
    #print('maxF:{:.5f} MAE:{:.5f}| G_maxF:{:.5f} G_MAE:{:.5f}'.format(maxF, mae, gmaxF, gmae))
    print("*****Task_name:", args.db_name)
    print("*****DSC: %.4f" % (np.sum(sam_dice_scores) / len(sam_dice_scores)))
    print("*****IOU: %.4f" % (np.sum(sam_iou_scores) / len(sam_iou_scores)))
    print("*****EP DSC: %.4f" % (np.sum(ep_sam_dice_scores) / len(ep_sam_dice_scores)))
    print("*****EP IOU: %.4f" % (np.sum(ep_sam_iou_scores) / len(ep_sam_iou_scores)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--model-path", type=str, default='../seg_tiny_1024/checkpoint.pth')
    parser.add_argument("--input-dir", "-i", default='../../data/DIS5K/DIS-VD/im',type=str, help="folder with input images")
    parser.add_argument("--output-dir", "-o", type=str, default='../res',help="folder with output images")
    parser.add_argument("--db_name", "-d", type=str,help="which dataset")
    parser.add_argument("--im-size", type=int, default=512,help="folder with output images")
    args = parser.parse_args()
    main(args)
