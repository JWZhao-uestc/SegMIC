#import click
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
import glob



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
    painter_depth = args.painter_depth
    output_dir_upper = args.output_dir + model_path.split('checkpoint')[-1].split('.')[0] + '_binary_abla'
    output_dir = os.path.join(output_dir_upper, model_path.split('/')[-2])   
    image_size = args.im_size
    task = args.db_name

    model_dir = Path(model_path).parent
    print(model_path)
    model, variant = load_model(model_path, painter_depth)
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

    #input_mask_dir = os.path.join(input_dir, args.db_name, 'Mask/Mask_png')
    input_mask_dir = os.path.join(input_dir, args.db_name, 'Mask/NEW_Mask_png/test') #support
    #print('input_mask_dir', input_mask_dir)
    input_image_dir = input_mask_dir.replace('Mask', 'Image')
    #print('input_image_dir', input_image_dir)
    
    mask_shuffix = os.listdir(input_mask_dir)[0].split('.')[-1]
    img_shuffix = os.listdir(input_dir)[0].split('.')[-1]
    
    input_dir = Path(input_dir)
    output_dir = os.path.join(output_dir, args.db_name)
    print('output_dir', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    #output_dir.mkdir(exist_ok=True)
    val_seg_gt, val_seg_pred, val_g_pred = [], [], []
    #list_dir = list(input_dir.iterdir())
    #ep_total_nums = len(list_dir)
    #print(input_dir)

    sam_dice_scores=[]
    sam_iou_scores = []
    ep_sam_dice_scores=[]
    ep_sam_iou_scores = []
    
    for patient in tqdm(os.listdir(input_mask_dir)):
        # total_label = len(os.listdir(os.path.join(input_mask_dir, patient))) - 1
        # task_label = np.arange(int((total_label / 3)*2)+1, total_label+1,1)
        task_label = os.listdir(os.path.join(input_mask_dir, patient))
        #task_label = [1]
        for label in task_label:
            label_path = os.path.join(input_mask_dir, patient, label)
            #print('label_path', label_path)
            
            list_dir = os.listdir(label_path)

            all_label_x = glob.glob(os.path.join(input_mask_dir.replace('test', 'train'), '*', label, '*.png')) + \
                          glob.glob(os.path.join(input_mask_dir.replace('test', 'train'), '*', label, '*.PNG'))   #support sets
            #ep_all_label_x
            for filename in list_dir:
                # print(filename.as_posix())
                #print('filename', filename)
                
                name = filename[:-4]
                
                img_path = os.path.join(input_image_dir, patient, filename)
                mask_path = os.path.join(label_path, filename)
                #print('img_path', img_path)
                #print('mask_path', mask_path)
                if not os.path.exists(img_path):
                    print("There is no image ", img_path)
                    continue
                if not os.path.exists(mask_path):
                    print("There is no image ", mask_path)
                    continue
                
                pil_im = cv2.imread(img_path)
                #pil_im = np.expand_dims(pil_im,2).repeat(3,-1)
                #cv2.imwrite('a.png', pil_im)
                
                gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                gt[gt > 255//2] = 255
                gt = gt / 255.
                #cv2.imwrite('b.png', gt*255)
                if 'SCD' in args.db_name:
                    pil_im = np.rot90(pil_im)
                    gt = np.rot90(gt)
                ori_shape = pil_im.shape[:2]
                im, seg_gt = trans(pil_im, gt.copy())
                im = im.cuda().unsqueeze(0)
                seg_gt = torch.as_tensor(seg_gt, dtype=torch.float, device='cuda:0').unsqueeze(0)

                
                ep_total_nums = len(all_label_x)
                #print(ep_total_nums)
                if 'WBC' in args.db_name:
                    flag = 1
                    while flag:
                        s_index = np.random.randint(0, ep_total_nums)
                        ep_mask_path = all_label_x[s_index]
                        ep_from_dataset = ep_mask_path.split('/')[-1].split('_')[0]
                        q_from_dataset = img_path.split('/')[-1].split('_')[0]
                        #print(ep_from_dataset, q_from_dataset)
                        if ep_from_dataset == q_from_dataset:
                            flag=0
                else:
                    s_index = np.random.randint(0, ep_total_nums)
                    ep_mask_path = all_label_x[s_index]

                label_x = ep_mask_path.split('/')[-2]
                ep_img_path = ep_mask_path.replace('Mask', 'Image').replace(label_x+'/', '')
                ep_file_name = ep_mask_path.split('/')[-1]

                #print('ep_img_path', ep_img_path)
                #print('ep_mask_path', ep_mask_path)
                if not os.path.exists(ep_img_path):
                    print("There is no image ", ep_img_path)
                    continue
                if not os.path.exists(ep_mask_path):
                    print("There is no image ", ep_mask_path)
                    continue

                ep_pil_im = cv2.imread(ep_img_path)
                #ep_pil_im = np.expand_dims(ep_pil_im,2).repeat(3,-1)
                #cv2.imwrite('aa.png', ep_pil_im)
                
                ep_gt = cv2.imread(ep_mask_path, cv2.IMREAD_GRAYSCALE)
                ep_gt[ep_gt > 255//2] = 255
                ep_gt = ep_gt / 255.
                if 'SCD' in args.db_name:
                    ep_pil_im = np.rot90(ep_pil_im)
                    ep_gt = np.rot90(ep_gt)
                
                # # ablation study for example zero ##################!!!!!!!!!!!
                # ep_gt = np.zeros_like(ep_gt)

                #ep_gt = np.zeros((512,512))
                #cv2.imwrite('bb.png', ep_gt*255)
                
                #print(ep_pil_im.shape, ep_gt.shape) #[512,512,3]/[512,512]
                #exit()
                #ep_pil_im = ep_pil_im*np.expand_dims(ep_gt, 2)##########

                ep_im, ep_seg_gt = trans(ep_pil_im, ep_gt.copy())
                ep_im = ep_im.cuda().unsqueeze(0)
                ep_seg_gt = torch.as_tensor(ep_seg_gt, dtype=torch.float, device='cuda:0').unsqueeze(0)
                
                
                ones = torch.ones((32,32))
                zeros = torch.zeros((32,32))
                half_mask = torch.cat((zeros,ones),dim=0)
                
                logits, g_pred, query_x, ep_x, query_mask_embed_, ep_mask_embed_ = inference_medsam(
                    model,
                    im,
                    ep_im,
                    seg_gt[None,:,:,:],
                    ep_seg_gt[None,:,:,:],
                    half_mask,
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
                query_x_map = cv2.resize(query_x.cpu().detach().numpy(), (512,512))
                query_att_map = cv2.resize(query_mask_embed_.cpu().detach().numpy(), (512,512))
                query_img = cv2.resize(pil_im, (512,512))
                #query_img_RGB = np.ones((query_img.shape[0], query_img.shape[1], query_img.shape[2]))
                
                # query_img_RGB[:,:,0] = query_img[:,:,2]
                # query_img_RGB[:,:,1] = query_img[:,:,1]
                # query_img_RGB[:,:,2] = query_img[:,:,0]

                ep_x_map = cv2.resize(ep_x.cpu().detach().numpy(), (512,512))
                ep_att_map = cv2.resize(ep_mask_embed_.cpu().detach().numpy(), (512,512))
                #
                ep_gt_32 = cv2.resize(ep_gt, (32,32), interpolation=0)
                ep_att_map_mask_pool = cv2.resize(ep_gt_32*ep_mask_embed_.cpu().detach().numpy(), (512,512))

                ep_img = cv2.resize(ep_pil_im, (512,512))
                
                #ep_img_RGB = np.ones((ep_img.shape[0], ep_img.shape[1], ep_img.shape[2]))
                
                # ep_img_RGB[:,:,0] = ep_img[:,:,2]
                # ep_img_RGB[:,:,1] = ep_img[:,:,1]
                # ep_img_RGB[:,:,2] = ep_img[:,:,0]

                
                
                # show ground truth and segmentation results in 4 subplots
                # fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # axes[0,0].imshow(query_img_RGB.astype('uint8'))
                # axes[0,0].set_title('Query Image')
                # axes[0,0].axis('off')

                # axes[0,1].imshow(query_att_map, cmap = plt.jet(), alpha=0.3, interpolation='nearest')
                # axes[0,1].set_title('Generated Query Mask Embedding')
                # axes[0,1].axis('off')
                
                # axes[0,2].imshow(query_x_map, cmap = plt.jet(), alpha=0.3, interpolation='nearest')
                # axes[0,2].set_title('Query Transformer Embedding')
                # axes[0,2].axis('off')

                # axes[1,0].imshow(gt)
                # #axes[1,0].set_title('Query Image GT / SOTA={:.2f}'.format(iou_standard[args.db_name]))
                # axes[1,0].axis('off')

                # axes[1,1].imshow(seg_map)
                # axes[1,1].set_title('Query Image Pred / IOU={:.2f}'.format(iou))
                # axes[1,1].axis('off')

                # axes[1,2].imshow(gt)
                # #axes[1,0].set_title('Query Image GT / SOTA={:.2f}'.format(iou_standard[args.db_name]))
                # axes[1,2].axis('off')

                query_output_dir = os.path.join(output_dir, patient, label+'/query')
                if not os.path.exists(query_output_dir):
                    os.makedirs(query_output_dir)
                
                query_img_save_path = query_output_dir + '/' + name + '_q_img.png'
                query_gt_save_path = query_output_dir + '/' + name + '_q_gt.png'
                query_pred_save_path = query_output_dir + '/' + name + '_q_pred.png'
                
                cv2.imwrite(query_img_save_path, query_img)
                cv2.imwrite(query_gt_save_path, gt*255)
                cv2.imwrite(query_pred_save_path, seg_map*255)

                # save figure
                #fig.savefig(query_save_path)
                
                # close figure
                #plt.close(fig)


                ########################################
                # fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
                # axes1[0,0].imshow(ep_img_RGB.astype('uint8'))
                # axes1[0,0].set_title('EP Image')
                # axes1[0,0].axis('off')

                # axes1[0,1].imshow(ep_att_map, cmap = plt.jet(), alpha=0.3, interpolation='nearest')
                # axes1[0,1].set_title('Generated EP Mask Embedding')
                # axes1[0,1].axis('off')

                # axes1[0,2].imshow(ep_x_map, cmap = plt.jet(), alpha=0.3, interpolation='nearest')
                # axes1[0,2].set_title('EP Transformer Embedding')
                # axes1[0,2].axis('off')

                # axes1[1,0].imshow(ep_gt)
                # #axes1[1,0].set_title('EP Image GT / SOTA={:.2f}'.format(iou_standard[args.db_name]))
                # axes1[1,0].axis('off')

                # axes1[1,1].imshow(ep_seg_map)
                # axes1[1,1].set_title('EP Image Pred / IOU={:.2f}'.format(ep_iou))
                # axes1[1,1].axis('off')

                # axes1[1,2].imshow(ep_att_map_mask_pool)
                # #axes1[1,0].set_title('EP Image GT / SOTA={:.2f}'.format(iou_standard[args.db_name]))
                # axes1[1,2].axis('off')
                
                ep_output_dir = os.path.join(output_dir, patient, label+'/ep')
                if not os.path.exists(ep_output_dir):
                    os.makedirs(ep_output_dir)
                
                ep_img_save_path = ep_output_dir + '/' + ep_file_name + '_e_img.png'
                ep_gt_save_path = ep_output_dir + '/' + ep_file_name + '_e_gt.png'
                ep_pred_save_path = ep_output_dir + '/' + ep_file_name + '_e_pred.png'
                
                cv2.imwrite(ep_img_save_path, ep_img)
                cv2.imwrite(ep_gt_save_path, ep_gt*255)
                cv2.imwrite(ep_pred_save_path, ep_seg_map*255)

                # # save figure
                # fig1.savefig(ep_save_path)

                # # close figure
                # plt.close(fig1)

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
    parser.add_argument("--painter_depth", default=4, type=int)
    parser.add_argument("--model-path", type=str, default='../seg_tiny_1024/checkpoint.pth')
    parser.add_argument("--input-dir", "-i", default='../../data/DIS5K/DIS-VD/im',type=str, help="folder with input images")
    parser.add_argument("--output-dir", "-o", type=str, default='../res',help="folder with output images")
    parser.add_argument("--db_name", "-d", type=str,help="which dataset")
    parser.add_argument("--im-size", type=int, default=512,help="folder with output images")
    args = parser.parse_args()
    main(args)
