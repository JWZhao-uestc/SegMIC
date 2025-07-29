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
    output_dir_upper = args.output_dir + model_path.split('checkpoint')[-1].split('.')[0] + '_sample_4' #+ '_binary_abla'
    output_dir = os.path.join(output_dir_upper, model_path.split('/')[-2])  
    image_size = args.im_size
    task = args.db_name

    model_dir = Path(model_path).parent
    print('model_path', model_path)
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
    input_mask_dir = os.path.join(input_dir, args.db_name, 'Mask/NEW_Mask_png/test')
    #print('input_mask_dir', input_mask_dir)
    input_image_dir = input_mask_dir.replace('Mask', 'Image')
    #print('input_image_dir', input_image_dir)
    
    mask_shuffix = os.listdir(input_mask_dir)[0].split('.')[-1]
    img_shuffix = os.listdir(input_dir)[0].split('.')[-1]
    
    input_dir = Path(input_dir)
    output_dir = os.path.join(output_dir, args.db_name)
    print('output_dir', output_dir)
    txt_path = os.path.join(output_dir, args.db_name.split('/')[-1]+'.txt')
    os.makedirs(output_dir, exist_ok=True)
    #output_dir.mkdir(exist_ok=True)
    val_seg_gt, val_seg_pred, val_g_pred = [], [], []
    #list_dir = list(input_dir.iterdir())
    #ep_total_nums = len(list_dir)
    #print(input_dir)

    # all
    sam_dice_scores=[]
    sam_iou_scores = []
    ep_sam_dice_scores=[]
    ep_sam_iou_scores = []

    # per_label
    per_label_sam_dice_scores=[]
    per_label_sam_iou_scores = []
    per_label_ep_sam_dice_scores=[]
    per_label_ep_sam_iou_scores = []
    
    total_labels = 0
    f = open(txt_path, 'w')
    for p_idx, patient in tqdm(enumerate(os.listdir(input_mask_dir))):
        # total_label = len(os.listdir(os.path.join(input_mask_dir, patient))) - 1
        # task_label = np.arange(int((total_label / 3)*2)+1, total_label+1,1)
        task_label = os.listdir(os.path.join(input_mask_dir, patient))
        if len(task_label) > total_labels: 
            if total_labels != 0:
                per_label_sam_dice_scores.append([])
                per_label_sam_iou_scores.append([])
                per_label_ep_sam_dice_scores.append([])
                per_label_ep_sam_iou_scores.append([])
            total_labels = len(task_label)

        if p_idx == 0:
            for i in range(len(task_label)):
                per_label_sam_dice_scores.append([])
                per_label_sam_iou_scores.append([])
                per_label_ep_sam_dice_scores.append([])
                per_label_ep_sam_iou_scores.append([])
   
        #print(task_label)
        for label in task_label:
            
            label_idx_str = label.split('_')[-1]
            if label_idx_str == 'all':
                label_idx = total_labels - 1
            else:
                label_idx = int(label_idx_str)-1
 
            if 'LUNA' in args.db_name and label_idx_str != 'all':
                label_idx = label_idx - 2

            label_path = os.path.join(input_mask_dir, patient, label)
            #print('label_path', label_idx, label_path)
            
            list_dir = os.listdir(label_path)

            all_label_x = glob.glob(os.path.join(input_mask_dir.replace('test', 'train'), '*', label, '*.png'))
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

                # print('ep_mask_path',ep_mask_path)
                # print('img_path',img_path)
                #exit()
                
                label_x = ep_mask_path.split('/')[-2]
                ep_img_path = ep_mask_path.replace('Mask', 'Image').replace(label_x+'/', '')
                ep_file_name = ep_mask_path.split('/')[-1]

                if not os.path.exists(ep_img_path):
                    print("There is no image ", ep_img_path)
                    continue
                if not os.path.exists(ep_mask_path):
                    print("There is no image ", ep_mask_path)
                    continue

                ep_pil_im = cv2.imread(ep_img_path)
                #ep_pil_im = np.expand_dims(ep_pil_im,2).repeat(3,-1)
                ep_gt = cv2.imread(ep_mask_path, cv2.IMREAD_GRAYSCALE)
                ep_gt[ep_gt > 255//2] = 255
                ep_gt = ep_gt / 255.
                
                if 'SCD' in args.db_name:
                    ep_pil_im = np.rot90(ep_pil_im)
                    ep_gt = np.rot90(ep_gt)
                #ep_gt = np.zeros((512,512))
                #cv2.imwrite('bb.png', ep_gt*255)
                
                # ablation study for example zero
                # ep_gt = np.zeros_like(ep_gt)

                
                ep_im, ep_seg_gt = trans(ep_pil_im, ep_gt.copy())
                ep_im = ep_im.cuda().unsqueeze(0)
                ep_seg_gt = torch.as_tensor(ep_seg_gt, dtype=torch.float, device='cuda:0').unsqueeze(0)
                
                ones = torch.ones((32,32))
                zeros = torch.zeros((32,32))
                half_mask = torch.cat((zeros,ones),dim=0)

                logits, g_pred, _, _, query_mask_embed_, ep_mask_embed_ = inference_medsam(
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
                ##### all labels query value
                dice = compute_dice(gt>0, seg_map>0)
                sam_dice_scores.append(dice)
                iou = jaccard(gt>0, seg_map>0)
                if np.sum(seg_map>0) == 0 or np.sum(gt>0) == 0:
                    continue
                iou = hd95(seg_map>0, gt>0, 0.5, 2)
                sam_iou_scores.append(iou)
                ##### per label 
                per_label_sam_dice_scores[label_idx].append(dice)
                per_label_sam_iou_scores[label_idx].append(iou)
                #####
                g_pred = np.squeeze(g_pred)
                g_pred = torch.argmax(g_pred, 0).cpu().detach().numpy()
                ep_seg_map = cv2.resize(g_pred, (512,512), cv2.INTER_LINEAR)
                ep_gt = cv2.resize(ep_gt, (512,512), cv2.INTER_LINEAR)
                ##### all labels ep value
                ep_dice = compute_dice(ep_gt>0, ep_seg_map>0)
                ep_sam_dice_scores.append(ep_dice)
                ep_iou = jaccard(ep_gt>0, ep_seg_map>0)
                ep_sam_iou_scores.append(ep_iou)
                ##### per label
                per_label_ep_sam_dice_scores[label_idx].append(ep_dice)
                per_label_ep_sam_iou_scores[label_idx].append(ep_iou)
    
    print('evalute the results...')
    print("*****Task_name:", args.db_name)
    print("#"*30)
    f.writelines("*****Task_name:"+args.db_name + '\n')
    f.writelines("#"*30 + '\n')

    for label_x in range(total_labels):
        dice1 = np.sum(per_label_sam_dice_scores[label_x]) / len(per_label_sam_dice_scores[label_x])
        iou1 = np.sum(per_label_sam_iou_scores[label_x]) / len(per_label_sam_iou_scores[label_x])
        dice2 = np.sum(per_label_ep_sam_dice_scores[label_x]) / len(per_label_ep_sam_dice_scores[label_x])
        iou2 = np.sum(per_label_ep_sam_iou_scores[label_x]) / len(per_label_ep_sam_iou_scores[label_x])
        
        if label_x == total_labels-1:
            label_name = 'label_all'
        else:
            label_name = 'label_'+str(label_x+1)

        print("*****label_name:", label_name)
        print("*****DSC: %.4f" % (dice1))
        print("*****IOU: %.4f" % (iou1))
        print("*****EP DSC: %.4f" % (dice2))
        print("*****EP IOU: %.4f" % (iou2))
        print("#"*30)

        f.writelines("*****label_name: "+label_name + '\n')
        f.writelines("*****DSC: %.4f" % (dice1) + '\n')
        f.writelines("*****IOU: %.4f" % (iou1) + '\n')
        f.writelines("*****EP DSC: %.4f" % (dice2) + '\n')
        f.writelines("*****EP IOU: %.4f" % (iou2) + '\n')
        f.writelines("#"*30 + '\n')

    print("*****All Mixed")
    print("*****DSC: %.4f" % (np.sum(sam_dice_scores) / len(sam_dice_scores)))
    print("*****IOU: %.4f" % (np.sum(sam_iou_scores) / len(sam_iou_scores)))
    print("*****EP DSC: %.4f" % (np.sum(ep_sam_dice_scores) / len(ep_sam_dice_scores)))
    print("*****EP IOU: %.4f" % (np.sum(ep_sam_iou_scores) / len(ep_sam_iou_scores)))
    
    f.writelines("*****DSC: %.4f" % (np.sum(sam_dice_scores) / len(sam_dice_scores)) + '\n')
    f.writelines("*****IOU: %.4f" % (np.sum(sam_iou_scores) / len(sam_iou_scores)) + '\n')
    f.writelines("*****EP DSC: %.4f" % (np.sum(ep_sam_dice_scores) / len(ep_sam_dice_scores)) + '\n')
    f.writelines("*****EP IOU: %.4f" % (np.sum(ep_sam_iou_scores) / len(ep_sam_iou_scores)) + '\n')
    
    f.close()


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
