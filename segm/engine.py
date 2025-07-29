import torch
import math
import numpy as np
from segm.utils.logger import MetricLogger
from segm.model import utils, hmetrics, gmetrics
from segm.data.utils import IGNORE_LABEL
import torch.nn.functional as F
import cv2
from tqdm import tqdm

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


def random_crop_images(image,label,size=(256,256)):
    h, w = image.shape[-2:]
    new_h, new_w = size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    new_image = image[:, :,top:top + new_h, left:left + new_w]
    new_label = label[:, :,top:top + new_h, left:left + new_w]

    return new_image, new_label

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    crop_size,
    IDS
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 1000 #100
    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for batch in tqdm(logger.log_every(data_loader, print_freq, header)):
        
        # im = batch['im'].cuda(device=IDS[0])
        # seg_gt = batch['gt'].float().cuda(device=IDS[0]).unsqueeze(1)
        # ep_im = batch['ep_im'].cuda(device=IDS[0])
        # ep_seg_gt = batch['ep_gt'].float().cuda(device=IDS[0]).unsqueeze(1)
        im = batch['im'].cuda()
        seg_gt = batch['gt'].float().cuda().unsqueeze(1)
        ep_im = batch['ep_im'].cuda()
        ep_seg_gt = batch['ep_gt'].float().cuda().unsqueeze(1)
        random_mask = batch['random_mask']

        with amp_autocast():
            query_seg_pred, ep_seg_pred, _, _, _, _, gen_loss = model.forward(im, ep_im, seg_gt, ep_seg_gt, random_mask) ####seg_gt
            query_loss = criterion(query_seg_pred, seg_gt.squeeze(1).long())
            ep_loss = criterion(ep_seg_pred, ep_seg_gt.squeeze(1).long())
            loss = 1.5*query_loss + ep_loss# + gen_loss query_loss#
                   #+ criterion_bce(recover_pred, ep_seg_gt) * 1.5

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(loss_value)
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        logger.update(
            loss=loss.item(),
            query_loss=query_loss.item(),
            ep_loss=ep_loss.item(),
            #gen_loss=gen_loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        #break

    return logger


@torch.no_grad()

def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    
    
    # all
    sam_dice_scores=[]
    sam_iou_scores = []
    ep_sam_dice_scores=[]
    ep_sam_iou_scores = []

    for batch in tqdm(data_loader):
        im = batch['im'].cuda()
        gt = batch['gt'].float().cuda().unsqueeze(1)
        ep_im = batch['ep_im'].cuda()
        ep_gt = batch['ep_gt'].float().cuda().unsqueeze(1)
        ones = torch.ones((32,32))
        zeros = torch.zeros((32,32))
        half_mask = torch.cat((zeros,ones),dim=0)

        with amp_autocast():
            logits, g_pred, _, _, _, _, gen_loss = model_without_ddp.forward(  
                                                                            im, 
                                                                            ep_im, 
                                                                            gt, 
                                                                            ep_gt, 
                                                                            half_mask
                                                                            )
            
        seg_map = np.squeeze(logits)
        seg_map = torch.argmax(seg_map, 0).cpu().detach().numpy()
        seg_map = cv2.resize(seg_map,(512,512),cv2.INTER_LINEAR)
        gt = gt.squeeze().cpu().detach().numpy() * 255
        gt = cv2.resize(gt.astype(np.uint8),(512,512),cv2.INTER_LINEAR)
        ##### all labels query value
        dice = compute_dice(gt>0, seg_map>0)
        sam_dice_scores.append(dice)
        iou = jaccard(gt>0, seg_map>0)
        sam_iou_scores.append(iou)
        ###################################
        g_pred = np.squeeze(g_pred)
        g_pred = torch.argmax(g_pred, 0).cpu().detach().numpy()
        ep_seg_map = cv2.resize(g_pred, (512,512), cv2.INTER_LINEAR)
        ep_gt = ep_gt.squeeze().cpu().detach().numpy() * 255
        ep_gt = cv2.resize(ep_gt.astype(np.uint8), (512,512), cv2.INTER_LINEAR)
        ##### all labels ep value
        ep_dice = compute_dice(ep_gt>0, ep_seg_map>0)
        ep_sam_dice_scores.append(ep_dice)
        ep_iou = jaccard(ep_gt>0, ep_seg_map>0)
        ep_sam_iou_scores.append(ep_iou)
        
        ##### all labels query value
        dice = compute_dice(gt>0, seg_map>0)
        sam_dice_scores.append(dice)
        iou = jaccard(gt>0, seg_map>0)
        sam_iou_scores.append(iou)

    print("*****DSC: %.4f" % (np.sum(sam_dice_scores) / len(sam_dice_scores)))
    print("*****IOU: %.4f" % (np.sum(sam_iou_scores) / len(sam_iou_scores)))
    print("*****EP DSC: %.4f" % (np.sum(ep_sam_dice_scores) / len(ep_sam_dice_scores)))
    print("*****EP IOU: %.4f" % (np.sum(ep_sam_iou_scores) / len(ep_sam_iou_scores)))

    return np.sum(sam_dice_scores) / len(sam_dice_scores)



# def evaluate(
#     model,
#     data_loader,
#     val_seg_gt,
#     window_size,
#     window_stride,
#     amp_autocast
# ):
#     model_without_ddp = model
#     if hasattr(model, "module"):
#         model_without_ddp = model.module
#     logger = MetricLogger(delimiter="  ")
#     header = "Eval:"
#     print_freq = 50
#     val_g_pred = {}
#     val_seg_pred = {}
#     model.eval()
#     for batch in logger.log_every(data_loader, print_freq, header):
#         im = batch["im"].cuda()
#         ori_shape = batch["ori_size"]
#         ori_shape = (ori_shape[0].item(), ori_shape[1].item())
#         filename = batch["filename"][0]

#         with amp_autocast():
#             seg_pred, global_pred = utils.inference_bce(
#                 model_without_ddp,
#                 im,
#                 window_size,
#                 window_stride
#             )
#         seg_pred = np.squeeze(seg_pred)
#         seg_pred = cv2.resize(seg_pred,ori_shape[::-1],cv2.INTER_LINEAR)
#         val_seg_pred[filename] = seg_pred
        
#         global_pred = np.squeeze(global_pred)
#         global_pred = cv2.resize(global_pred,ori_shape[::-1],cv2.INTER_LINEAR)
#         val_g_pred[filename] = global_pred

#     maxF, mae = hmetrics.compute_metrics(val_seg_pred, val_seg_gt)
#     g_maxF, g_mae = hmetrics.compute_metrics(val_g_pred, val_seg_gt)
#     scores = {'P_maxF':maxF,'P_mae':mae, 'G_maxF':g_maxF, 'g_mae':g_mae}
#     for k, v in scores.items():
#         logger.update(**{f"{k}": v, "n": 1})

#     return logger
