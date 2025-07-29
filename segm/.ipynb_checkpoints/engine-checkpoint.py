import torch
import math
import numpy as np
from segm.utils.logger import MetricLogger
from segm.model import utils, hmetrics
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu
import torch.nn.functional as F
import cv2

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
    crop_size
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100
    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch['im'].cuda()
        seg_gt = batch['gt'].float().cuda().unsqueeze(1)

        gim = F.interpolate(im,(crop_size,crop_size),mode='bilinear',align_corners=True)
        g_seg_gt = F.interpolate(seg_gt,(crop_size,crop_size),mode='nearest')

        lim,l_seg_gt = random_crop_images(im,seg_gt,(crop_size,crop_size))

        with amp_autocast():
            gseg_pred, lseg_pred, recover_pred = model.forward(gim,lim)
            loss = criterion(gseg_pred, g_seg_gt.squeeze(1).long()) + criterion(lseg_pred, l_seg_gt.squeeze(1).long()) \
                    + criterion_bce(recover_pred, l_seg_gt) * 1.5

        loss_value = loss.item()
        if not math.isfinite(loss_value):
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
            learning_rate=optimizer.param_groups[0]["lr"],
        )

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
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50
    val_g_pred = {}
    val_seg_pred = {}
    model.eval()
    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].cuda()
        ori_shape = batch["ori_size"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["filename"][0]

        with amp_autocast():
            seg_pred, global_pred = utils.inference_bce(
                model_without_ddp,
                im,
                window_size,
                window_stride
            )
        seg_pred = np.squeeze(seg_pred)
        seg_pred = cv2.resize(seg_pred,ori_shape[::-1],cv2.INTER_LINEAR)
        val_seg_pred[filename] = seg_pred
        
        global_pred = np.squeeze(global_pred)
        global_pred = cv2.resize(global_pred,ori_shape[::-1],cv2.INTER_LINEAR)
        val_g_pred[filename] = global_pred

    maxF, mae = hmetrics.compute_metrics(val_seg_pred, val_seg_gt)
    g_maxF, g_mae = hmetrics.compute_metrics(val_g_pred, val_seg_gt)
    scores = {'P_maxF':maxF,'P_mae':mae, 'G_maxF':g_maxF, 'g_mae':g_mae}
    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger
