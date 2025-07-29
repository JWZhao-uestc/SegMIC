import torch
import numpy as np
from tqdm import tqdm
def mae_torch(pred,gt):
    h,w = gt.shape[0:2]
    sumError = torch.sum(torch.absolute(torch.sub(pred.float(), gt.float())))
    maeError = torch.divide(sumError,float(h)*float(w)*255.0+1e-4)
    return maeError

def f1score_torch(pd,gt):

    # print(gt.shape)
    gtNum = torch.sum((gt>128).float()*1) ## number of ground truth pixels

    pp = pd[gt>128]
    nn = pd[gt<=128]

    pp_hist =torch.histc(pp,bins=255,min=0,max=255)
    nn_hist = torch.histc(nn,bins=255,min=0,max=255)


    pp_hist_flip = torch.flipud(pp_hist)
    nn_hist_flip = torch.flipud(nn_hist)

    pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
    nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

    precision = (pp_hist_flip_cum)/(pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)#torch.divide(pp_hist_flip_cum,torch.sum(torch.sum(pp_hist_flip_cum, nn_hist_flip_cum), 1e-4))
    recall = (pp_hist_flip_cum)/(gtNum + 1e-4)
    f1 = (1+0.3)*precision*recall/(0.3*precision+recall + 1e-4)

    return torch.reshape(precision,(1,precision.shape[0])),torch.reshape(recall,(1,recall.shape[0])),torch.reshape(f1,(1,f1.shape[0]))

def f1_mae_torch(pred, gt):
    import time
    tic = time.time()
    if(len(gt.shape)>2):
        gt = gt[:,:,0]
    pre, rec, f1 = f1score_torch(pred,gt)
    mae = mae_torch(pred,gt)

    return pre.data.numpy(), rec.data.numpy(), f1.data.numpy(), mae.data.numpy()

def compute_metrics(preds, gts):
    num = len(preds)
    mybins = np.arange(0, 256)
    PRE = np.zeros((num, len(mybins) - 1))
    REC = np.zeros((num, len(mybins) - 1))
    F1 = np.zeros((num, len(mybins) - 1))
    MAE = np.zeros((num))
    for i, key in enumerate(preds.keys()):
        pred = preds[key]
        gt = gts[key]
        if pred.max()<=1:
            pred=pred*255
        if gt.max()<=1:
            gt=gt*255
        pred = torch.tensor(pred)
        gt = torch.tensor(gt)
        pre, rec, f1, mae = f1_mae_torch(pred, gt)
        PRE[i, :] = pre
        REC[i, :] = rec
        F1[i, :] = f1
        MAE[i] = mae
    PRE_m = np.mean(PRE, 0)
    REC_m = np.mean(REC, 0)
    f1_m = (1 + 0.3) * PRE_m * REC_m / (0.3 * PRE_m + REC_m + 1e-8)

    return np.amax(f1_m), np.mean(MAE)

def eval_pr(y_pred, y, num):
    h, w = y.shape
    pred = y_pred.expand(num, h, w)
    gt = y.expand(num, h, w)
    thlist = torch.linspace(0, 1 - 1e-10, num).reshape(num, 1)
    mask = thlist.expand(num, h*w).reshape(num, h, w)
    pred_threshold = torch.where(pred >= mask, 1, 0).float()
    tp = torch.sum(pred_threshold * gt, dim=(1,2))
    prec, recall = tp / (torch.sum(pred_threshold, dim=(1,2)) + 1e-20), tp / (torch.sum(gt, dim=(1,2)) + 1e-20)
    return prec, recall

def compute_metrics(preds, gts):
    mae = 0
    f_score = 0
    beta2 = 0.3
    for i, key in enumerate(tqdm(preds.keys())):
        pred = preds[key]
        gt = gts[key]
        if pred.max()>1:
            pred=pred/255
        if gt.max()>1:
            gt=gt/255
        pred = torch.tensor(pred)
        gt = torch.tensor(gt)
        mae += torch.abs(pred-gt).mean()
        prec, recall = eval_pr(pred, gt, 256)
        f_score += (1 + beta2) * prec * recall / (beta2 * prec + recall+1e-10)
    f_score /= len(preds)
    mae /= len(preds)
    return f_score.max().item(), mae.item()
