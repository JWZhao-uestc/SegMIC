import numpy as np

def mae_torch(pred,gt):
    h,w = gt.shape[0:2]
    sumError = np.sum(np.absolute(pred-gt))
    maeError = np.divide(sumError,float(h)*float(w)*255.0+1e-4)

    return maeError

def f1score_torch(pd,gt):

    # print(gt.shape)
    gtNum = np.sum((gt>128).astype(np.float32)*1) ## number of ground truth pixels

    pp = pd[gt>128]
    nn = pd[gt<=128]

    pp_hist,_ =np.histogram(pp,bins=255,range=(0,255))
    nn_hist,_ = np.histogram(nn,bins=255,range=(0,255))


    pp_hist_flip = np.flipud(pp_hist)
    nn_hist_flip = np.flipud(nn_hist)

    pp_hist_flip_cum = np.cumsum(pp_hist_flip, axis=0)
    nn_hist_flip_cum = np.cumsum(nn_hist_flip, axis=0)

    precision = (pp_hist_flip_cum)/(pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)#torch.divide(pp_hist_flip_cum,torch.sum(torch.sum(pp_hist_flip_cum, nn_hist_flip_cum), 1e-4))
    recall = (pp_hist_flip_cum)/(gtNum + 1e-4)
    f1 = (1+0.3)*precision*recall/(0.3*precision+recall + 1e-4)

    return np.reshape(precision,(1,precision.shape[0])),np.reshape(recall,(1,recall.shape[0])),np.reshape(f1,(1,f1.shape[0]))


def f1_mae_torch(pred, gt):
    if(len(gt.shape)>2):
        gt = gt[:,:,0]

    pre, rec, f1 = f1score_torch(pred,gt)
    mae = mae_torch(pred,gt)
    return pre, rec, f1, mae

def compute_metrics(preds, gts):
    num = len(preds)
    mybins = np.arange(0, 256)
    PRE = np.zeros((num, len(mybins) - 1))
    REC = np.zeros((num, len(mybins) - 1))
    F1 = np.zeros((num, len(mybins) - 1))
    MAE = np.zeros((num))
    for i, key in enumerate(preds.keys()):
        pred = preds[key]
        gt =gts[key]
        if pred.max()<=1:
            pred=pred*255
        if gt.max()<=1:
            gt=gt*255
        pre, rec, f1, mae = f1_mae_torch(pred, gt)
        PRE[i, :] = pre
        REC[i, :] = rec
        F1[i, :] = f1
        MAE[i] = mae
    PRE_m = np.mean(PRE, 0)
    REC_m = np.mean(REC, 0)
    f1_m = (1 + 0.3) * PRE_m * REC_m / (0.3 * PRE_m + REC_m + 1e-8)

    return np.amax(f1_m), np.mean(MAE)
