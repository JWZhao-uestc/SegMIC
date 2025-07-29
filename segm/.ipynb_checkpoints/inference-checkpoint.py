import click
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import segm.utils.torch as ptu
import cv2
from segm.data.utils import STATS
from segm.data.ade20k import ADE20K_CATS_PATH
from segm.data.utils import dataset_cat_description, seg_to_rgb
from segm.model import hmetrics
from segm.model.factory import load_model
from segm.model.utils import inference_binary, inference_bce
from segm.data import transform

def main(args):
    model_path = args.model_path
    input_dir = args.input_dir
    output_dir = args.output_dir
    image_size = args.im_size

    model_dir = Path(model_path).parent
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

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    val_seg_gt, val_seg_pred, val_g_pred = {}, {}, {}
    list_dir = list(input_dir.iterdir())
    for filename in tqdm(list_dir, ncols=80):
        # print(filename.as_posix())
        pil_im = cv2.imread(filename.as_posix(), cv2.IMREAD_COLOR).copy()
        gt = cv2.imread(filename.as_posix().replace('/im/','/gt/').replace('.jpg','.png'),cv2.IMREAD_GRAYSCALE) / 255.
        ori_shape = pil_im.shape[:2]
        im, _ = trans(pil_im, gt.copy())
        im = im.cuda().unsqueeze(0)
        
        logits, g_pred = inference_bce(
            model,
            im,
            window_size=variant["inference_kwargs"]["window_size"],
            window_stride=256,
        )
        seg_map = np.squeeze(logits)
        # seg_map = (seg_map - seg_map.min()) / (seg_map.max()-seg_map.min())
        seg_map = cv2.resize(seg_map,ori_shape[::-1],cv2.INTER_LINEAR)
        name = filename.name[:-4]
        val_seg_gt[name] = gt
        val_seg_pred[name] = seg_map

        g_pred = np.squeeze(g_pred)
        g_pred = cv2.resize(g_pred, ori_shape[::-1], cv2.INTER_LINEAR)
        val_g_pred[name] = g_pred
        cv2.imwrite(output_dir.as_posix() + '/' + name + '_local.png', seg_map * 255)
        cv2.imwrite(output_dir.as_posix() + '/' + name + '_global.png', g_pred * 255)
    print('evalute the results...')
    maxF, mae = hmetrics.compute_metrics(val_seg_pred, val_seg_gt)
    gmaxF, gmae = hmetrics.compute_metrics(val_g_pred, val_seg_gt)
    print('maxF:{:.5f} MAE:{:.5f}| G_maxF:{:.5f} G_MAE:{:.5f}'.format(maxF, mae, gmaxF, gmae))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--model-path", type=str, default='../seg_patch16_384_bce/checkpoint.pth')
    parser.add_argument("--input-dir", "-i", default='../data/DIS5K/DIS-VD/im',type=str, help="folder with input images")
    parser.add_argument("--output-dir", "-o", type=str, default='./res',help="folder with output images")
    parser.add_argument("--im-size", type=int, default=1024,help="folder with output images")
    args = parser.parse_args()
    main(args)
