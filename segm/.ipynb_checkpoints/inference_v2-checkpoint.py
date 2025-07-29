import click
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import argparse
import segm.utils.torch as ptu
import cv2
from segm.data.utils import STATS
from segm.data.ade20k import ADE20K_CATS_PATH
from segm.data.utils import dataset_cat_description, seg_to_rgb
from segm.model import hmetrics
from segm.model.factory import load_model
from segm.model.utils import inference_binary

def main(args):
    model_path = args.model_path
    input_dir = args.input_dir
    output_dir = args.output_dir
    ptu.set_gpu_mode(True,args.gpu_id)

    model_dir = Path(model_path).parent
    model, variant = load_model(model_path)
    model.to(ptu.device)

    normalization_name = variant["dataset_kwargs"]["normalization"]
    normalization = STATS[normalization_name]
    # cat_names, cat_colors = dataset_cat_description(ADE20K_CATS_PATH)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    val_seg_gt, val_seg_pred, val_g_pred = {}, {}, {}
    list_dir = list(input_dir.iterdir())
    for filename in tqdm(list_dir, ncols=80):
        pil_im = Image.open(filename).copy()
        im = F.pil_to_tensor(pil_im).float() / 255
        ori_shape = im.shape[-2:]
        im = F.resize(im,[1024,1024])
        im = F.normalize(im, normalization["mean"], normalization["std"])
        im = im.to(ptu.device).unsqueeze(0)
        
        logits， g_pred = inference_binary(
            model,
            im,
            window_size=variant["inference_kwargs"]["window_size"],
            window_stride=256,
        )
        seg_map = np.squeeze(logits)
        seg_map = cv2.resize(seg_map,ori_shape[::-1],cv2.INTER_LINEAR)
        name = filename.name[:-4]
        gt = cv2.imread(filename.as_posix().replace('/im/','/gt/').replace('.jpg','.png'),cv2.IMREAD_GRAYSCALE).copy()
        val_seg_gt[name] = gt
        val_seg_pred[name] = seg_map * 255
        
        g_pred = np.squeeze(g_pred)
        g_pred = cv2.resize(g_pred,ori_shape[::-1],cv2.INTER_LINEAR)
        val_g_pred[name] = g_pred
        # cv2.imwrite(output_dir.as_posix() +'/'+ name + '_local.png',seg_map*255)
        # cv2.imwrite(output_dir.as_posix() +'/'+ name + '_global.png',g_pred*255)
    print('evalute the results...')
    maxF, mae = hmetrics.compute_metrics(val_seg_pred, val_seg_gt)
    print('maxF:{:.5f} MAE:{:.5f}'.format(maxF, mae))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--model-path", type=str, default='seg_dual_mask/checkpoint.pth')
    parser.add_argument("--input-dir", "-i", default='../data/DIS5K/DIS-VD/im',type=str, help="folder with input images")
    parser.add_argument("--output-dir", "-o", type=str, default='./res',help="folder with output images")
    parser.add_argument("--gpu_id", default='0', type=str)
    args = parser.parse_args()
    main(args)
