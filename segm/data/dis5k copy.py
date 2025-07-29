from pathlib import Path
import torch
from torch.utils.data import Dataset
from segm.data import utils
from PIL import Image
import cv2
from segm.config import dataset_dir
from segm.data import transform
from mmcv.utils import Config
PASCAL_CONTEXT_CONFIG_PATH = Path(__file__).parent / "config" / "dis5k.py"
PASCAL_CONTEXT_CATS_PATH = Path(__file__).parent / "config" / "dis5k.yml"


class DIS5KDataset(Dataset):
    def __init__(self, data_root, image_size, crop_size, split, normalization, **kwargs):
        super().__init__()
        self.names, self.colors = utils.dataset_cat_description(
            PASCAL_CONTEXT_CATS_PATH
        )

        self.n_cls = 2
        self.ignore_label = 255
        self.reduce_zero_label = False
        self.split = split
        self.image_size = image_size
        self.crop_size = crop_size
        self.data_root = data_root

        config = Config.fromfile(PASCAL_CONTEXT_CONFIG_PATH)
        self.ratio = config.max_ratio
        self.normalization = utils.STATS[normalization].copy()
        self.config = self.update_default_config(config)
        data_root = self.config.data[split]['data_root'].as_posix()
        filename = self.config.data[split]['split']
        self.imglist = self.get_imglist(data_root+'/'+filename)
        if split == 'train':
            trans = transform.Compose([
                transform.Resize((image_size, image_size)),
                # transform.RandRotate([-30,30],padding=list(self.normalization['mean']), ignore_label=0),
                transform.RandomHorizontalFlip(),
                transform.RandomVerticalFlip(),
                transform.ToTensor(),
                transform.Normalize(mean=self.normalization['mean'], std=self.normalization['std'])
            ])
        elif split == 'val':
            trans = transform.Compose([
                transform.Resize((image_size, image_size)),
                transform.ToTensor(),
                transform.Normalize(mean=self.normalization['mean'], std=self.normalization['std'])
            ])
        else:
            trans = None
        self.transform = trans

    def get_imglist(self, path):
        img_list = []
        with open(path,'r') as f:
            lines = f.readlines()
        for imgname in lines:
            img_list.append(imgname.strip())
        return img_list

    def update_root_config(self, config):

        train_splits = ["train", "trainval"]
        if self.split in train_splits:
            config_pipeline = getattr(config, f"train_pipeline")
        else:
            config_pipeline = getattr(config, f"{self.split}_pipeline")

        img_scale = (self.ratio * self.image_size, self.image_size)
        if self.split not in train_splits:
            assert config_pipeline[1]["type"] == "MultiScaleFlipAug"
            config_pipeline = config_pipeline[1]["transforms"]
        for i, op in enumerate(config_pipeline):
            op_type = op["type"]
            if op_type == "Resize":
                op["img_scale"] = img_scale
            elif op_type == "RandomCrop":
                op["crop_size"] = (
                    self.crop_size,
                    self.crop_size,
                )
            elif op_type == "Normalize":
                op["mean"] = self.normalization["mean"]
                op["std"] = self.normalization["std"]
            elif op_type == "Pad":
                op["size"] = (self.crop_size, self.crop_size)
            config_pipeline[i] = op
        if self.split == "train":
            config.data.train.pipeline = config_pipeline
        elif self.split == "trainval":
            config.data.trainval.pipeline = config_pipeline
        elif self.split == "val":
            config.data.val.pipeline[1]["img_scale"] = img_scale
            config.data.val.pipeline[1]["transforms"] = config_pipeline
        elif self.split == "test":
            config.data.test.pipeline[1]["img_scale"] = img_scale
            config.data.test.pipeline[1]["transforms"] = config_pipeline
            config.data.test.test_mode = True
        else:
            raise ValueError(f"Unknown split: {self.split}")
        return config

    def update_default_config(self, config):
        if self.data_root is None:
            root_dir = '../../data'
        else:
            root_dir = self.data_root
        path = Path(root_dir) / "DIS5K"
        config.data_root = path
        if self.split == "train":
            config.data.train.data_root = path / "DIS-TR/"
        elif self.split == "val":
            config.data.val.data_root = path / "DIS-VD/"
        elif self.split == "test":
            raise ValueError("Test split is not valid for Pascal Context dataset")
        config = self.update_root_config(config)
        return config

    def test_post_process(self, labels):
        return labels

    def get_gt_seg_maps(self):
        gt_seg_maps = {}
        for filename in self.imglist:
            label_path = self.config.data[self.split]['data_root'].as_posix() + '/' + self.config.data[self.split][
                'ann_dir'] + '/' + filename + '.png'
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            gt_seg_maps[filename] = label / 255
        return gt_seg_maps

    def __getitem__(self, idx):
        imgname = self.imglist[idx]
        image_path = self.config.data[self.split]['data_root'].as_posix()+'/'+self.config.data[self.split]['img_dir']+'/'+imgname+'.jpg'

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        org_size = image.shape
        label_path = self.config.data[self.split]['data_root'].as_posix() + '/' + self.config.data[self.split][
            'ann_dir'] + '/' + imgname + '.png'
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return {'im':image, 'gt':label, 'filename':imgname,'ori_size':org_size}

    def __len__(self):
        return len(self.imglist)

    @property
    def unwrapped(self):
        return self

    def set_epoch(self, epoch):
        pass

    def get_diagnostics(self, logger):
        pass

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return
