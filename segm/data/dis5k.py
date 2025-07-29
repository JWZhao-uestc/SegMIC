from pathlib import Path
import torch
from torch.utils.data import Dataset
from segm.data import utils
from PIL import Image
import cv2
from tqdm import tqdm
from segm.config import dataset_dir
from segm.data import transform
from mmcv.utils import Config
from glob import glob
import argparse
import os
import numpy as np
join = os.path.join
PASCAL_CONTEXT_CONFIG_PATH = Path(__file__).parent / "config" / "dis5k.py"
PASCAL_CONTEXT_CATS_PATH = Path(__file__).parent / "config" / "dis5k.yml"

# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--tr_npy_path', type=str, default='./Raw_data/total_8/', help='path to training npy files; two subfolders: npy_gts and npy_embs')
# args = parser.parse_args()

tr_npy_path = "./Raw_data/total_8/"
class DIS5KDataset(Dataset):
    def __init__(self, data_root, db_name, image_size, crop_size, split, normalization, **kwargs):
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
        #self.data_root = data_root
        #print(data_root)
        self.data_root = data_root
        config = Config.fromfile(PASCAL_CONTEXT_CONFIG_PATH)
        self.ratio = config.max_ratio
        self.normalization = utils.STATS[normalization].copy() # nomalization='vit
        #print(self.normalization)
        self.config = self.update_default_config(config)
        #data_root = self.config.data[split]['data_root'].as_posix()
        #filename = self.config.data[split]['split']
        #self.imglist = self.get_imglist(data_root+'/'+filename)
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

        self.gt_path = data_root
        self.embed_path = data_root
        #print(join(self.gt_path,db_name,tmp,'labels','*.jpg'))
        
        if split == 'train':
            tmp = 'Training'
        else:
            tmp = 'Testing'

        if db_name == 'all':
            self.imglist = sorted( glob(join(self.gt_path,'*',tmp,'labels','*.jpg'))+
                                    glob(join(self.gt_path,'*',tmp,'labels','*.png'))+ 
                                    glob(join(self.gt_path,'*',tmp,'labels','*.bmp')) 
                                   ) 
                                
        else:
            self.imglist = sorted(glob(join(self.gt_path,db_name,tmp,'labels','*.jpg'))+
                                    glob(join(self.gt_path,db_name,tmp,'labels','*.png'))+
                                    glob(join(self.gt_path,db_name,tmp,'labels','*.bmp'))
                                   )
                               
        print("total_training_images:", len(self.imglist))
        #print(self.npy_files[0:10])
        #print(self.npy_files[len(self.npy_files)-10 : len(self.npy_files)])
        if split == 'train':
            self.img_files = {
            'endovis17': tr_npy_path + 'endovis17/Training/images',
            'endovis18': tr_npy_path + 'endovis18/Training/images',
            'isic17': tr_npy_path + 'isic17/Training/images',
            'isic18': tr_npy_path + 'isic18/Training/images',
            'CVC-ClinicDB': tr_npy_path + 'CVC-ClinicDB/Training/images',
            'CVC-ColonDB': tr_npy_path + 'CVC-ColonDB/Training/images',
            'ETIS-laribPolypDB': tr_npy_path + 'ETIS-laribPolypDB/Training/images', 
            'Kvasir-SEG': tr_npy_path + 'Kvasir-SEG/Training/images',
            #'DRIVE': tr_npy_path + 'DRIVE/Training/images',
            #'RITE': tr_npy_path + 'RITE/Training/images',
            #'CHASEDB1': tr_npy_path + 'CHASEDB1/Training/images',
            #'HRF': tr_npy_path + 'HRF/Training/images',
            #'STARE': tr_npy_path + 'STARE/Training/images',
            }
            self.mask_files = {
                'endovis17': tr_npy_path + 'endovis17/Training/labels',
                'endovis18': tr_npy_path + 'endovis18/Training/labels',
                'isic17': tr_npy_path + 'isic17/Training/labels',
                'isic18': tr_npy_path + 'isic18/Training/labels',
                'CVC-ClinicDB': tr_npy_path + 'CVC-ClinicDB/Training/labels',
                'CVC-ColonDB': tr_npy_path + 'CVC-ColonDB/Training/labels',
                'ETIS-laribPolypDB': tr_npy_path + 'ETIS-laribPolypDB/Training/labels', 
                'Kvasir-SEG': tr_npy_path + 'Kvasir-SEG/Training/labels',        
                #'DRIVE': tr_npy_path + 'DRIVE/Training/labels',
                #'RITE': tr_npy_path + 'RITE/Training/labels',
                #'CHASEDB1': tr_npy_path + 'CHASEDB1/Training/labels',
                #'HRF': tr_npy_path + 'HRF/Training/labels',
                #'STARE': tr_npy_path + 'STARE/Training/labels', 
            }
        else:
            self.img_files = {
                'endovis17': tr_npy_path + 'endovis17/Testing/images',
                'endovis18': tr_npy_path + 'endovis18/Testing/images',
                'isic17': tr_npy_path + 'isic17/Testing/images',
                'isic18': tr_npy_path + 'isic18/Testing/images',
                'CVC-ClinicDB': tr_npy_path + 'CVC-ClinicDB/Testing/images',
                'CVC-ColonDB': tr_npy_path + 'CVC-ColonDB/Testing/images',
                'ETIS-laribPolypDB': tr_npy_path + 'ETIS-laribPolypDB/Testing/images', 
                'Kvasir-SEG': tr_npy_path + 'Kvasir-SEG/Testing/images',
                #'DRIVE': tr_npy_path + 'DRIVE/Testing/images',
                #'RITE': tr_npy_path + 'RITE/Testing/images',
                #'CHASEDB1': tr_npy_path + 'CHASEDB1/Testing/images',
                #'HRF': tr_npy_path + 'HRF/Testing/images',
                #'STARE': tr_npy_path + 'STARE/Testing/images',
            }
            self.mask_files = {
                'endovis17': tr_npy_path + 'endovis17/Testing/labels',
                'endovis18': tr_npy_path + 'endovis18/Testing/labels',
                'isic17': tr_npy_path + 'isic17/Testing/labels',
                'isic18': tr_npy_path + 'isic18/Testing/labels',
                'CVC-ClinicDB': tr_npy_path + 'CVC-ClinicDB/Testing/labels',
                'CVC-ColonDB': tr_npy_path + 'CVC-ColonDB/Testing/labels',
                'ETIS-laribPolypDB': tr_npy_path + 'ETIS-laribPolypDB/Testing/labels', 
                'Kvasir-SEG': tr_npy_path + 'Kvasir-SEG/Testing/labels',        
                #'DRIVE': tr_npy_path + 'DRIVE/Testing/labels',
                #'RITE': tr_npy_path + 'RITE/Testing/labels',
                #'CHASEDB1': tr_npy_path + 'CHASEDB1/Testing/labels',
                #'HRF': tr_npy_path + 'HRF/Testing/labels',
                #'STARE': tr_npy_path + 'STARE/Testing/labels', 
            }



    def get_imglist(self, path):
        imglist = []
        with open(path,'r') as f:
            lines = f.readlines()
        for imgname in lines:
            imglist.append(imgname.strip())
        return imglist

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
            root_dir = './Raw_data'
        else:
            root_dir = self.data_root
        #print(root_dir)
        path = Path(root_dir)
        #print(path)
        config.data_root = path
        if self.split == "train":
            config.data.train.data_root = path / "total_8"
        elif self.split == "val":
            config.data.val.data_root = path / "All_test_data"
        elif self.split == "test":
            raise ValueError("Test split is not valid for Pascal Context dataset")
        config = self.update_root_config(config)
        return config

    def test_post_process(self, labels):
        return labels

    def get_gt_seg_maps(self):
        gt_seg_maps = {}
        total_num = 9
        for index, filename in enumerate(tqdm(self.imglist)):
             
            label_path = filename.replace('images', 'labels')
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            gt_seg_maps[filename.split('/')[-1]] = label / 255

            if index == total_num:
                break
        return gt_seg_maps

    def __getitem__(self, idx):
        
        image_path = self.imglist[idx].replace('labels', 'images')
        #print("image_path", image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) #
        org_size = image.shape
        label_path = self.imglist[idx]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) #
        label = (label >= (255 // 2)) * 255
        #cv2.imwrite('q_image.png', image)
        #cv2.imwrite('q_label.png', label)
        object_aera = 0
        database_name = self.imglist[idx].split('/')[-4]
        images_path = self.img_files[database_name]
        masks_path = self.mask_files[database_name]
        images_name_sets = os.listdir(images_path)
        ep_total_nums = len(images_name_sets)
        #print("ep_total_nums", ep_total_nums)
        while object_aera == 0 or object_aera == (self.image_size // 16)**2: # all 0 or all 1
            s_index = np.random.randint(0, ep_total_nums)
            #print(total_nums, s_index)
            image_name = images_name_sets[s_index]
            ep_image_path = os.path.join(images_path, image_name)
            ep_label_path = os.path.join(masks_path, image_name)
            ep_image = cv2.imread(ep_image_path, cv2.IMREAD_COLOR)
            ep_label = cv2.imread(ep_label_path, cv2.IMREAD_GRAYSCALE)
            ep_label = (ep_label >= (255 // 2)) * 255
            #cv2.imwrite('ep_image.png', ep_image)
            #cv2.imwrite('ep_label.png', ep_label)
            #ep_count = cv2.resize(ep_label, (self.image_size, self.image_size), interpolation = cv2.INTER_NEAREST)
            ep_count_size_32 = cv2.resize(ep_label, (self.image_size // 16, self.image_size // 16), interpolation = cv2.INTER_NEAREST) #####resize to 32x32, must still has value 1
            object_aera = np.sum(ep_count_size_32 // 255)
            #print("object_aera", object_aera)
            #print("unique", np.unique(ep_label), np.unique(ep_count))
        if self.transform is not None:
            image, label = self.transform(image, label)
            ep_image, ep_label = self.transform(ep_image, ep_label)
            # cv2.imwrite('q_image.png', image.permute(1,2,0).numpy() * 255)
            # cv2.imwrite('q_label.png', label.numpy() * 255)
            # cv2.imwrite('ep_image.png', ep_image.permute(1,2,0).numpy() * 255)
            # cv2.imwrite('ep_label.png', ep_label.numpy() * 255)
            # print("unique", torch.unique(ep_label))
        #exit()
        return {'im':image, 'gt':label, 
                'ep_im':ep_image, 'ep_gt':ep_label, 
                'filename':image_path,'ori_size':org_size}

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
