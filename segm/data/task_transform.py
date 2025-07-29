import random
import math
import numpy as np
import numbers
import collections
import cv2
import torch
import torchvision.transforms.functional as F
from PIL import Image
from albumentations import RandomBrightnessContrast
class Compose(object):
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label, ep_image, ep_label):
        for t in self.segtransform:
            image, label, ep_image, ep_label = t(image, label, ep_image, ep_label)
        return image, label, ep_image, ep_label


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image, label, ep_image, ep_label):
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if len(ep_image.shape) == 2:
            ep_image = np.expand_dims(ep_image, axis=2)

        if not len(label.shape) == 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        ep_image = torch.from_numpy(ep_image.transpose((2, 0, 1)))

        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        if image.max()>1:
            image = image / 255
        if label.max()>1:
            label = label / 255
        
        if not isinstance(ep_image, torch.FloatTensor):
            ep_image = ep_image.float()
        ep_label = torch.from_numpy(ep_label)
        if not isinstance(ep_label, torch.LongTensor):
            ep_label = ep_label.long()
        if ep_image.max()>1:
            ep_image = ep_image / 255
        if ep_label.max()>1:
            ep_label = ep_label / 255

        return image, label, ep_image, ep_label


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label, ep_image, ep_label):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
            for t1, m1 in zip(ep_image, self.mean):
                t1.sub_(m1)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
            for t1, m1, s1 in zip(ep_image, self.mean, self.std):
                t1.sub_(m1).div_(s1)
        return image, label, ep_image, ep_label


class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, image, label, ep_image, ep_label):
        image = cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        label = Image.fromarray(label.astype(np.uint8))
        label = label.resize(self.size[::-1],resample=Image.NEAREST)
        label = np.array(label)
        ep_image = cv2.resize(ep_image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        ep_label = Image.fromarray(ep_label.astype(np.uint8))
        ep_label = ep_label.resize(self.size[::-1],resample=Image.NEAREST)
        ep_label = np.array(ep_label)
        return image, label, ep_image, ep_label


class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=0):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        h, w = label.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)

        h, w = label.shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        return image, label

class RandJoint(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, min_index, max_index):
        self.min_index = min_index
        self.max_index = max_index

    def __call__(self, image, label):
        h, w = label.shape
        split_index = random.randint(self.min_index, self.max_index)

        image = np.hstack((image[:, split_index:],image[:, :split_index]))
        label = np.hstack((label[:, split_index:],label[:, :split_index]))
        return image, label

class RandColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, contrast=0, saturation=0, hue=0):
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(img, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            img = F.adjust_contrast(img, contrast_factor)

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            img = F.adjust_saturation(img, saturation_factor)

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            img = F.adjust_hue(img, hue_factor)
        return img

    def __call__(self, image, label):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.

        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.get_params(image, self.contrast,
                                    self.saturation, self.hue)
        image = np.array(image.copy(), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label

class RandomFlipIntensities(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label, ep_image, ep_label):
        
        if random.random() < self.p:
            image = 255-image
            ep_image = 255 - ep_image

        return image, label, ep_image, ep_label

# class RandomBrightnessContrast(object):
#     def __init__(self, p=0.5, b_up=0.1, b_down=-0.1, c_up=1.2, c_down=0.8):
#         self.p = p
#         self.brightness_up = b_up
#         self.brightness_down = b_down
#         self.contrast_up = c_up
#         self.contrast_down = c_down

#     def __call__(self, image, label):
#         if random.random() < self.p:
#             # blank = np.zeros(image.shape, image.dtype)
#             # white = np.ones(image.shape, image.dtype)*255
#             # rate = random.uniform(self.down, self.up)
#             # if rate<1.0:
#             #     image = cv2.addWeighted(image, rate, blank, 1 - rate, 0)
#             # else:
#             #     image = cv2.addWeighted(image, rate, white, rate - 1, 0)
#             alpha = random.uniform(self.contrast_down, self.contrast_up)
#             beta = random.uniform(self.brightness_down, self.brightness_up)
#             image = cv2.convertScaleAbs(image,alpha,beta)

#         return image, label

class RandomBrightness(object): # 255
    def __init__(self, p=0.5, up=2.0, down=0.5):
        self.p = p
        self.up = up
        self.down = down

    def __call__(self, image, label):
        if random.random() < self.p:
            beta = random.uniform(self.down, self.up)
            image = F.adjust_brightness(Image.fromarray(image), beta)
        return np.asarray(image), label

class RandomContrast(object):
    def __init__(self, p=0.5, c_up=2.0, c_down=0.5):
        self.p = p
        self.up = c_up
        self.down = c_down

    def __call__(self, image, label):
        if random.random() < self.p:
            beta = random.uniform(self.down, self.up)
            image = F.adjust_contrast(Image.fromarray(image), beta)
        return np.asarray(image), label

class RandomFlipLable(object):
    def __init__(self, p=0.25, c_up=2.0, c_down=0.5):
        self.p = p
        

    def __call__(self, image, label, ep_image, ep_label):
        if random.random() < self.p:
            label = 255 - label
            ep_label = 255 - ep_label
        return image, label, ep_image, ep_label

class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label

class RandomNoise(object):
    def __init__(self, noise_type=['gaussian','sp'],p=0.5):
        self.noise_type = noise_type
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            idx = random.randint(0,len(self.noise_type)-1)
            noise_type = self.noise_type[idx]
            '''
            ### Adding Noise ###
            img: image
            cj_type: {gauss: gaussian, sp: salt & pepper}
            '''
            if noise_type == "gauss":
                image = image.copy()
                mean = 0
                st = 0.7
                gauss = np.random.normal(mean, st, image.shape)
                gauss = gauss.astype('uint8')
                image = cv2.add(image, gauss)

            elif noise_type == "sp":
                image = image.copy()
                prob = 0.05
                if len(image.shape) == 2:
                    black = 0
                    white = 255
                else:
                    colorspace = image.shape[2]
                    if colorspace == 3:  # RGB
                        black = np.array([0, 0, 0], dtype='uint8')
                        white = np.array([255, 255, 255], dtype='uint8')
                    else:  # RGBA
                        black = np.array([0, 0, 0, 255], dtype='uint8')
                        white = np.array([255, 255, 255, 255], dtype='uint8')
                probs = np.random.random(image.shape[:2])
                image[probs < (prob / 2)] = black
                image[probs > 1 - (prob / 2)] = white
        return image, label

class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label



if __name__ == '__main__':
    
    func = RandomFlipIntensities()
    func2 = RandomBrightnessContrast(0.0)
    
    img = cv2.imread('/mnt/ZJW/Research_code/Medical_Segment/segmenter_painter_v2/Meddata/not_train/PanDental/Image/NEW_Image_png/test/16/new_16_s06.png')
    mask = cv2.imread('/mnt/ZJW/Research_code/Medical_Segment/segmenter_painter_v2/Meddata/not_train/PanDental/Mask/NEW_Mask_png/test/16/label_all/new_16_s06.png')
    #cv2.imwrite('aaaaa.png', img)
    #cv2.imwrite('bbbbb.png', img)
    
    # bright = RandomBrightness(p=1.0)
    # bright_img,_ = bright(img,img)
    # cv2.imwrite('bright_img.png', bright_img)

    # contrast = RandomContrast(p=1.0)
    # contrast_img,_ = contrast(img,img)
    # cv2.imwrite('contrast_img.png', contrast_img)

    fliplabel = RandomFlipLable()
    _, flipmask,_,_ = fliplabel(img, mask,img,mask)
    cv2.imwrite('flip_mask.png', flipmask)


