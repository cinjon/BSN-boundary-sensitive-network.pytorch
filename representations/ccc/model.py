import copy
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from . import resnet_res4s1
from . import inflated_resnet
from .. import video_transforms, functional_video


transforms_augment_video = transforms.Compose([
    video_transforms.ToTensorVideo(),
    video_transforms.ResizeVideo((256, 256), interpolation='nearest'),
    video_transforms.RandomHorizontalFlipVideo(p=0.5),    
    video_transforms.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
])
transforms_regular_video = transforms.Compose([
    video_transforms.ToTensorVideo(),
    video_transforms.ResizeVideo((256, 256), interpolation='nearest'),
    video_transforms.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
])


def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    img = cv2.resize( img, (owidth, oheight) )
    img = im_to_torch(img)
    return img


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    return img


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def load_image(img_path):
    img = np.load(img_path)
    img = img.astype(np.float32) / 255.0
    img = img.copy()
    return im_to_torch(img)


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)


def img_loading_func(path, do_augment=False):
    imgSize = 256
    
    img = load_image(path)
    ht, wd = img.size(1), img.size(2)
    if ht <= wd:
        ratio  = float(wd) / float(ht)
        # width, height
        img = resize(img, int(imgSize * ratio), imgSize)
    else:
        ratio  = float(ht) / float(wd)
        # width, height
        img = resize(img, imgSize, int(imgSize * ratio))

    if do_augment:
        if random.random() > 0.5:
            img = torch.from_numpy(fliplr(img.numpy())).float()
            
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img = color_normalize(img, mean, std)
    return img


class Model(nn.Module):

    def __init__(self, opts):
        super(Model, self).__init__()
        # Spatial Feature Encoder φ
        resnet = resnet_res4s1.resnet50(pretrained=True)
        self.encoderVideo = inflated_resnet.InflatedResNet(copy.deepcopy(resnet))

        self.afterconv1 = nn.Conv3d(1024, 512, kernel_size=1, bias=False)

        self.spatial_out1 = 30
        self.spatial_out2 = 10
        self.temporal_out = 4

        self.afterconv3_trans = nn.Conv2d(self.spatial_out1 * self.spatial_out1, 128, kernel_size=4, padding=0, bias=False)
        self.afterconv4_trans = nn.Conv2d(128, 64, kernel_size=4, padding=0, bias=False)

        corrdim = 64 * 4 * 4
        corrdim_trans = 64 * 4 * 4
        trans_param_num = 3
        self.linear2 = nn.Linear(corrdim_trans, trans_param_num)


    def forward(self, imgs):
        # video feature clip1
        batch_size, num_videoframes, ch, h, w = imgs.shape
        imgs = imgs.transpose(1, 2)
        # now imgs is [bs, ch, num_videoframes, h, w]
        img_feat = self.encoderVideo(imgs)
        # now img_feat is [bs, 1024, num_videoframes, 57, 32]
        img_feat = self.afterconv1(img_feat)
        # now img_feat is [bs, 512, num_videoframes, 57, 32]
        img_feat = img_feat.transpose(1, 2)
        # --> img_feat is [bs, num_videoframes, 512, 57, 32]
        new_shape = [batch_size * num_videoframes] + list(img_feat.shape[2:])
        img_feat = img_feat.reshape(*new_shape)
        # --> img_feat is [bs*num_videoframes, 512, 57, 32]
        # ... TODO: We should probably be using the normalize here! havent done that yet
        # x_norm = F.normalize(img_feat, p=2, dim=1)
        return img_feat

    @staticmethod
    def translate(pretrained):
        return {
            k.replace('module', 'representation_model'): v
            for k, v in pretrained.items()
        }
