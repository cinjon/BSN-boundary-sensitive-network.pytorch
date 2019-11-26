import copy
import random

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50

from . import resnet as resnet_model
from .. import video_transforms, functional_video


transforms_augment_video = transforms.Compose([
    video_transforms.ToTensorVideo(),
    video_transforms.RandomResizedCropVideo(224),
    video_transforms.RandomHorizontalFlipVideo(),
    video_transforms.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
])
transforms_regular_video = transforms.Compose([
    video_transforms.ToTensorVideo(),
    video_transforms.ResizeVideo((256, 256), interpolation='nearest'),
    video_transforms.CenterCropVideo(224),
    video_transforms.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
])


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
    img_path = str(img_path)
    if img_path.endswith('png'):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    elif img_path.endswith('npy'):
        img = np.load(img_path)
        img = Image.fromarray(img)        
    return img


def img_loading_func(path, do_augment=False):    
    img = load_image(path)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if do_augment:
        transforms_ = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transforms_ = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms_(img)


class Model(nn.Module):

    def __init__(self, opts):
        super(Model, self).__init__()
        # Spatial Feature Encoder φ
        self.opts = opts
        if self.opts['dataset'] != 'gymnasticsfeatures':
            print('Doing resnet', flush=True)
            self.resnet = resnet50(pretrained=not opts['do_random_model'], progress=True)
        else:
            print('NOT DOING resnet', flush=True)
        
    def forward(self, imgs):
        # video feature clip1
        if self.opts['dataset'] == 'gymnasticsfeatures':
            # we are doing TSN features.
            bs, num_videoframes, repr_size = imgs.shape
            imgs = imgs.view(bs * num_videoframes, repr_size)
            return imgs
        
        batch_size, num_videoframes, ch, h, w = imgs.shape
        imgs = imgs.view(batch_size * num_videoframes, ch, h, w)
        # imgs = imgs.contiguous()
        # now imgs is [bs * nf, ch, h, w]
        imgs = self.resnet(imgs)
        # now img_feat is [bs * nf, 1000]
        return imgs

    @staticmethod
    def translate(pretrained):
        return {
            k.replace('module', 'representation_model'): v
            for k, v in pretrained.items()
        }
    
