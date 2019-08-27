"""Module for running corrflow.

To load the images: run r_prep(r_loader(path))

def r_loader(path):
    # This should output RGB.
    img = np.load(path) / 255.
    return img.astype(np.float32)
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return image

def r_prep(image, M):
    h,w = image.shape[0], image.shape[1]
    if w%M != 0: image = image[:,:-(w%M)]
    if h%M != 0: image = image[:-(h%M),]
    return transforms.ToTensor()(image)
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import math
from .submodule import ResNet18

import numpy as np


def img_loading_func(path):
    # This should output RGB.
    img = np.load(path) / 255.
    img = img.astype(np.float32)

    M = 8
    h, w = img.shape[0], img.shape[1]
    if w % M != 0: img = img[:, :-(w % M)]
    if h % M != 0: img = img[:-(h % M),]
    return transforms.ToTensor()(img)


class Model(nn.Module):

    def __init__(self, opts):
        super(Model, self).__init__()

        # Model options
        self.p = 0.3
        self.feature_extraction = ResNet18(3)
        self.post_convolution = nn.Conv2d(256, 64, 3, 1, 1)

    def forward(self, imgs):
        # The return for a [3,256,488] image is [1, 64, 64, 112]
        batch_size, num_videoframes, ch, h, w = imgs.shape
        imgs = imgs.view(batch_size * num_videoframes, ch, h, w)
        imgs = self.post_convolution(self.feature_extraction(imgs))
        return imgs

    def dropout2d(self, arr):  # drop same layers for all images
        if not self.training:
            return arr

        if np.random.random() < self.p:
            return arr

        drop_ch_num = int(np.random.choice(np.arange(1, 2 + 1), 1))
        drop_ch_ind = np.random.choice(np.arange(3), drop_ch_num, replace=False)

        for a in arr:
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= (3 / (3 - drop_ch_num))

        return arr

    @staticmethod
    def translate(pretrained):
        return {
            k.replace('module', 'representation_model'): v
            for k, v in pretrained.items()
        }
