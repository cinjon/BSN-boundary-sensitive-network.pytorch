import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import inflated_resnet, resnet


class CycleTime(nn.Module):

    # def __init__(self, pretrained, T=None, detach_network=False):
    def __init__(self, opts):
        super(CycleTime, self).__init__()
        # In opts is information needed for the linear mapping.
        # In aprticular, we need to output a representation of size
        # opt['tem_feat_dim'].
        self.opts = opts
        
        # Spatial Feature Encoder φ
        self.encoderVideo = inflated_resnet.InflatedResNet(
            copy.deepcopy(resnet.resnet50(pretrained=True)))

    def representation(self, imgs):
        # video feature clip1
        img_feat = self.encoderVideo(imgs.transpose(1, 2))
        # 2,8,3,320,320 ... 2,1024,8,40,40
        # i.e. the encoder takes the batch of ... 8 images (i think?)
        # each of which are 3x320x320 and converts them into 1024x40x40.
        print(imgs.shape, img_feat.shape)
        return img_feat

    def linear_mapping(self, representation):
        return representation
        
    def forward(self, imgs):
        raise NotImplementedError
