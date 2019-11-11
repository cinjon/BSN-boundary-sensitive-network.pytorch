import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self,
                 inchannel,
                 outchannel,
                 stride=1,
                 kernel_size=3,
                 activation=F.relu):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,
                      outchannel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=(kernel_size - 1) // 2,
                      bias=False), nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,
                      outchannel,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=(kernel_size - 1) // 2,
                      bias=False), nn.BatchNorm2d(outchannel))
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel,
                          outchannel,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(outchannel))
        self.activation = activation

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Representation(nn.Module):

    def __init__(self, opts):
        super(Representation, self).__init__()
        self.opts = opts
        # channels = 64 # 128
        # self.repr_conv1 = nn.Sequential(
        #     nn.Conv2d(512, channels, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(),
        # )
        # self.repr_layer1 = self.make_layer(ResidualBlock, channels, channels, 2, stride=2)
        # self.repr_layer2 = self.make_layer(ResidualBlock, channels, channels, 2, stride=2)
        self.fc_layer = nn.Linear(1024, 400)
        # if opts['dataset'] == 'gymnastics':
        #     self.fc_layer = nn.Linear(2048, 400)
        # elif opts['dataset'] == 'thumosimages':
        #     self.fc_layer = nn.Linear(2048, 400)
        # elif opts['dataset'] == 'activitynet':
        #     self.fc_layer = nn.Linear(1024, 400)

    def forward(self, representation):
        # out = self.repr_conv1(representation)
        # # thumosimages shape is [bs*nf, 128, 29, 16]
        # out = self.repr_layer1(out)
        # # thumosimages shape is [bs*nf, 64, 15, 8]
        # out = self.repr_layer2(out)
        # # thumosimages shape is [bs*nf, 64, 8, 4]
        out = representation
        out = out.view(out.shape[0], -1)
        out = self.fc_layer(out)
        return out

    # def make_layer(self, block, initial_in, out_channels, num_blocks, stride):
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     in_channels = initial_in
    #     for stride in strides:
    #         layers.append(block(in_channels, out_channels, stride))
    #         in_channels = out_channels
    #     return nn.Sequential(*layers)
