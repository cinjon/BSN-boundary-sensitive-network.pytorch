import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


THUMOS_OUTPUT_DIM = 2560 * (1 + 25 + 49) # 192000
GYMNASTICS_OUTPUT_DIM = 2560 * (1 + 25 + 49)
ACTIVITYNET_OUTPUT_DIM = 2560 * (1 + 25 + 49)

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
        # In opts is information needed for the mapping.
        # In particlar, we need to output a representation of size
        # opt['tem_feat_dim'] from mapping.

        # NOTE: We are going to residual block the shit out of this to get it to a manageable sized representation.
        # The first input is [64, 64, 112]. We need to get it to
        # [opt['tem_feat_dim']]. That's 180m parameters if you do
        # a straight shot linear layer. No good.
        
        self.opts = opts
        channels = 64
        
        self.repr_conv1 = nn.Sequential(
            nn.Conv2d(2560, channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.repr_layer1 = self.make_layer(ResidualBlock, channels, channels, 2, stride=2)
        self.repr_layer2 = self.make_layer(ResidualBlock, channels, channels, 2, stride=2)
        if opts['dataset'] == 'gymnastics':
            self.fc_layer = nn.Linear(2432, 400)
        elif opts['dataset'] == 'thumosimages':
            self.fc_layer = nn.Linear(2432, 400)
        elif opts['dataset'] == 'activitynet':
            self.fc_layer = nn.Linear(2432, 400)

    def forward(self, representation):
        # thumosimages shape representation is [bs*nf, 2560, 75]
        out = representation.unsqueeze(3)
        # bs*nf,2560,75,1
        out = self.repr_conv1(out)
        # thumosimages shape representation is [bs*nf, 64, 38, 1]
        out = self.repr_layer1(out)
        # print('3', out.shape)
        # # thumosimages shape representation is [800, 32, 10, 1]
        out = self.repr_layer2(out)
        # # thumosimages shape representation is [800, 32, 10, 1]
        # print('4', out.shape)
        out = out.view(out.shape[0], -1)
        out = self.fc_layer(out)
        return out

    def make_layer(self, block, initial_in, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        in_channels = initial_in
        for stride in strides:
            layers.append(block(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)
