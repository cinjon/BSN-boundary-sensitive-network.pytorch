import torch.nn as nn

from . import inflate


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = nn.Sequential(
        inflate.inflate_conv(downsample2d[0],
                             time_dim=1,
                             time_stride=time_stride,
                             center=True),
        inflate.inflate_batch_norm(downsample2d[1]))

    return downsample3d


class Bottleneck3d(nn.Module):

    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()
        self.stride = bottleneck2d.stride
        self.relu = nn.ReLU(inplace=True)
        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(bottleneck2d.downsample,
                                                 time_stride=1)
        else:
            self.downsample = None

        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1,
                                          time_dim=1,
                                          center=True)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(bottleneck2d.conv2,
                                          time_dim=1,
                                          center=True)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3,
                                          time_dim=1,
                                          center=True)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)

    def forward(self, x):
        residual = self.downsample(x) if self.downsample is not None else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)

        return out


def inflate_reslayer(reslayer2d):
    reslayer3d = []
    for layer2d in reslayer2d:
        reslayer3d.append(Bottleneck3d(layer2d))

    return nn.Sequential(*reslayer3d)


class InflatedResNet(nn.Module):

    def __init__(self,
                 resnet2d,
                 num_frames=16,
                 num_classes=1000,
                 conv_class=False):
        super(InflatedResNet, self).__init__()
        self.conv_class = conv_class
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = inflate.inflate_conv(resnet2d.conv1,
                                          time_dim=1,
                                          center=True)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                     stride=(1, 2, 2),
                                     padding=(0, 1, 1))

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        #self.layer4 = inflate_reslayer(resnet2d.layer4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
