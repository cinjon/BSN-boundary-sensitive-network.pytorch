import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision import models
import pdb
from . import resnets
from torch.nn.functional import unfold
from torch.distributions import Normal
import torch.utils.checkpoint as checkpoint


def has_many_gpus():
    return torch.cuda.device_count() >= 6


class Patchify(object):

    def __init__(self, patch_size, overlap_size):
        self.patch_size = patch_size
        self.overlap_size = self.patch_size - overlap_size

    def __call__(self, x):
        x = x.unsqueeze(0)
        b, c, h, w = x.size()

        # patch up the images
        # (b, c, h, w) -> (b, c*patch_size, L)
        x = unfold(x, kernel_size=self.patch_size, stride=self.overlap_size)

        # (b, c*patch_size, L) -> (b, nb_patches, width, height)
        x = x.transpose(2, 1).contiguous().view(b, -1, self.patch_size, self.patch_size)

        # reshape to have (b x patches, c, h, w)
        x = x.view(-1, c, self.patch_size, self.patch_size)

        x = x.squeeze(0)

        return x


class DropChannels(object):

    def __init__(self, nb_drop_channels):
        self.nb_drop_channels = nb_drop_channels

    def __call__(self, x):
        b, c, h, w = x.size()

        channels_to_keep = c - self.nb_drop_channels
        idxs = np.random.choice(list(range(c)), replace=False, size=channels_to_keep)

        x = x[:, idxs, :, :]

        # add channel dim if only 1 channel in result
        if len(x.size()) == 3:
            x = x.unsqueeze(1)

        return x


class WR1AMDIM(nn.Module):
    def __init__(self, stages, image_input_size, depth=10, widen_factor=2, dropout_rate=0.4, num_classes=10, ddt=False,
                 sampling_patch_pct=0.2, use_amp=False):
        super(WR1AMDIM, self).__init__()
        stages = [int(x.strip()) for x in stages.split(',')]

        self.in_planes = stages[0]
        self.ddt = ddt
        self.sampling_patch_pct = sampling_patch_pct
        self.use_amp = use_amp

        ndf = 192
        n_depth = 8
        n_rkhs = 1536
        self.conv1 = Conv3x3(3, ndf, 3, 1, 0, False)
        self.layer1 = ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, True)
        self.layer2 = ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, True)
        self.layer3 = ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, True)
        self.bn1 = MaybeBatchNorm2d(ndf * 8, True, True)
        self.layer4 = ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, True)
        self.layer5 = ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, True)
        self.layer6 = ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, True)
        self.bn2 = MaybeBatchNorm2d(n_rkhs, True, True)

    def init_weights(self, init_scale=1.):
        '''
        Run custom weight init for modules...
        '''
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)

    def fine_tune_forward(self, x1):
        dtype = self.conv1.weight.dtype
        x1 = x1.type(dtype)
        x1 = self.conv1(x1)

        x2 = self.layer1(x1)
        x1 = F.max_pool2d(x1, 4)

        x3 = self.layer2(x2)
        x3 = F.max_pool2d(x3, 4)

        x4 = self.layer3(x3)
        x4 = F.relu(self.bn1(x4))
        x4 = F.max_pool2d(x4, 4)

        return [x1, x2, x3, x4]

    def forward(self, x, fine_tuning=False, *args, **kwargs):
        dtype = self.conv1.conv.weight.dtype
        x = x.type(dtype)

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn1(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.bn2(x)
        return [x]


class Encoder(nn.Module):
    def __init__(self, dummy_batch, num_channels=3, ndf=64, n_rkhs=512,
                n_depth=3, encoder_size=32, use_bn=False):
        super(Encoder, self).__init__()
        # NDF = encoder hidden feat size
        # RKHS = output dim
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None
        self.encoder_size = encoder_size

        # encoding block for local features
        print('Using a {}x{} encoder'.format(encoder_size, encoder_size))
        if encoder_size == 32:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 3, 1, 0, False),
                ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 4, True, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 4, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        elif encoder_size == 64:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 8, True, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        elif encoder_size == 128:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 5, 2, 2, False, pad_mode='reflect'),
                Conv3x3(ndf, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 8, True, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        else:
            raise RuntimeError("Could not build encoder."
                               "Encoder size {} is not supported".format(encoder_size))
        self._config_modules(
            dummy_batch,
            output_widths=[1, 5, 7],
            n_rkhs=n_rkhs,
            use_bn=use_bn
        )

    def init_weights(self, init_scale=1.):
        '''
        Run custom weight init for modules...
        '''
        for layer in self.layer_list:
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)

    def _config_modules(self, x, output_widths, n_rkhs, use_bn):
        '''
        Configure the modules for extracting fake rkhs embeddings for infomax.
        '''
        # get activations from each block to see output dims
        enc_acts = self._forward_acts(x)

        # out dimension to layer index
        # dim = number of output feature vectors
        self.dim2layer = {}

        # pull out layer indexes for the requested output_widths
        for layer_i, conv_out in enumerate(enc_acts):
            for output_width in output_widths:
                b, c, w, h = conv_out.size()
                if w == output_width:
                    self.dim2layer[w] = layer_i

        # get projected activation sizes at different layers
        ndf_1 = enc_acts[self.dim2layer[1]].size(1)
        ndf_5 = enc_acts[self.dim2layer[5]].size(1)
        ndf_7 = enc_acts[self.dim2layer[7]].size(1)

        # configure modules for fake rkhs embeddings
        self.rkhs_block_1 = NopNet()
        self.rkhs_block_5 = FakeRKHSConvNet(ndf_5, n_rkhs, use_bn)
        self.rkhs_block_7 = FakeRKHSConvNet(ndf_7, n_rkhs, use_bn)

    def _forward_acts(self, x):
        '''
        Return activations from all layers.
        '''
        # run forward pass through all layers
        layer_acts = [x]
        for _, layer in enumerate(self.layer_list):
            layer_in = layer_acts[-1]
            layer_out = layer(layer_in)
            layer_acts.append(layer_out)

        # remove input from the returned list of activations
        return_acts = layer_acts[1:]
        return return_acts

    def forward(self, x):
        # compute activations in all layers for x
        activations = self._forward_acts(x)

        # gather rkhs embeddings from certain layers
        # last feature map with (b, d, 1, 1) (ie: last network out)
        r1 = activations[self.dim2layer[1]]
        r1 = self.rkhs_block_1(r1)

        # last feature map with (b, d, 5, 5)
        r5 = activations[self.dim2layer[5]]
        r5 = self.rkhs_block_5(r5)

        # last feature map with (b, d, 7, 7)
        r7 = activations[self.dim2layer[7]]
        r7 = self.rkhs_block_7(r7)

        return r1, r5, r7


class PatchesEncoder(nn.Module):
    def __init__(self, dummy_batch, nb_out_maps=1, num_channels=3, ndf=64, n_rkhs=512,
                n_depth=3, patch_size=32, use_bn=False, norm_out=False, net='a'):
        super(PatchesEncoder, self).__init__()
        # NDF = encoder hidden feat size
        # RKHS = output dim
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None
        self.patch_size = patch_size
        self.nb_out_maps = nb_out_maps
        self.norm_out = norm_out

        # encoding block for local features
        print('Using a {}x{} patch encoder'.format(patch_size, patch_size))

        if patch_size == 8:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 3, 1, 0, False), # 3 - 320
                ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),# 320 320
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn), # 320 - 640
                ConvResBlock(ndf * 2, n_rkhs, 2, 2, 0, n_depth, use_bn), # 640 - 1280
            ])
        elif patch_size == 16:
            if net == 'a':
                self.layer_list = nn.ModuleList([
                    Conv3x3(num_channels, ndf, 3, 1, 0, False),
                    ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                    ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ])
            if net == 'b':
                self.layer_list = nn.ModuleList([
                    Conv3x3(num_channels, ndf, 3, 1, 0, False),
                    ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                    ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                    ConvResNxN(ndf * 4, n_rkhs, 3, 1, 0, use_bn),
                ])

        elif patch_size == 32:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 3, 1, 0, False),
                ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 4, n_rkhs, 3, 1, 0, use_bn),
            ])
        elif patch_size == 64:
            if net == 'a':
                self.layer_list = nn.ModuleList([
                    Conv3x3(num_channels, ndf, 3, 1, 0, False),
                    ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 8, n_rkhs, 3, 1, 0, n_depth, use_bn),
                ])
            if net == 'b':
                self.layer_list = nn.ModuleList([
                    Conv3x3(num_channels, ndf, 3, 1, 0, False),
                    ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                    ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                ])
        elif patch_size == 128:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 4, 2, 2, False, pad_mode='reflect'),
                Conv3x3(ndf, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 3, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 3, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 4, True, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 2, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        else:
            raise RuntimeError("Could not build encoder."
                               "Encoder size {} is not supported".format(patch_size))
        self._config_modules(
            dummy_batch,
            nb_last_maps=nb_out_maps,
            n_rkhs=n_rkhs,
            use_bn=use_bn
        )

    def init_weights(self, init_scale=1.):
        '''
        Run custom weight init for modules...
        '''
        for layer in self.layer_list:
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)

    def _config_modules(self, x, nb_last_maps, n_rkhs, use_bn):
        '''
        Configure the modules for extracting fake rkhs embeddings for infomax.
        '''
        # get activations from each block to see output dims
        enc_acts = self._forward_acts(x)

        # out dimension to layer index
        # dim = number of output feature vectors
        self.dim2layer = {}

        # pull out layer indexes for the requested output_widths
        for layer_i, conv_out in enumerate(enc_acts):
            b, c, w, h = conv_out.size()
            self.dim2layer[w] = layer_i

        # find the dim of the lask k requested maps
        dims = list(self.dim2layer.keys())
        dims = sorted(dims)
        dims = dims[:nb_last_maps]

        # configure first feat map norm layer
        self.rkhs_block_1 = NopNet()
        dims.pop(0)

        # for the remaining ones, make a projection conv
        rkhs_blocks = {}
        for dim in dims:
            feat_map = enc_acts[self.dim2layer[dim]].size(1)
            proj_block = FakeRKHSConvNet(feat_map, n_rkhs, use_bn)
            rkhs_blocks[str(dim)] = proj_block

        self.rkhs_blocks = nn.ModuleDict(rkhs_blocks)

    def _forward_acts(self, x):
        '''
        Return activations from all layers.
        '''
        # run forward pass through all layers
        layer_acts = [x]
        for _, layer in enumerate(self.layer_list):
            layer_in = layer_acts[-1]
            # layer_in = layer_in.contiguous()
            layer_out = layer(layer_in)
            # print(layer_in.size(-1), '->', layer_out.size(-1))
            layer_acts.append(layer_out)

        # remove input from the returned list of activations
        # pdb.set_trace()
        return_acts = layer_acts[1:]
        return return_acts

    def forward(self, x):
        # compute activations in all layers for x
        activations = self._forward_acts(x)

        # gather rkhs embeddings from certain layers
        # last feature map with (b, d, 1, 1) (ie: last network out)
        r1 = activations[self.dim2layer[1]]
        if self.norm_out:
            r1 = self.rkhs_block_1(r1)

        results = []
        results.append(r1)

        # get the rest of the requested maps
        if self.nb_out_maps > 1:
            # find the remaining maps (excluding first map)
            dims = list(self.dim2layer.keys())
            dims = sorted(dims)
            dims = dims[:self.nb_out_maps]
            dims.pop(0)

            for dim in dims:
                block = self.rkhs_blocks[str(dim)]
                act = activations[self.dim2layer[dim]]
                act = block(act)
                results.append(act)

        return results


class PatchesEncoderCheckpointed(nn.Module):
    def __init__(self, dummy_batch, nb_out_maps=1, num_channels=3, ndf=64, n_rkhs=512,
                n_depth=3, patch_size=32, use_bn=False, norm_out=False, net='a'):
        super(PatchesEncoderCheckpointed, self).__init__()
        # NDF = encoder hidden feat size
        # RKHS = output dim
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None
        self.patch_size = patch_size
        self.nb_out_maps = nb_out_maps
        self.norm_out = norm_out

        # encoding block for local features
        print('Using a {}x{} patch encoder'.format(patch_size, patch_size))

        # Conv3x3(num_channels, ndf, 3, 1, 0, False),
        # ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
        # ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
        # ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
        # MaybeBatchNorm2d(ndf * 4, True, use_bn),
        # ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
        # ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
        # ConvResNxN(ndf * 4, n_rkhs, 3, 1, 0, use_bn),
        # MaybeBatchNorm2d(n_rkhs, True, True)

        if patch_size == 8:
            self.layer_list = nn.Sequential(
                Conv3x3(num_channels, ndf, 3, 1, 0, False), # 3 - 320
                ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),# 320 320
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn), # 320 - 640
                ConvResBlock(ndf * 2, n_rkhs, 2, 2, 0, n_depth, use_bn), # 640 - 1280
            )
        elif patch_size == 16:
            if net == 'a':
                self.layer_list = nn.Sequential(
                    Conv3x3(num_channels, ndf, 3, 1, 0, False),
                    ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                    ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                )
            if net == 'b':
                self.layer_list = nn.Sequential(
                    Conv3x3(num_channels, ndf, 3, 1, 0, False),
                    ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                    ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                    ConvResNxN(ndf * 4, n_rkhs, 3, 1, 0, use_bn),
                )

        elif patch_size == 32:
            self.layer_list = nn.Sequential(
                Conv3x3(num_channels, ndf, 3, 1, 0, False),
                ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 4, n_rkhs, 3, 1, 0, use_bn),
            )
        elif patch_size == 64:
            if net == 'a':
                self.layer_list = nn.Sequential(
                    Conv3x3(num_channels, ndf, 3, 1, 0, False),
                    ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 8, n_rkhs, 3, 1, 0, n_depth, use_bn),
                )
            if net == 'b':
                self.layer_list = nn.Sequential(
                    Conv3x3(num_channels, ndf, 3, 1, 0, False),
                    ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                    ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                )
        elif patch_size == 128:
            self.layer_list = nn.Sequential(
                Conv3x3(num_channels, ndf, 4, 2, 2, False, pad_mode='reflect'),
                Conv3x3(ndf, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 3, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 3, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 4, True, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 2, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            )
        else:
            raise RuntimeError("Could not build encoder."
                               "Encoder size {} is not supported".format(patch_size))

    def init_weights(self, init_scale=1.):
        '''
        Run custom weight init for modules...
        '''
        for layer in self.layer_list.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)


    def forward(self, x):
        # compute activations in all layers for x
        modules = [module for k, module in self.layer_list._modules.items()]
        r1 = checkpoint.checkpoint_sequential(modules, len(modules), x)
        return [r1]

class EncoderResnet(nn.Module):
    def __init__(self, dummy_batch, resnet_name, num_channels=3, ndf=64, n_rkhs=512,
                n_depth=3, encoder_size=32, use_bn=False, expansion=1):
        super(EncoderResnet, self).__init__()
        # NDF = encoder hidden feat size
        # RKHS = output dim
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None
        self.encoder_size = encoder_size

        # self.resnet = resnets.ResNet101NoBN()
        self.resnet = resnets.ResNet101_Xpanded(expansion=expansion)

        self._config_modules(
            dummy_batch,
            output_widths=[1, 2, 4, 8],
            n_rkhs=n_rkhs,
            use_bn=use_bn
        )

    def init_weights(self, init_scale=1.):
        '''
        Run custom weight init for modules...
        '''
        for layer in self.layer_list:
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)

    def _config_modules(self, x, output_widths, n_rkhs, use_bn):
        '''
        Configure the modules for extracting fake rkhs embeddings for infomax.
        '''
        # get activations from each block to see output dims
        enc_acts = self._forward_acts(x)

        # out dimension to layer index
        # dim = number of output feature vectors
        self.dim2layer = {}

        # pull out layer indexes for the requested output_widths
        for layer_i, conv_out in enumerate(enc_acts):
            for output_width in output_widths:
                b, c, w, h = conv_out.size()
                if w == output_width:
                    self.dim2layer[w] = layer_i

        # get projected activation sizes at different layers
        ndf_1 = enc_acts[self.dim2layer[1]].size(1)
        ndf_2 = enc_acts[self.dim2layer[2]].size(1)
        ndf_4 = enc_acts[self.dim2layer[4]].size(1)
        ndf_8 = enc_acts[self.dim2layer[8]].size(1)

        # configure modules for fake rkhs embeddings
        self.rkhs_block_1a = FakeRKHSConvNet(ndf_1, n_rkhs, use_bn)
        self.rkhs_block_1b = NopNet()
        self.rkhs_block_2 = FakeRKHSConvNet(ndf_2, n_rkhs, use_bn)
        self.rkhs_block_4 = FakeRKHSConvNet(ndf_4, n_rkhs, use_bn)
        self.rkhs_block_8 = FakeRKHSConvNet(ndf_8, n_rkhs, use_bn)

    def _forward_acts(self, x):
        '''
        Return activations from all layers.
        '''
        return self.resnet(x)

    def forward(self, x):
        # compute activations in all layers for x
        activations = self._forward_acts(x)

        # gather rkhs embeddings from certain layers
        # last feature map with (b, d, 1, 1) (ie: last network out)
        r1 = activations[self.dim2layer[1]]
        r1 = self.rkhs_block_1a(r1)
        r1 = self.rkhs_block_1b(r1)

        # last feature map with (b, d, 5, 5)
        r2 = activations[self.dim2layer[2]]
        r2 = self.rkhs_block_2(r2)

        # last feature map with (b, d, 7, 7)
        r4 = activations[self.dim2layer[4]]
        r4 = self.rkhs_block_4(r4)

        r8 = activations[self.dim2layer[8]]
        r8 = self.rkhs_block_8(r8)

        return r1, r2, r4, r8


class EncoderResnetOriginalPatches(nn.Module):
    def __init__(self, dummy_batch, resnet_name, ndf=64, n_rkhs=512,
                encoder_size=32, use_bn=False, norm_out=False):
        super(EncoderResnetOriginalPatches, self).__init__()
        # NDF = encoder hidden feat size
        # RKHS = output dim
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None
        self.encoder_size = encoder_size
        self.norm_out = norm_out

        model = getattr(resnets, resnet_name)
        self.resnet = model(pretrained=False)
        # self.resnet = resnets.ResNet101_Xpanded(expansion=expansion)

        self._config_modules(
            dummy_batch,
            output_widths=[1],
            n_rkhs=n_rkhs,
            use_bn=use_bn
        )

    def init_weights(self, init_scale=1.):
        '''
        Run custom weight init for modules...
        '''
        for layer in self.layer_list:
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)

    def _config_modules(self, x, output_widths, n_rkhs, use_bn):
        '''
        Configure the modules for extracting fake rkhs embeddings for infomax.
        '''
        # get activations from each block to see output dims
        enc_acts = self._forward_acts(x)

        # out dimension to layer index
        # dim = number of output feature vectors
        self.dim2layer = {}

        # pull out layer indexes for the requested output_widths
        for layer_i, conv_out in enumerate(enc_acts):
            for output_width in output_widths:
                b, c, w, h = conv_out.size()
                if w == output_width:
                    self.dim2layer[w] = layer_i

        # get projected activation sizes at different layers
        # configure modules for fake rkhs embeddings
        self.rkhs_block_1 = NopNet()

    def _forward_acts(self, x):
        '''
        Return activations from all layers.
        '''
        return self.resnet(x)

    def forward(self, x):
        # compute activations in all layers for x
        activations = self._forward_acts(x)

        # gather rkhs embeddings from certain layers
        # last feature map with (b, d, 1, 1) (ie: last network out)
        r1 = activations[self.dim2layer[1]]

        # if self.norm_out:
        #     r1 = self.rkhs_block_1(r1)

        return [r1]


class EncoderResnetOriginal(nn.Module):
    def __init__(self, dummy_batch, resnet_name, num_channels=3, ndf=64, n_rkhs=512,
                n_depth=3, encoder_size=32, use_bn=False):
        super(EncoderResnetOriginal, self).__init__()
        # NDF = encoder hidden feat size
        # RKHS = output dim
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None
        self.encoder_size = encoder_size

        model = getattr(resnets, resnet_name)
        self.resnet = model(pretrained=False)

        self._config_modules(
            dummy_batch,
            output_widths=[1, 2, 4, 8],
            n_rkhs=n_rkhs,
            use_bn=use_bn
        )

    def init_weights(self, init_scale=1.):
        '''
        Run custom weight init for modules...
        '''
        for layer in self.layer_list:
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)

    def _config_modules(self, x, output_widths, n_rkhs, use_bn):
        '''
        Configure the modules for extracting fake rkhs embeddings for infomax.
        '''
        # get activations from each block to see output dims
        enc_acts = self._forward_acts(x)

        # out dimension to layer index
        # dim = number of output feature vectors
        self.dim2layer = {}

        # pull out layer indexes for the requested output_widths
        for layer_i, conv_out in enumerate(enc_acts):
            for output_width in output_widths:
                b, c, w, h = conv_out.size()
                if w == output_width:
                    self.dim2layer[w] = layer_i

        # get projected activation sizes at different layers
        ndf_1 = enc_acts[self.dim2layer[1]].size(1)
        ndf_2 = enc_acts[self.dim2layer[2]].size(1)
        ndf_4 = enc_acts[self.dim2layer[4]].size(1)
        ndf_8 = enc_acts[self.dim2layer[8]].size(1)

        # configure modules for fake rkhs embeddings
        self.rkhs_block_1a = FakeRKHSConvNet(ndf_1, n_rkhs, use_bn)
        self.rkhs_block_1b = NopNet()
        self.rkhs_block_2 = FakeRKHSConvNet(ndf_2, n_rkhs, use_bn)
        self.rkhs_block_4 = FakeRKHSConvNet(ndf_4, n_rkhs, use_bn)
        self.rkhs_block_8 = FakeRKHSConvNet(ndf_8, n_rkhs, use_bn)

    def _forward_acts(self, x):
        '''
        Return activations from all layers.
        '''
        return self.resnet(x)

    def forward(self, x):
        # compute activations in all layers for x
        activations = self._forward_acts(x)

        # gather rkhs embeddings from certain layers
        # last feature map with (b, d, 1, 1) (ie: last network out)
        r1 = activations[self.dim2layer[1]]
        r1 = self.rkhs_block_1a(r1)
        r1 = self.rkhs_block_1b(r1)

        # last feature map with (b, d, 5, 5)
        r2 = activations[self.dim2layer[2]]
        r2 = self.rkhs_block_2(r2)

        # last feature map with (b, d, 7, 7)
        r4 = activations[self.dim2layer[4]]
        r4 = self.rkhs_block_4(r4)

        r8 = activations[self.dim2layer[8]]
        r8 = self.rkhs_block_8(r8)

        return r1, r2, r4, r8


class EncoderDDT(nn.Module):
    def __init__(self, dummy_batch, num_channels=3, ndf=64, n_rkhs=512,
                n_depth=3, encoder_size=32, use_bn=False):
        super(EncoderDDT, self).__init__()
        # NDF = encoder hidden feat size
        # RKHS = output dim
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None
        self.encoder_size = encoder_size

        # encoding block for local features
        print('Using a {}x{} encoder'.format(encoder_size, encoder_size))
        if encoder_size == 32:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 3, 1, 0, False),
                ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 4, True, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 4, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        elif encoder_size == 64:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 8, True, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        elif encoder_size == 128:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 5, 2, 2, False, pad_mode='reflect'),
                Conv3x3(ndf, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 8, True, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        else:
            raise RuntimeError("Could not build encoder."
                               "Encoder size {} is not supported".format(encoder_size))
        self._config_modules(
            dummy_batch,
            output_widths=[1, 3, 5, 7, 14, 30],
            n_rkhs=n_rkhs,
            use_bn=use_bn
        )

    def init_weights(self, init_scale=1.):
        '''
        Run custom weight init for modules...
        '''
        for layer in self.layer_list:
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)

    def _config_modules(self, x, output_widths, n_rkhs, use_bn):
        '''
        Configure the modules for extracting fake rkhs embeddings for infomax.
        '''
        # get activations from each block to see output dims
        enc_acts = self._forward_acts(x)

        # out dimension to layer index
        # dim = number of output feature vectors
        self.dim2layer = {}

        # pull out layer indexes for the requested output_widths
        for layer_i, conv_out in enumerate(enc_acts):
            for output_width in output_widths:
                b, c, w, h = conv_out.size()
                if w == output_width:
                    self.dim2layer[w] = layer_i

        # get projected activation sizes at different layers
        ndf_1 = enc_acts[self.dim2layer[1]].size(1)
        ndf_3 = enc_acts[self.dim2layer[3]].size(1)
        ndf_5 = enc_acts[self.dim2layer[5]].size(1)
        ndf_7 = enc_acts[self.dim2layer[7]].size(1)
        ndf_14 = enc_acts[self.dim2layer[14]].size(1)
        ndf_30 = enc_acts[self.dim2layer[30]].size(1)

        # configure modules for fake rkhs embeddings
        self.rkhs_block_1 = NopNet()
        self.rkhs_block_3 = FakeRKHSConvNet(ndf_3, n_rkhs, use_bn)
        self.rkhs_block_5 = FakeRKHSConvNet(ndf_5, n_rkhs, use_bn)
        self.rkhs_block_7 = FakeRKHSConvNet(ndf_7, n_rkhs, use_bn)
        self.rkhs_block_14 = FakeRKHSConvNet(ndf_14, n_rkhs, use_bn)
        self.rkhs_block_30 = FakeRKHSConvNet(ndf_30, n_rkhs, use_bn)

    def _forward_acts(self, x):
        '''
        Return activations from all layers.
        '''
        # run forward pass through all layers
        layer_acts = [x]
        for layer_i, layer in enumerate(self.layer_list):
            layer_in = layer_acts[-1]
            layer_out = layer(layer_in)
            layer_acts.append(layer_out)

        # remove input from the returned list of activations
        return_acts = layer_acts[1:]
        return return_acts

    def forward(self, x):
        # compute activations in all layers for x
        activations = self._forward_acts(x)

        # gather rkhs embeddings from certain layers
        # last feature map with (b, d, 1, 1) (ie: last network out)
        r1 = activations[self.dim2layer[1]]
        r1 = self.rkhs_block_1(r1)

        r3 = activations[self.dim2layer[3]]
        r3 = self.rkhs_block_3(r3)

        # last feature map with (b, d, 5, 5)
        r5 = activations[self.dim2layer[5]]
        r5 = self.rkhs_block_5(r5)

        # last feature map with (b, d, 7, 7)
        r7 = activations[self.dim2layer[7]]
        r7 = self.rkhs_block_7(r7)

        r14 = activations[self.dim2layer[14]]
        r14 = self.rkhs_block_14(r14)

        r30 = activations[self.dim2layer[30]]
        r30 = self.rkhs_block_30(r30)

        return r1, r3, r5, r7, r14, r30

##############################
# Layers for use in model... #
##############################


class MaybeBatchNorm2d(nn.Module):
    def __init__(self, n_ftr, affine, use_bn):
        super(MaybeBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(n_ftr, affine=affine)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        return x


class NopNet(nn.Module):
    def __init__(self, norm_dim=None):
        super(NopNet, self).__init__()
        self.norm_dim = norm_dim

    def forward(self, x):
        if self.norm_dim is not None:
            x_norms = torch.sum(x**2., dim=self.norm_dim, keepdim=True)
            x_norms = torch.sqrt(x_norms + 1e-6)
            x = x / x_norms
        return x


class Conv3x3(nn.Module):
    def __init__(self, n_in, n_out, n_kern, n_stride, n_pad,
                 use_bn=True, pad_mode='constant'):
        super(Conv3x3, self).__init__()
        assert(pad_mode in ['constant', 'reflect'])
        self.n_pad = (n_pad, n_pad, n_pad, n_pad)
        self.pad_mode = pad_mode
        self.conv = nn.Conv2d(n_in, n_out, n_kern, n_stride, 0,
                              bias=(not use_bn))
        self.relu = nn.ReLU(inplace=True)
        self.bn = MaybeBatchNorm2d(n_out, True, use_bn)

    def forward(self, x):
        if self.n_pad[0] > 0:
            # maybe pad the input
            x = F.pad(x, self.n_pad, mode=self.pad_mode)
        # always apply conv
        x = self.conv(x)
        # maybe apply batchnorm
        x = self.bn(x)
        # always apply relu
        out = self.relu(x)
        return out


class FakeRKHSConvNet(nn.Module):
    def __init__(self, n_input, n_output, use_bn=False):
        super(FakeRKHSConvNet, self).__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn1 = MaybeBatchNorm2d(n_output, True, use_bn)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output, n_output, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn_out = MaybeBatchNorm2d(n_output, True, True)
        self.shortcut = nn.Conv2d(n_input, n_output, kernel_size=1,
                                  stride=1, padding=0, bias=True)
        # when possible, initialize shortcut to be like identity
        if n_output >= n_input:
            eye_mask = np.zeros((n_output, n_input, 1, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.shortcut.weight.data.uniform_(-0.01, 0.01)
            self.shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)
        return

    def init_weights(self, init_scale=1.):
        # initialize first conv in res branch
        # -- rescale the default init for nn.Conv2d layers
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        # initialize second conv in res branch
        # -- set to 0, like fixup/zero init
        nn.init.constant_(self.conv2.weight, 0.)

    def forward(self, x):
        h_res = self.conv2(self.relu1(self.bn1(self.conv1(x))))
        h = self.bn_out(h_res + self.shortcut(x))
        return h


class ConvResNxN(nn.Module):
    def __init__(self, n_in, n_out, width, stride, pad, use_bn=False):
        super(ConvResNxN, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.width = width
        self.stride = stride
        self.pad = pad
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_in, n_out, width, stride, pad, bias=False)
        self.conv2 = nn.Conv2d(n_out, n_out, 1, 1, 0, bias=False)
        self.n_grow = n_out - n_in
        if self.n_grow < 0:
            # use self.conv3 to downsample feature dim
            self.conv3 = nn.Conv2d(n_in, n_out, width, stride, pad, bias=True)
        else:
            # self.conv3 is not used when n_out >= n_in
            self.conv3 = None
        self.bn1 = MaybeBatchNorm2d(n_out, True, use_bn)

    def init_weights(self, init_scale=1.):
        # initialize first conv in res branch
        # -- rescale the default init for nn.Conv2d layers
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        # initialize second conv in res branch
        # -- set to 0, like fixup/zero init
        nn.init.constant_(self.conv2.weight, 0.)

    def forward(self, x):
        h1 = self.bn1(self.conv1(x))
        h2 = self.conv2(self.relu2(h1))
        if self.n_out < self.n_in:
            h3 = self.conv3(x)
        elif self.n_in == self.n_out:
            h3 = F.avg_pool2d(x, self.width, self.stride, self.pad)
        else:
            h3_pool = F.avg_pool2d(x, self.width, self.stride, self.pad)
            h3 = F.pad(h3_pool, (0, 0, 0, 0, 0, self.n_grow))
        h23 = h2 + h3
        return h23


class ConvResBlock(nn.Module):
    def __init__(self, n_in, n_out, width, stride, pad, depth, use_bn):
        super(ConvResBlock, self).__init__()
        layer_list = [ConvResNxN(n_in, n_out, width, stride, pad, use_bn)]
        for i in range(depth - 1):
            layer_list.append(ConvResNxN(n_out, n_out, 1, 1, 0, use_bn))
        self.layer_list = nn.Sequential(*layer_list)

    def init_weights(self, init_scale=1.):
        '''
        Do a fixup-ish init for each ConvResNxN in this block.
        '''
        for m in self.layer_list:
            m.init_weights(init_scale)

    def forward(self, x):
        # run forward pass through the list of ConvResNxN layers
        x_out = self.layer_list(x)
        return x_out


class CPCLossNCE(nn.Module):
    def __init__(self, tclip=10.):
        super(CPCLossNCE, self).__init__()
        self.masks_r5 = self.feat_size_5_mask()

    def feat_size_5_mask(self):
        masks_r5 = np.zeros((5, 5, 1, 5, 5))
        for i in range(5):
            for j in range(5):
                masks_r5[i, j, 0, i, j] = 1
        masks_r5 = torch.tensor(masks_r5).type(torch.uint8)
        masks_r5 = masks_r5.reshape(-1, 1, 5, 5)
        return nn.Parameter(masks_r5, requires_grad=False)

    def nce_loss(self, anchor, pos_scores, negative_samples, mask_mat):

        # RKHS = embedding dim
        pos_scores = pos_scores.float()
        batch_size, emb_dim = anchor.size()
        nb_feat_vectors = negative_samples.size(1) // batch_size

        # (b, b) -> (b, b, nb_feat_vectors)
        # all zeros with ones in diagonal tensor... (ie: b1 b1 are all 1s, b1 b2 are all zeros)
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, nb_feat_vectors).float()

        # negative mask
        mask_neg = 1. - mask_pos

        # -------------------------------
        # ALL SCORES COMPUTATION
        # (b, dim) x (dim, nb_feats*b) -> (b, b, nb_feats)
        # vector for each img in batch times all the vectors of all images in batch
        raw_scores = torch.mm(anchor, negative_samples)
        raw_scores = raw_scores.reshape(batch_size, batch_size, nb_feat_vectors).float()

        # ----------------------
        # EXTRACT NEGATIVE SCORES
        # (batch_size, batch_size, nb_feat_vectors)
        neg_scores = (mask_neg * raw_scores)

        # (batch_size, batch_size * nb_feat_vectors) -> (batch_size, batch_size, nb_feat_vectors)
        neg_scores = neg_scores.reshape(batch_size, -1)
        mask_neg = mask_neg.reshape(batch_size, -1)

        # ---------------------
        # STABLE SOFTMAX
        # (n_batch_gpu, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]

        # DENOMINATOR
        # sum over only negative samples (none from the diagonal)
        neg_sumexp = (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)

        # NUMERATOR
        # compute numerators for the NCE log-softmaxes
        pos_shiftexp = pos_scores - neg_maxes

        # FULL NCE
        nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = -nce_scores.mean()

        return nce_scores

    def forward(self, Z, C, W_list):
        """

        :param Z: latent vars (b*patches, emb_dim, h, w)
        :param C: context var (b*patches, emb_dim, h, w)
        :param W_list: list of k-1 W projections
        :return:
        """
        # (b, dim, w. h)
        batch_size, emb_dim, h, w = Z.size()

        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(Z.device.index)
        diag_mat = diag_mat.float()

        losses = []
        # calculate loss for each k

        Z_neg = Z.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        for i in range(0, h-1):
            for j in range(0, w):
                cij = C[:, :, i, j]

                # make predictions far and non-overlapping
                min_k_dist = 2

                for k in range(i+min_k_dist, h):
                    Wk = W_list[str(k)]

                    z_hat_ik_j = Wk(cij)

                    zikj = Z[:, :, k, j]

                    # BATCH DOT PRODUCT
                    # (b, d) x (b, d) -> (b, 1)
                    pos_scores = torch.bmm(z_hat_ik_j.unsqueeze(1), zikj.unsqueeze(2))
                    pos_scores = pos_scores.squeeze(-1).squeeze(-1)

                    loss = self.nce_loss(z_hat_ik_j, pos_scores, Z_neg, diag_mat)
                    losses.append(loss)

        losses = torch.stack(losses)
        loss = losses.mean()
        return loss


class LossMultiNCE(nn.Module):
    '''
    Input is fixed as r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2.
    '''

    def __init__(self, tclip=10.):
        super(LossMultiNCE, self).__init__()
        # construct masks for sampling source features from 5x5 layer
        # (b, 1, 5, 5)
        self.masks_r5 = self.feat_size_5_mask()

        self.tclip = tclip

    def nce_loss(self, r_src, r_trg, mask_mat):
        '''
        Compute the NCE scores for predicting r_src->r_trg.
        Input:
          r_src    : (batch_size, emb_dim)
          r_trg    : (emb_dim, n_batch * w* h) (ie: nb_feat_vectors x embedding_dim)
          mask_mat : (n_batch_gpu, n_batch)
        Output:
          raw_scores : (n_batch_gpu, n_locs)
          nce_scores : (n_batch_gpu, n_locs)
          lgt_reg    : scalar
        '''
        # RKHS = embedding dim
        batch_size, emb_dim = r_src.size()
        nb_feat_vectors = r_trg.size(1) // batch_size

        # (b, b) -> (b, b, nb_feat_vectors)
        # all zeros with ones in diagonal tensor... (ie: b1 b1 are all 1s, b1 b2 are all zeros)
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, nb_feat_vectors).float()

        # negative mask
        mask_neg = 1. - mask_pos

        # -------------------------------
        # ALL SCORES COMPUTATION
        # compute src->trg raw scores for batch
        # (b, dim) x (dim, nb_feats*b) -> (b, b, nb_feats)
        # vector for each img in batch times all the vectors of all images in batch
        raw_scores = torch.mm(r_src, r_trg).float()
        raw_scores = raw_scores.reshape(batch_size, batch_size, nb_feat_vectors)

        # -----------------------
        # STABILITY TRICKS
        # trick 1: weighted regularization term
        raw_scores = raw_scores / emb_dim**0.5
        lgt_reg = 5e-2 * (raw_scores**2.).mean()

        # trick 2: tanh clip
        raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)

        '''
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        '''
        # ----------------------
        # EXTRACT POSITIVE SCORES
        # use the index mask to pull all the diagonals which are b1 x b1
        # (batch_size, nb_feat_vectors)
        pos_scores = (mask_pos * raw_scores).sum(dim=1)

        # ----------------------
        # EXTRACT NEGATIVE SCORES
        # pull everything except diagonal and apply clipping
        # (batch_size, batch_size, nb_feat_vectors)
        # diagonals have - clip vals. everything else has actual negative stores
        neg_scores = (mask_neg * raw_scores) - (self.tclip * mask_pos)

        # (batch_size, batch_size * nb_feat_vectors) -> (batch_size, batch_size, nb_feat_vectors)
        neg_scores = neg_scores.reshape(batch_size, -1)
        mask_neg = mask_neg.reshape(batch_size, -1)

        # ---------------------
        # STABLE SOFTMAX
        # max for each row of negative samples
        # will use max in safe softmax
        # (n_batch_gpu, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]

        # DENOMINATOR
        # sum over only negative samples (none from the diagonal)
        neg_sumexp = (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)

        # NUMERATOR
        # compute numerators for the NCE log-softmaxes
        pos_shiftexp = pos_scores - neg_maxes

        # FULL NCE
        nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = -nce_scores.mean()

        return nce_scores, lgt_reg

    def feat_size_5_mask(self):
        masks_r5 = np.zeros((5, 5, 1, 5, 5))
        for i in range(5):
            for j in range(5):
                masks_r5[i, j, 0, i, j] = 1
        masks_r5 = torch.tensor(masks_r5).type(torch.uint8)
        masks_r5 = masks_r5.reshape(-1, 1, 5, 5)
        return nn.Parameter(masks_r5, requires_grad=False)

    def _sample_src_ftr(self, r_cnv, masks):
        # get feature dimensions
        n_batch = r_cnv.size(0)
        feat_dim = r_cnv.size(1)

        if masks is not None:
            # subsample from conv-ish r_cnv to get a single vector
            mask_idx = torch.randint(0, masks.size(0), (n_batch,))
            r_cnv = torch.masked_select(r_cnv, masks[mask_idx])

        # flatten features for use as globals in glb->lcl nce cost
        r_vec = r_cnv.reshape(n_batch, feat_dim)
        return r_vec

    def forward(self, r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2):
        '''
        Compute nce infomax costs for various combos of source/target layers.
        Compute costs in both directions, i.e. from/to both images (x1, x2).
        rK_x1 are features from source image x1.
        rK_x2 are features from source image x2.
        '''
        # (b, dim, w. h)
        batch_size, emb_dim, _, _ = r1_x1.size()

        # -----------------
        # SOURCE VECTORS
        # 1 feature vector per image per feature map location
        # img 1
        r1_src_1 = self._sample_src_ftr(r1_x1, None)
        r5_src_1 = self._sample_src_ftr(r5_x1, self.masks_r5)

        # img 2
        r1_src_2 = self._sample_src_ftr(r1_x2, None)
        r5_src_2 = self._sample_src_ftr(r5_x2, self.masks_r5)

        # -----------------
        # TARGET VECTORS
        # before shape: (n_batch, emb_dim, w, h)
        r5_trg_1 = r5_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r7_trg_1 = r7_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r5_trg_2 = r5_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r7_trg_2 = r7_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        # after shape: (emb_dim, n_batch * w * h)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_x1.device.index)

        # -----------------
        # NCE COSTS
        # compute costs for 1->5 prediction
        # use last layer to predict the layer with (5x5 features)
        loss_1t5_1, regularizer_1t5_1 = self.nce_loss(r1_src_1, r5_trg_2, diag_mat)  # img 1
        loss_1t5_2, regularizer_1t5_2 = self.nce_loss(r1_src_2, r5_trg_1, diag_mat)  # img 2

        # compute costs for 1->7 prediction
        # use last layer to predict the layer with (7x7 features)
        loss_1t7_1, regularizer_1t7_1 = self.nce_loss(r1_src_1, r7_trg_2, diag_mat)  # img 1
        loss_1t7_2, regularizer_1t7_2 = self.nce_loss(r1_src_2, r7_trg_1, diag_mat)  # img 2

        # compute costs for 5->5 prediction
        # use (5x5) layer to predict the (5x5) layer
        loss_5t5_1, regularizer_5t5_1 = self.nce_loss(r5_src_1, r5_trg_2, diag_mat)  # img 1
        loss_5t5_2, regularizer_5t5_2 = self.nce_loss(r5_src_2, r5_trg_1, diag_mat)  # img 2

        # combine costs for optimization
        loss_1t5 = 0.5 * (loss_1t5_1 + loss_1t5_2)
        loss_1t7 = 0.5 * (loss_1t7_1 + loss_1t7_2)
        loss_5t5 = 0.5 * (loss_5t5_1 + loss_5t5_2)

        # regularizer
        regularizer = 0.5 * (regularizer_1t5_1 + regularizer_1t5_2 +
                             regularizer_1t7_1 + regularizer_1t7_2 +
                             regularizer_5t5_1 + regularizer_5t5_2)

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss_1t5 = loss_1t5.mean()
        loss_1t7 = loss_1t7.mean()
        loss_5t5 = loss_5t5.mean()
        regularizer = regularizer.mean()
        return loss_1t5, loss_1t7, loss_5t5, regularizer


class LossMultiNCEResnet(nn.Module):
    '''
    Input is fixed as r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2.
    '''

    def __init__(self, tclip=10.):
        super(LossMultiNCEResnet, self).__init__()
        # construct masks for sampling source features from 5x5 layer
        # (b, 1, 5, 5)
        self.masks_r2 = self.feat_size_w_mask(2)
        self.masks_r4 = self.feat_size_w_mask(4)

        self.tclip = tclip

    def nce_loss(self, r_src, r_trg, mask_mat):
        '''
        Compute the NCE scores for predicting r_src->r_trg.
        Input:
          r_src    : (batch_size, emb_dim)
          r_trg    : (emb_dim, n_batch * w* h) (ie: nb_feat_vectors x embedding_dim)
          mask_mat : (n_batch_gpu, n_batch)
        Output:
          raw_scores : (n_batch_gpu, n_locs)
          nce_scores : (n_batch_gpu, n_locs)
          lgt_reg    : scalar
        '''
        # RKHS = embedding dim
        batch_size, emb_dim = r_src.size()
        nb_feat_vectors = r_trg.size(1) // batch_size

        # (b, b) -> (b, b, nb_feat_vectors)
        # all zeros with ones in diagonal tensor... (ie: b1 b1 are all 1s, b1 b2 are all zeros)
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, nb_feat_vectors).float()

        # negative mask
        mask_neg = 1. - mask_pos

        # -------------------------------
        # ALL SCORES COMPUTATION
        # compute src->trg raw scores for batch
        # (b, dim) x (dim, nb_feats*b) -> (b, b, nb_feats)
        # vector for each img in batch times all the vectors of all images in batch
        raw_scores = torch.mm(r_src, r_trg).float()
        raw_scores = raw_scores.reshape(batch_size, batch_size, nb_feat_vectors)

        # -----------------------
        # STABILITY TRICKS
        # trick 1: weighted regularization term
        raw_scores = raw_scores / emb_dim**0.5
        lgt_reg = 5e-2 * (raw_scores**2.).mean()

        # trick 2: tanh clip
        raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)

        '''
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        '''
        # ----------------------
        # EXTRACT POSITIVE SCORES
        # use the index mask to pull all the diagonals which are b1 x b1
        # (batch_size, nb_feat_vectors)
        pos_scores = (mask_pos * raw_scores).sum(dim=1)

        # ----------------------
        # EXTRACT NEGATIVE SCORES
        # pull everything except diagonal and apply clipping
        # (batch_size, batch_size, nb_feat_vectors)
        # diagonals have - clip vals. everything else has actual negative stores
        neg_scores = (mask_neg * raw_scores) - (self.tclip * mask_pos)

        # (batch_size, batch_size * nb_feat_vectors) -> (batch_size, batch_size, nb_feat_vectors)
        neg_scores = neg_scores.reshape(batch_size, -1)
        mask_neg = mask_neg.reshape(batch_size, -1)

        # ---------------------
        # STABLE SOFTMAX
        # max for each row of negative samples
        # will use max in safe softmax
        # (n_batch_gpu, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]

        # DENOMINATOR
        # sum over only negative samples (none from the diagonal)
        neg_sumexp = (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)

        # NUMERATOR
        # compute numerators for the NCE log-softmaxes
        pos_shiftexp = pos_scores - neg_maxes

        # FULL NCE
        nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = -nce_scores.mean()

        return nce_scores, lgt_reg

    def feat_size_w_mask(self, w):
        masks_r5 = np.zeros((w, w, 1, w, w))
        for i in range(w):
            for j in range(w):
                masks_r5[i, j, 0, i, j] = 1
        masks_r5 = torch.tensor(masks_r5).type(torch.uint8)
        masks_r5 = masks_r5.reshape(-1, 1, w, w)
        return nn.Parameter(masks_r5, requires_grad=False)

    def _sample_src_ftr(self, r_cnv, masks):
        # get feature dimensions
        n_batch = r_cnv.size(0)
        feat_dim = r_cnv.size(1)

        if masks is not None:
            # subsample from conv-ish r_cnv to get a single vector
            mask_idx = torch.randint(0, masks.size(0), (n_batch,))
            r_cnv = torch.masked_select(r_cnv, masks[mask_idx])

        # flatten features for use as globals in glb->lcl nce cost
        r_vec = r_cnv.reshape(n_batch, feat_dim)
        return r_vec

    def forward(self, r1_x1, r2_x1, r4_x1, r8_x1, r1_x2, r2_x2, r4_x2, r8_x2):
        '''
        Compute nce infomax costs for various combos of source/target layers.
        Compute costs in both directions, i.e. from/to both images (x1, x2).
        rK_x1 are features from source image x1.
        rK_x2 are features from source image x2.
        '''
        # (b, dim, w. h)
        batch_size, emb_dim, _, _ = r1_x1.size()

        # -----------------
        # SOURCE VECTORS
        # 1 feature vector per image per feature map location
        # img 1
        r1_src_1 = self._sample_src_ftr(r1_x1, None)
        r2_src_1 = self._sample_src_ftr(r2_x1, self.masks_r2)

        # img 2
        r1_src_2 = self._sample_src_ftr(r1_x2, None)
        r2_src_2 = self._sample_src_ftr(r2_x2, self.masks_r2)

        # -----------------
        # TARGET VECTORS
        # before shape: (n_batch, emb_dim, w, h)
        r2_trg_1 = r2_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r4_trg_1 = r4_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r2_trg_2 = r2_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r4_trg_2 = r4_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        # after shape: (emb_dim, n_batch * w * h)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_x1.device.index)

        # -----------------
        # NCE COSTS
        # compute costs for 1->5 prediction
        # use last layer to predict the layer with (5x5 features)
        loss_1t5_1, regularizer_1t5_1 = self.nce_loss(r1_src_1, r2_trg_2, diag_mat)  # img 1
        loss_1t5_2, regularizer_1t5_2 = self.nce_loss(r1_src_2, r2_trg_1, diag_mat)  # img 2

        # compute costs for 1->7 prediction
        # use last layer to predict the layer with (7x7 features)
        loss_1t7_1, regularizer_1t7_1 = self.nce_loss(r1_src_1, r4_trg_2, diag_mat)  # img 1
        loss_1t7_2, regularizer_1t7_2 = self.nce_loss(r1_src_2, r4_trg_1, diag_mat)  # img 2

        # compute costs for 5->5 prediction
        # use (5x5) layer to predict the (5x5) layer
        loss_5t5_1, regularizer_5t5_1 = self.nce_loss(r2_src_1, r2_trg_2, diag_mat)  # img 1
        loss_5t5_2, regularizer_5t5_2 = self.nce_loss(r2_src_2, r2_trg_1, diag_mat)  # img 2

        # combine costs for optimization
        loss_1t5 = 0.5 * (loss_1t5_1 + loss_1t5_2)
        loss_1t7 = 0.5 * (loss_1t7_1 + loss_1t7_2)
        loss_5t5 = 0.5 * (loss_5t5_1 + loss_5t5_2)

        # regularizer
        regularizer = 0.5 * (regularizer_1t5_1 + regularizer_1t5_2 +
                             regularizer_1t7_1 + regularizer_1t7_2 +
                             regularizer_5t5_1 + regularizer_5t5_2)

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss_1t5 = loss_1t5.mean()
        loss_1t7 = loss_1t7.mean()
        loss_5t5 = loss_5t5.mean()
        regularizer = regularizer.mean()
        return loss_1t5, loss_1t7, loss_5t5, regularizer


class LossMultiNCEPatches(nn.Module):
    '''
    Input is fixed as r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2.
    '''

    def __init__(self, strategy='1:1', tclip=10.):
        super(LossMultiNCEPatches, self).__init__()
        # construct masks for sampling source features from 5x5 layer
        # (b, 1, 5, 5)
        self.tclip = tclip
        self.strategy = strategy

        self.masks = {}

    def nce_loss(self, r_src, r_trg, mask_mat):
        '''
        Compute the NCE scores for predicting r_src->r_trg.
        Input:
          r_src    : (batch_size, emb_dim)
          r_trg    : (emb_dim, n_batch * w* h) (ie: nb_feat_vectors x embedding_dim)
          mask_mat : (n_batch_gpu, n_batch)
        Output:
          raw_scores : (n_batch_gpu, n_locs)
          nce_scores : (n_batch_gpu, n_locs)
          lgt_reg    : scalar
        '''
        # RKHS = embedding dim
        batch_size, emb_dim = r_src.size()
        nb_feat_vectors = r_trg.size(1) // batch_size

        # (b, b) -> (b, b, nb_feat_vectors)
        # all zeros with ones in diagonal tensor... (ie: b1 b1 are all 1s, b1 b2 are all zeros)
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, nb_feat_vectors).float()

        # negative mask
        mask_neg = 1. - mask_pos

        # -------------------------------
        # ALL SCORES COMPUTATION
        # compute src->trg raw scores for batch
        # (b, dim) x (dim, nb_feats*b) -> (b, b, nb_feats)
        # vector for each img in batch times all the vectors of all images in batch
        raw_scores = torch.mm(r_src, r_trg).float()
        raw_scores = raw_scores.reshape(batch_size, batch_size, nb_feat_vectors)

        # -----------------------
        # STABILITY TRICKS
        # trick 1: weighted regularization term
        raw_scores = raw_scores / emb_dim**0.5
        lgt_reg = 5e-2 * (raw_scores**2.).mean()

        # trick 2: tanh clip
        raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)

        '''
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        '''
        # ----------------------
        # EXTRACT POSITIVE SCORES
        # use the index mask to pull all the diagonals which are b1 x b1
        # (batch_size, nb_feat_vectors)
        pos_scores = (mask_pos * raw_scores).sum(dim=1)

        # ----------------------
        # EXTRACT NEGATIVE SCORES
        # pull everything except diagonal and apply clipping
        # (batch_size, batch_size, nb_feat_vectors)
        # diagonals have - clip vals. everything else has actual negative stores
        neg_scores = (mask_neg * raw_scores) - (self.tclip * mask_pos)

        # (batch_size, batch_size * nb_feat_vectors) -> (batch_size, batch_size, nb_feat_vectors)
        neg_scores = neg_scores.reshape(batch_size, -1)
        mask_neg = mask_neg.reshape(batch_size, -1)

        # ---------------------
        # STABLE SOFTMAX
        # max for each row of negative samples
        # will use max in safe softmax
        # (n_batch_gpu, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]

        # DENOMINATOR
        # sum over only negative samples (none from the diagonal)
        neg_sumexp = (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)

        # NUMERATOR
        # compute numerators for the NCE log-softmaxes
        pos_shiftexp = pos_scores - neg_maxes

        # FULL NCE
        nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = -nce_scores.mean()

        return nce_scores, lgt_reg

    def feat_size_w_mask(self, w):
        masks_r5 = np.zeros((w, w, 1, w, w))
        for i in range(w):
            for j in range(w):
                masks_r5[i, j, 0, i, j] = 1
        masks_r5 = torch.tensor(masks_r5).type(torch.uint8)
        masks_r5 = masks_r5.reshape(-1, 1, w, w)
        return masks_r5

    def _sample_src_ftr(self, r_cnv, masks):
        # get feature dimensions
        n_batch = r_cnv.size(0)
        feat_dim = r_cnv.size(1)

        if masks is not None:
            # subsample from conv-ish r_cnv to get a single vector
            mask_idx = torch.randint(0, masks.size(0), (n_batch,))
            mask = masks[mask_idx]
            mask = mask.cuda(r_cnv.device.index)
            r_cnv = torch.masked_select(r_cnv, mask)

        # flatten features for use as globals in glb->lcl nce cost
        r_vec = r_cnv.reshape(n_batch, feat_dim)
        return r_vec

    def one_one_loss(self, x1_maps, x2_maps):
        # (b, dim, w. h)
        batch_size, emb_dim, h, w = x1_maps[0].size()

        mask = self.masks[h]

        # -----------------
        # SOURCE VECTORS
        # 1 feature vector per image per feature map location
        # img1 -> img2
        r1_src_x1 = self._sample_src_ftr(x1_maps[0], mask)
        r1_src_x2 = self._sample_src_ftr(x2_maps[0], mask)

        # pick which map to use for negative samples
        x2_tgt = x2_maps[0]
        x1_tgt = x1_maps[0]

        # adjust the maps for neg samples
        x2_tgt = x2_tgt.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        x1_tgt = x1_tgt.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_src_x1.device.index)

        # -----------------
        # NCE COSTS
        # compute costs for 1->5 prediction
        # use last layer to predict the layer with (5x5 features)
        loss_fwd, regularizer_fwd = self.nce_loss(r1_src_x1, x2_tgt, diag_mat)  # img 1
        loss_back, regularizer_back = self.nce_loss(r1_src_x2, x1_tgt, diag_mat)  # img 1

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss = 0.5 * (loss_fwd + loss_back)
        loss = loss.mean()

        regularizer = 0.5 * (regularizer_fwd + regularizer_back)
        regularizer = regularizer.mean()
        return loss, regularizer

    def strat_15_17_55(self, x1_maps, x2_maps):
        r1_x1, r5_x1, r7_x1 = x1_maps
        r1_x2, r5_x2, r7_x2 = x2_maps

        batch_size, emb_dim, _, _ = r1_x1.size()

        # -----------------
        # SOURCE VECTORS
        # 1 feature vector per image per feature map location
        # img 1
        b_1, e_1, h_1, w_1 = r1_x1.size()
        mask_1 = self.masks[h_1]
        r1_src_1 = self._sample_src_ftr(r1_x1, mask_1)
        r1_src_2 = self._sample_src_ftr(r1_x2, mask_1)

        b_5, e_5, h_5, w_5 = r5_x1.size()
        mask_5 = self.masks[h_5]
        r5_src_1 = self._sample_src_ftr(r5_x1, mask_5)
        r5_src_2 = self._sample_src_ftr(r5_x2, mask_5)

        # -----------------
        # TARGET VECTORS
        # before shape: (n_batch, emb_dim, w, h)
        r5_trg_1 = r5_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r5_trg_2 = r5_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r7_trg_1 = r7_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r7_trg_2 = r7_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        # after shape: (emb_dim, n_batch * w * h)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_x1.device.index)

        # -----------------
        # NCE COSTS
        # compute costs for 1->5 prediction
        # use last layer to predict the layer with (5x5 features)
        loss_1t5_1, regularizer_1t5_1 = self.nce_loss(r1_src_1, r5_trg_2, diag_mat)  # img 1
        loss_1t5_2, regularizer_1t5_2 = self.nce_loss(r1_src_2, r5_trg_1, diag_mat)  # img 2

        # compute costs for 1->7 prediction
        # use last layer to predict the layer with (7x7 features)
        loss_1t7_1, regularizer_1t7_1 = self.nce_loss(r1_src_1, r7_trg_2, diag_mat)  # img 1
        loss_1t7_2, regularizer_1t7_2 = self.nce_loss(r1_src_2, r7_trg_1, diag_mat)  # img 2

        # compute costs for 5->5 prediction
        # use (5x5) layer to predict the (5x5) layer
        loss_5t5_1, regularizer_5t5_1 = self.nce_loss(r5_src_1, r5_trg_2, diag_mat)  # img 1
        loss_5t5_2, regularizer_5t5_2 = self.nce_loss(r5_src_2, r5_trg_1, diag_mat)  # img 2

        # combine costs for optimization
        loss_1t5 = 0.5 * (loss_1t5_1 + loss_1t5_2)
        loss_1t7 = 0.5 * (loss_1t7_1 + loss_1t7_2)
        loss_5t5 = 0.5 * (loss_5t5_1 + loss_5t5_2)

        # regularizer
        regularizer = 0.5 * (regularizer_1t5_1 + regularizer_1t5_2 +
                             regularizer_1t7_1 + regularizer_1t7_2 +
                             regularizer_5t5_1 + regularizer_5t5_2)

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss_1t5 = loss_1t5.mean()
        loss_1t7 = loss_1t7.mean()
        loss_5t5 = loss_5t5.mean()
        regularizer = regularizer.mean()
        return loss_1t5 + loss_1t7 + loss_5t5, regularizer

    def strat_11_55_77(self, x1_maps, x2_maps):
        r1_x1, r5_x1, r7_x1 = x1_maps
        r1_x2, r5_x2, r7_x2 = x2_maps

        batch_size, emb_dim, _, _ = r1_x1.size()

        # -----------------
        # SOURCE VECTORS
        # 1 feature vector per image per feature map location
        # img 1
        b_1, e_1, h_1, w_1 = r1_x1.size()
        mask_1 = self.masks[h_1]
        r1_src_1 = self._sample_src_ftr(r1_x1, mask_1)
        r1_src_2 = self._sample_src_ftr(r1_x2, mask_1)

        # img 2
        b_5, e_5, h_5, w_5 = r5_x1.size()
        mask_5 = self.masks[h_5]
        r5_src_1 = self._sample_src_ftr(r5_x1, mask_5)
        r5_src_2 = self._sample_src_ftr(r5_x2, mask_5)

        b_7, e_7, h_7, w_7 = r7_x1.size()
        mask_7 = self.masks[h_7]
        r7_src_1 = self._sample_src_ftr(r7_x1, mask_7)
        r7_src_2 = self._sample_src_ftr(r7_x2, mask_7)

        # -----------------
        # TARGET VECTORS
        # before shape: (n_batch, emb_dim, w, h)
        r1_trg_1 = r1_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r5_trg_1 = r5_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r7_trg_1 = r7_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        r1_trg_2 = r1_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r5_trg_2 = r5_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        r7_trg_2 = r7_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        # after shape: (emb_dim, n_batch * w * h)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_x1.device.index)

        # -----------------
        # NCE COSTS
        # compute costs for 1->1 prediction
        # use last layer to predict the layer with (5x5 features)
        loss_1t1_1, regularizer_1t1_1 = self.nce_loss(r1_src_1, r1_trg_2, diag_mat)  # img 1
        loss_1t1_2, regularizer_1t1_2 = self.nce_loss(r1_src_2, r1_trg_1, diag_mat)  # img 2

        # compute costs for 5->5 prediction
        # use last layer to predict the layer with (7x7 features)
        loss_5t5_1, regularizer_1t7_1 = self.nce_loss(r5_src_1, r5_trg_2, diag_mat)  # img 1
        loss_5t5_2, regularizer_1t7_2 = self.nce_loss(r5_src_2, r5_trg_1, diag_mat)  # img 2

        # compute costs for 7->7 prediction
        # use (5x5) layer to predict the (5x5) layer
        loss_7t7_1, regularizer_7t7_1 = self.nce_loss(r7_src_1, r7_trg_2, diag_mat)  # img 1
        loss_7t7_2, regularizer_7t7_2 = self.nce_loss(r7_src_2, r7_trg_1, diag_mat)  # img 2

        # combine costs for optimization
        loss_1t1 = 0.5 * (loss_1t1_1 + loss_1t1_2)
        loss_5t5 = 0.5 * (loss_5t5_1 + loss_5t5_2)
        loss_7t7 = 0.5 * (loss_7t7_1 + loss_7t7_2)

        # regularizer
        regularizer = 0.5 * (regularizer_1t1_1 + regularizer_1t1_2 +
                             regularizer_1t7_1 + regularizer_1t7_2 +
                             regularizer_7t7_1 + regularizer_7t7_2)

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss_1t1 = loss_1t1.mean()
        loss_5t5 = loss_5t5.mean()
        loss_7t7 = loss_7t7.mean()
        regularizer = regularizer.mean()
        return loss_1t1 + loss_5t5 + loss_7t7, regularizer

    def strat_1_random(self, x1_maps, x2_maps):
        r1_x1, r5_x1, r7_x1 = x1_maps
        r1_x2, r5_x2, r7_x2 = x2_maps

        batch_size, emb_dim, _, _ = r1_x1.size()

        # -----------------
        # SOURCE VECTORS
        # 1 feature vector per image per feature map location
        # img 1
        b_1, e_1, h_1, w_1 = r1_x1.size()
        mask_1 = self.masks[h_1]

        r1_src_1 = self._sample_src_ftr(r1_x1, mask_1)
        r1_src_2 = self._sample_src_ftr(r1_x2, mask_1)

        # pick the target map
        target_map_idx = np.random.randint(0, len(x2_maps), 1)[0]
        target_map_x2 = x2_maps[target_map_idx]
        target_map_x1 = x1_maps[target_map_idx]

        # -----------------
        # TARGET VECTORS
        # before shape: (n_batch, emb_dim, w, h)
        target_map_x2 = target_map_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)
        target_map_x1 = target_map_x1.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_x1.device.index)

        # -----------------
        # NCE COSTS
        # compute costs for 1->5 prediction
        # use last layer to predict the layer with (5x5 features)
        loss_1tR_1, regularizer_1t5_1 = self.nce_loss(r1_src_1, target_map_x2, diag_mat)  # img 1
        loss_1tR_2, regularizer_1t5_2 = self.nce_loss(r1_src_2, target_map_x1, diag_mat)  # img 2

        # combine costs for optimization
        loss_1tR = 0.5 * (loss_1tR_1 + loss_1tR_2)

        # regularizer
        regularizer = 0.5 * (regularizer_1t5_1 + regularizer_1t5_2)

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss_1tR = loss_1tR.mean()
        regularizer = regularizer.mean()
        return loss_1tR, regularizer

    def forward(self, x1_maps, x2_maps):
        '''
        Compute nce infomax costs for various combos of source/target layers.
        Compute costs in both directions, i.e. from/to both images (x1, x2).
        rK_x1 are features from source image x1.
        rK_x2 are features from source image x2.
        '''
        # cache masks
        if len(self.masks) == 0:
            for m1, m2 in zip(x1_maps, x2_maps):
                batch_size, emb_dim, h, w = m1.size()

                # make mask
                if h not in self.masks:
                    mask = self.feat_size_w_mask(h)
                    mask = mask.cuda(m1.device.index)
                    self.masks[h] = mask

        # if self.strategy == '1:1':
        return self.one_one_loss(x1_maps, x2_maps)
        # elif self.strategy == '1:5,1:7,5:5':
        #     return self.strat_15_17_55(x1_maps, x2_maps)
        # elif self.strategy == '1:1,5:5,7:7':
        #     return self.strat_11_55_77(x1_maps, x2_maps)
        # elif self.strategy == '1:random':
        #     return self.strat_1_random(x1_maps, x2_maps)


class DDTNCE(nn.Module):
    '''
    Input is fixed as r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2.
    '''

    def __init__(self, tclip=10.):
        super(DDTNCE, self).__init__()
        self.masks_r3 = self.generate_sample_mask(3)
        self.masks_r5 = self.generate_sample_mask(5)
        self.masks_r7 = self.generate_sample_mask(7)
        self.masks_r14 = self.generate_sample_mask(14)
        self.masks_r30 = self.generate_sample_mask(30)

        self.tclip = tclip

    def nce_loss(self, r_src, r_trg, mask_mat):
        '''
        Compute the NCE scores for predicting r_src->r_trg.
        Input:
          r_src    : (batch_size, emb_dim)
          r_trg    : (emb_dim, n_batch * w* h) (ie: nb_feat_vectors x embedding_dim)
          mask_mat : (n_batch_gpu, n_batch)
        Output:
          raw_scores : (n_batch_gpu, n_locs)
          nce_scores : (n_batch_gpu, n_locs)
          lgt_reg    : scalar
        '''
        # RKHS = embedding dim

        batch_size, emb_dim = r_src.size()
        nb_feat_vectors = r_trg.size(1) // batch_size

        # (b, b) -> (b, b, nb_feat_vectors)
        # all zeros with ones in diagonal tensor... (ie: b1 b1 are all 1s, b1 b2 are all zeros)
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, nb_feat_vectors).float()

        # negative mask
        mask_neg = 1. - mask_pos

        # -------------------------------
        # ALL SCORES COMPUTATION
        # compute src->trg raw scores for batch
        # (b, dim) x (dim, nb_feats*b) -> (b, b, nb_feats)
        # vector for each img in batch times all the vectors of all images in batch
        raw_scores = torch.mm(r_src, r_trg).float()
        raw_scores = raw_scores.reshape(batch_size, batch_size, nb_feat_vectors)

        # -----------------------
        # STABILITY TRICKS
        # trick 1: weighted regularization term
        raw_scores = raw_scores / emb_dim**0.5
        lgt_reg = 5e-2 * (raw_scores**2.).mean()

        # trick 2: tanh clip
        raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)

        '''
        pos_scores includes scores for all the positive samples
        neg_scores includes scores for all the negative samples, with
        scores for positive samples set to the min score (-self.tclip here)
        '''
        # ----------------------
        # EXTRACT POSITIVE SCORES
        # use the index mask to pull all the diagonals which are b1 x b1
        # (batch_size, nb_feat_vectors)
        pos_scores = (mask_pos * raw_scores).sum(dim=1)

        # ----------------------
        # EXTRACT NEGATIVE SCORES
        # pull everything except diagonal and apply clipping
        # (batch_size, batch_size, nb_feat_vectors)
        # diagonals have - clip vals. everything else has actual negative stores
        neg_scores = (mask_neg * raw_scores) - (self.tclip * mask_pos)

        # (batch_size, batch_size * nb_feat_vectors) -> (batch_size, batch_size, nb_feat_vectors)
        neg_scores = neg_scores.reshape(batch_size, -1)
        mask_neg = mask_neg.reshape(batch_size, -1)

        # ---------------------
        # STABLE SOFTMAX
        # max for each row of negative samples
        # will use max in safe softmax
        # (n_batch_gpu, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]

        # DENOMINATOR
        # sum over only negative samples (none from the diagonal)
        neg_sumexp = (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)

        # NUMERATOR
        # compute numerators for the NCE log-softmaxes
        pos_shiftexp = pos_scores - neg_maxes

        # FULL NCE
        nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = -nce_scores.mean()

        return nce_scores, lgt_reg

    def generate_sample_mask(self, w):
        masks_r5 = np.zeros((w, w, 1, w, w))
        for i in range(w):
            for j in range(w):
                masks_r5[i, j, 0, i, j] = 1
        masks_r5 = torch.tensor(masks_r5).type(torch.uint8)
        masks_r5 = masks_r5.reshape(-1, 1, w, w)

        # use param so gpu is done automatically
        masks_r5 = nn.Parameter(masks_r5, requires_grad=False)

        return masks_r5

    def forward(self, mode, r1_x1, r3_x1, r5_x1, r7_x1, r14_x1, r30_x1, r1_x2, r3_x2, r5_x2, r7_x2, r14_x2, r30_x2):

        if mode == 'random':
            return self.random_way(r1_x1, r3_x1, r5_x1, r7_x1, r14_x1, r30_x1, r1_x2, r3_x2, r5_x2, r7_x2, r14_x2, r30_x2)
        elif mode == 'single_same_layer':
            return self.layer_pairs(r1_x1, r3_x1, r5_x1, r7_x1, r14_x1, r30_x1, r1_x2, r3_x2, r5_x2, r7_x2, r14_x2, r30_x2)
        elif mode == 'single_many_same_layer':
            return self.layer_pairs_all_negatives(r1_x1, r3_x1, r5_x1, r7_x1, r14_x1, r30_x1, r1_x2, r3_x2, r5_x2, r7_x2, r14_x2, r30_x2)

    def _sample_src_ftr(self, r_cnv):
        # get feature dimensions
        b, c, h, w = r_cnv.size()

        mask = getattr(self, f'masks_r{h}')

        rands = torch.randint(0, h*w, (b,))
        r_cnv = torch.masked_select(r_cnv, mask[rands])

        # flatten features for use as globals in glb->lcl nce cost
        r_vec = r_cnv.reshape(b, c)
        return r_vec

    def layer_pairs_all_negatives(self, r1_x1, r3_x1, r5_x1, r7_x1, r14_x1, r30_x1, r1_x2, r3_x2, r5_x2, r7_x2, r14_x2, r30_x2):
        '''
        Compute nce infomax costs for various combos of source/target layers.
        Compute costs in both directions, i.e. from/to both images (x1, x2).
        rK_x1 are features from source image x1.
        rK_x2 are features from source image x2.
        '''
        # (b, dim, w. h)
        batch_size, emb_dim, _, _ = r1_x1.size()

        # -----------------
        # SOURCE VECTOR
        # R1
        r1_x1 = r1_x1.view(batch_size, -1)
        r1_tgt_x2 = r1_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        r3_x1 = self._sample_src_ftr(r3_x1)
        r3_tgt_x2 = r3_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        r5_x1 = self._sample_src_ftr(r5_x1)
        r5_tgt_x2 = r5_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        r7_x1 = self._sample_src_ftr(r7_x1)
        r7_tgt_x2 = r7_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        r14_x1 = self._sample_src_ftr(r14_x1)
        r14_tgt_x2 = r14_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        r30_x1 = self._sample_src_ftr(r30_x1)
        r30_tgt_x2 = r30_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_x1.device.index)

        # -----------------
        # NCE COSTS
        # single vector NCE
        loss_1, reg_1 = self.nce_loss(r1_x1, r1_tgt_x2, diag_mat)
        loss_3, reg_3 = self.nce_loss(r3_x1, r3_tgt_x2, diag_mat)
        loss_5, reg_5 = self.nce_loss(r5_x1, r5_tgt_x2, diag_mat)
        loss_7, reg_7 = self.nce_loss(r7_x1, r7_tgt_x2, diag_mat)
        loss_14, reg_14 = self.nce_loss(r14_x1, r14_tgt_x2, diag_mat)
        loss_30, reg_30 = self.nce_loss(r30_x1, r30_tgt_x2, diag_mat)

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss_1t5 = loss_1.mean() + loss_3.mean() + loss_5.mean() + loss_7.mean() + loss_14.mean() + loss_30.mean()
        regularizer = reg_1.mean() + reg_3.mean() + reg_5.mean() + reg_7.mean() + reg_14.mean() + reg_30.mean()
        return loss_1t5, regularizer

    def layer_pairs(self, r1_x1, r3_x1, r5_x1, r7_x1, r14_x1, r30_x1, r1_x2, r3_x2, r5_x2, r7_x2, r14_x2, r30_x2):
        '''
        Compute nce infomax costs for various combos of source/target layers.
        Compute costs in both directions, i.e. from/to both images (x1, x2).
        rK_x1 are features from source image x1.
        rK_x2 are features from source image x2.
        '''
        # (b, dim, w. h)
        batch_size, emb_dim, _, _ = r1_x1.size()

        # -----------------
        # SOURCE VECTOR
        # R1
        r1_x1 = r1_x1.view(batch_size, -1)
        r1_tgt_x2 = r1_x2.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        r3_x1 = self._sample_src_ftr(r3_x1)
        r3_x2 = self._sample_src_ftr(r3_x2)
        r3_tgt_x2 = r3_x2.permute(1, 0)

        r5_x1 = self._sample_src_ftr(r5_x1)
        r5_x2 = self._sample_src_ftr(r5_x2)
        r5_tgt_x2 = r5_x2.permute(1, 0)

        r7_x1 = self._sample_src_ftr(r7_x1)
        r7_x2 = self._sample_src_ftr(r7_x2)
        r7_tgt_x2 = r7_x2.permute(1, 0)

        r14_x1 = self._sample_src_ftr(r14_x1)
        r14_x2 = self._sample_src_ftr(r14_x2)
        r14_tgt_x2 = r14_x2.permute(1, 0)

        r30_x1 = self._sample_src_ftr(r30_x1)
        r30_x2 = self._sample_src_ftr(r30_x2)
        r30_tgt_x2 = r30_x2.permute(1, 0)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_x1.device.index)

        # -----------------
        # NCE COSTS
        # single vector NCE
        loss_1, reg_1 = self.nce_loss(r1_x1, r1_tgt_x2, diag_mat)
        loss_3, reg_3 = self.nce_loss(r3_x1, r3_tgt_x2, diag_mat)
        loss_5, reg_5 = self.nce_loss(r5_x1, r5_tgt_x2, diag_mat)
        loss_7, reg_7 = self.nce_loss(r7_x1, r7_tgt_x2, diag_mat)
        loss_14, reg_14 = self.nce_loss(r14_x1, r14_tgt_x2, diag_mat)
        loss_30, reg_30 = self.nce_loss(r30_x1, r30_tgt_x2, diag_mat)

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss_1t5 = loss_1.mean() + loss_3.mean() + loss_5.mean() + loss_7.mean() + loss_14.mean() + loss_30.mean()
        regularizer = reg_1.mean() + reg_3.mean() + reg_5.mean() + reg_7.mean() + reg_14.mean() + reg_30.mean()
        return loss_1t5, regularizer

    def random_way(self, r1_x1, r3_x1, r5_x1, r7_x1, r14_x1, r30_x1, r1_x2, r3_x2, r5_x2, r7_x2, r14_x2, r30_x2):
        '''
        Compute nce infomax costs for various combos of source/target layers.
        Compute costs in both directions, i.e. from/to both images (x1, x2).
        rK_x1 are features from source image x1.
        rK_x2 are features from source image x2.
        '''
        # (b, dim, w. h)
        batch_size, emb_dim, _, _ = r1_x1.size()

        # -----------------
        # SOURCE VECTOR
        # 1 feature vector per image per feature map location
        # img 1
        r1_x1 = r1_x1.view(batch_size, -1)

        # pick target map
        options = [r1_x2, r3_x2, r5_x2, r7_x2, r14_x2, r30_x2]
        np.random.shuffle(options)

        x2_map = options[0]
        x2_map = x2_map.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        # make masking matrix to help compute nce costs
        # (b x b) zero matrix with 1s in the diag
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.cuda(r1_x1.device.index)

        # -----------------
        # NCE COSTS
        # single vector NCE
        loss_1t5, regularizer = self.nce_loss(r1_x1, x2_map, diag_mat)

        # ------------------
        # FINAL LOSS MEAN
        # loss mean
        loss_1t5 = loss_1t5.mean()
        regularizer = regularizer.mean()
        return loss_1t5, regularizer


class Evaluator(nn.Module):
    def __init__(self, n_classes, ftr_1=None, dim_1=None, p=0.2):
        super(Evaluator, self).__init__()
        if ftr_1 is None:
            # rely on provided input feature dimensions
            self.dim_1 = dim_1
        else:
            # infer input feature dimensions from provided features
            self.dim_1 = ftr_1.size(1)
        self.n_classes = n_classes
        self.block_glb_mlp = \
            MLPClassifier(self.dim_1, self.n_classes, n_hidden=1024, p=p)
        self.block_glb_lin = \
            MLPClassifier(self.dim_1, self.n_classes, n_hidden=None, p=0.0)

    def forward(self, ftr_1):
        '''
        Input:
          ftr_1 : features at 1x1 layer
        Output:
          lgt_glb_mlp: class logits from global features
          lgt_glb_lin: class logits from global features
        '''
        # collect features to feed into classifiers
        # - always detach() -- send no grad into encoder!
        h_top_cls = flatten(ftr_1)

        # compute predictions
        lgt_glb_mlp = self.block_glb_mlp(h_top_cls)
        lgt_glb_lin = self.block_glb_lin(h_top_cls)
        return lgt_glb_mlp, lgt_glb_lin


class MLPClassifier(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super(MLPClassifier, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


def flatten(x):
    return x.reshape(x.size(0), -1)


def tanh_clip(x, clip_val=10.):
    '''
    soft clip values to the range [-clip_val, +clip_val]
    '''
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip


class TransformsImageNet128:
    '''
    ImageNet dataset, for use with 128x128 full image encoder.
    '''
    def __init__(self):
        # image augmentation functions
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = \
            transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=3)
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(146, interpolation=3),
            transforms.CenterCrop(128),
            post_transform
        ])
        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class CPCTransformsC10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        # main transform for self-supervised training
        self.train_transform = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=8, overlap_size=4),
        ])
        # transform for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=8, overlap_size=4),
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        return out1


class CPCTransformsSTL10Patches:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self, patch_size, overlap):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
        # image augmentation functions
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        rand_crop = \
            transforms.RandomResizedCrop(64, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=3)

        self.test_transform = transforms.Compose([
            transforms.Resize(70, interpolation=3),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=patch_size, overlap_size=overlap)
        ])

        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=patch_size, overlap_size=overlap)
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        return out1


class CPCTransformsImageNet128Patches:
    '''
    ImageNet dataset, for use with 128x128 full image encoder.
    '''
    def __init__(self, patch_size, overlap):
        # image augmentation functions
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = \
            transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=3)
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            Patchify(patch_size=patch_size, overlap_size=overlap),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(146, interpolation=3),
            transforms.CenterCrop(128),
            post_transform
        ])
        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        return out1


class TransformsC10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        # main transform for self-supervised training
        self.train_transform = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize
        ])
        # transform for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class TransformsC10Patches:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self, patch_size=8, patch_overlap=None):
        if patch_overlap is None:
            patch_overlap = patch_size // 2

        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        # main transform for self-supervised training
        self.train_transform = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=patch_size, overlap_size=patch_overlap),
        ])
        # transform for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=patch_size, overlap_size=patch_overlap),
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2

class TransformsC10PatchFirst:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self, patch_size=8, patch_overlap=None):
        if patch_overlap is None:
            patch_overlap = patch_size // 2

        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        self.patch = transforms.Compose([
            transforms.ToTensor(),
            Patchify(patch_size=patch_size, overlap_size=patch_overlap),
        ])
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        # main transform for self-supervised training
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize,
        ])
        # transform for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=patch_size, overlap_size=patch_overlap),
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        inp = self.patch(inp)
        out1 = []
        out2 = []
        for i in range(0, inp.size(0)):
            out1.append(self.train_transform(inp[i]))
            out2.append(self.train_transform(inp[i]))

        out1 = torch.stack(out1)
        out2 = torch.stack(out2)
        return out1, out2

class TransformsImageNet128Patches:
    '''
    ImageNet dataset, for use with 128x128 full image encoder.
    '''
    def __init__(self, patch_size, overlap):
        # image augmentation functions
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = \
            transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=3)
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            Patchify(patch_size=patch_size, overlap_size=overlap),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(146, interpolation=3),
            transforms.CenterCrop(128),
            post_transform
        ])
        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class TransformsC10PatchesSingleChannel:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self, ft=False):
        self.ft = ft
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        # main transform for self-supervised training
        self.train_transform = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            DropChannels(nb_drop_channels=2),
            normalize,
            Patchify(patch_size=8, overlap_size=4),
        ])

        self.finetune_transform = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=8, overlap_size=4),
        ])

        # transform for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=8, overlap_size=4),
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)

        if self.ft:
            out1 = self.finetune_transform(inp)
            out2 = self.finetune_transform(inp)

        else:
            out1 = self.train_transform(inp)
            out2 = self.train_transform(inp)
        return out1, out2


class DropChannels(object):

    def __init__(self, nb_drop_channels):
        self.nb_drop_channels = nb_drop_channels
        self.normal = Normal(0, 1)

    def __call__(self, x):
        c, h, w = x.size()

        idxs = np.random.choice(list(range(c)), replace=False, size=self.nb_drop_channels)
        for i in idxs:
            x[i, :, :] = torch.rand(1, h, w)

        return x


class RandomTranslateWithReflect:
    '''
    Translate image randomly
    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].
    Fill the uncovered blank area with reflect padding.
    '''

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image


import torch.nn.init as init


def init_pytorch_defaults(m, version='041'):
    '''
    Apply default inits from pytorch version 0.4.1 or 1.0.0.
    pytorch 1.0 default inits are wonky :-(
    '''
    if version == '041':
        # print('init.pt041: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == '100':
        # print('init.pt100: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == 'custom':
        # print('init.custom: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        else:
            assert False
    else:
        assert False


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm1d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
