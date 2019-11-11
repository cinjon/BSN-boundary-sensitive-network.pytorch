import logging

import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from .TSN2D import BACKBONES


__all__ = ['BNInception']

