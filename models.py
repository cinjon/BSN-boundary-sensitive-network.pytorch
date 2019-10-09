# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from torch.nn import init
from representations.ccc.model import Model as CCCModel
from representations.ccc.model import img_loading_func as ccc_img_loading_func
from representations.ccc.representation import Representation as CCCRepresentation
from representations.ccc.representation import THUMOS_OUTPUT_DIM as CCCThumosDim
from representations.ccc.representation import GYMNASTICS_OUTPUT_DIM as CCCGymnasticsDim

from representations.corrflow.model import Model as CorrFlowModel
from representations.corrflow.model import img_loading_func as corrflow_img_loading_func
from representations.corrflow.representation import Representation as CorrFlowRepresentation
from representations.corrflow.representation import THUMOS_OUTPUT_DIM as CorrFlowThumosDim
from representations.corrflow.representation import GYMNASTICS_OUTPUT_DIM as CorrFlowGymnasticsDim


def _get_module(key):
    return {
        # 'timecycle': CycleTime,
        'corrflow-thumosimages':
            (CorrFlowModel, CorrFlowRepresentation, corrflow_img_loading_func, CorrFlowThumosDim),
        'corrflow-gymnastics':
            (CorrFlowModel, CorrFlowRepresentation, corrflow_img_loading_func, CorrFlowGymnasticsDim),
        'ccc-thumosimages':
            (CCCModel, CCCRepresentation, ccc_img_loading_func, CCCThumosDim),
        'ccc-gymnastics':
            (CCCModel, CCCRepresentation, ccc_img_loading_func, CCCGymnasticsDim)
        
    }.get(key)


def get_img_loader(opt):
    key = '%s-%s' % (opt['representation_module'], opt['dataset'])
    _, _, img_loading_func, _ = _get_module(key)
    return img_loading_func


def partial_load(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = model.translate(pretrained_dict)
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


# class TEMGC(torch.nn.Module):
#     def __init__(self, opt):
#         super(TEM, self).__init__()

#         self.temporal_dim = opt["temporal_scale"]
#         self.nonlinear_factor = opt["tem_nonlinear_factor"]
#         self.batch_size = opt["tem_batch_size"]
#         self.num_videoframes = opt["num_videoframes"]
#         self.c_hidden = opt["tem_hidden_dim"]
#         self.feat_dim = opt["tem_feat_dim"]

#         key = '%s-%s' % (opt['representation_module'], opt['dataset'])
#         model, representation, _, representation_dim = _get_module(key)
#         self.features = nn.Sequential(OrderedDict([
            
#         self.representation_model = model(opt)
#             if self.do_feat_conversion:
#                 self.representation_mapping = representation(opt)
#             else:
#                 self.feat_dim = representation_dim
#         print('Feature Dimension: ', self.feat_dim)
                
#         self.tem_best_loss = 10000000
#         self.output_dim = 3

#         self.conv1 = torch.nn.Conv1d(in_channels=self.feat_dim,
#                                      out_channels=self.c_hidden,
#                                      kernel_size=3,
#                                      stride=1,
#                                      padding=1,
#                                      groups=1)
#         self.conv2 = torch.nn.Conv1d(in_channels=self.c_hidden,
#                                      out_channels=self.c_hidden,
#                                      kernel_size=3,
#                                      stride=1,
#                                      padding=1,
#                                      groups=1)
#         self.conv3 = torch.nn.Conv1d(in_channels=self.c_hidden,
#                                      out_channels=self.output_dim,
#                                      kernel_size=1,
#                                      stride=1,
#                                      padding=0)

#         if opt['tem_reset_params']:
#             self.reset_params()

            
class TEM(torch.nn.Module):

    def __init__(self, opt):
        super(TEM, self).__init__()

        self.temporal_dim = opt["temporal_scale"]
        self.nonlinear_factor = opt["tem_nonlinear_factor"]
        self.batch_size = opt["tem_batch_size"]
        self.num_videoframes = opt["num_videoframes"]
        self.c_hidden = opt["tem_hidden_dim"]
        self.do_representation = opt['do_representation']
        self.do_feat_conversion = opt['do_feat_conversion']
        self.feat_dim = opt["tem_feat_dim"]
        self.do_gradient_checkpointing = opt['do_gradient_checkpointing']
        
        if self.do_representation:
            key = '%s-%s' % (opt['representation_module'], opt['dataset'])
            model, representation, _, representation_dim = _get_module(key)
            self.representation_model = model(opt)
            if self.do_feat_conversion:
                self.representation_mapping = representation(opt)
            else:
                self.feat_dim = representation_dim
        print('Feature Dimension: ', self.feat_dim)
                
        self.tem_best_loss = 10000000
        self.output_dim = 3

        self.conv1 = torch.nn.Conv1d(in_channels=self.feat_dim,
                                     out_channels=self.c_hidden,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.c_hidden,
                                     out_channels=self.c_hidden,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=1)
        self.conv3 = torch.nn.Conv1d(in_channels=self.c_hidden,
                                     out_channels=self.output_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)

        if opt['tem_reset_params']:
            self.reset_params()

    def set_eval_representation(self):
        self.representation_model.eval()
        if self.do_feat_conversion:
            self.representation_mapping.eval()

    def translate(self, pretrained):
        return self.representation_model.translate(pretrained)

    def img_loading_func(self):
        return getattr(self.representation_model, 'img_loading_func')

    @staticmethod
    def weight_init(m):
        print('Doing weight init!!')
        if isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        if self.do_representation:
            # Input is [bs, num_videoframes, 3, 256, 448]
            with torch.no_grad():
                x = self.representation_model(x)
            if self.do_feat_conversion:
                x = self.representation_mapping(x)
                adj_batch_size, num_features = x.shape
                # This might be different because of data parallelism
                batch_size = int(adj_batch_size / self.num_videoframes)
                x = x.view(batch_size, num_features, self.num_videoframes)
            else:
                adj_batch_size = x.shape[0]
                batch_size = int(adj_batch_size / self.num_videoframes)
                x = x.reshape(batch_size, -1, self.num_videoframes)
        else:
            # From the Thumos features...
            x = x.transpose(1, 2)

        if self.do_gradient_checkpointing:
            modules = [module for k, module in self._modules.items()]
            segments = 2
            x = checkpoint_sequential(modules, segments, x)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.nonlinear_factor * self.conv3(x))
        return x


class PEM(torch.nn.Module):

    def __init__(self, opt):
        super(PEM, self).__init__()

        self.feat_dim = opt["pem_feat_dim"]
        self.batch_size = opt["pem_batch_size"]
        self.hidden_dim = opt["pem_hidden_dim"]
        self.nonlinear_factor = opt["pem_nonlinear_factor"]
        self.output_dim = 1
        self.pem_best_loss = 1000000

        self.fc1 = torch.nn.Linear(in_features=self.feat_dim,
                                   out_features=self.hidden_dim,
                                   bias=True)
        self.fc2 = torch.nn.Linear(in_features=self.hidden_dim,
                                   out_features=self.output_dim,
                                   bias=True)

    def forward(self, x):
        x = F.relu(self.nonlinear_factor * self.fc1(x))
        x = torch.sigmoid(self.nonlinear_factor * self.fc2(x))
        return x
