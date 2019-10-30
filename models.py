# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn import init

from representations.ccc.model import Model as CCCModel
from representations.ccc.model import img_loading_func as ccc_img_loading_func
from representations.ccc.model import transforms_augment_video as ccc_augment_transforms
from representations.ccc.model import transforms_regular_video as ccc_regular_transforms
from representations.ccc.representation import Representation as CCCRepresentation
from representations.ccc.representation import THUMOS_OUTPUT_DIM as CCCThumosDim
from representations.ccc.representation import GYMNASTICS_OUTPUT_DIM as CCCGymnasticsDim
from representations.ccc.representation import ACTIVITYNET_OUTPUT_DIM as CCCActivitynetDim

from representations.corrflow.model import Model as CorrFlowModel
from representations.corrflow.model import img_loading_func as corrflow_img_loading_func
from representations.corrflow.model import transforms_augment_video as corrflow_augment_transforms
from representations.corrflow.model import transforms_regular_video as corrflow_regular_transforms
from representations.corrflow.representation import Representation as CorrFlowRepresentation
from representations.corrflow.representation import THUMOS_OUTPUT_DIM as CorrFlowThumosDim
from representations.corrflow.representation import GYMNASTICS_OUTPUT_DIM as CorrFlowGymnasticsDim
from representations.corrflow.representation import ACTIVITYNET_OUTPUT_DIM as CorrFlowActivitynetDim

from representations.resnet.model import Model as ResnetModel
from representations.resnet.model import img_loading_func as resnet_img_loading_func
from representations.resnet.model import transforms_augment_video as resnet_augment_transforms
from representations.resnet.model import transforms_regular_video as resnet_regular_transforms
from representations.resnet.representation import Representation as ResnetRepresentation
from representations.resnet.representation import THUMOS_OUTPUT_DIM as ResnetThumosDim
from representations.resnet.representation import GYMNASTICS_OUTPUT_DIM as ResnetGymnasticsDim
from representations.resnet.representation import ACTIVITYNET_OUTPUT_DIM as ResnetActivitynetDim

from representations.amdim.model import Model as AMDIMModel
from representations.amdim.model import img_loading_func as amdim_img_loading_func
from representations.amdim.model import transforms_augment_video as amdim_augment_transforms
from representations.amdim.model import transforms_regular_video as amdim_regular_transforms
from representations.amdim.representation import Representation as AMDIMRepresentation
from representations.amdim.representation import THUMOS_OUTPUT_DIM as AMDIMThumosDim
from representations.amdim.representation import GYMNASTICS_OUTPUT_DIM as AMDIMGymnasticsDim
from representations.amdim.representation import ACTIVITYNET_OUTPUT_DIM as AMDIMActivityNetDim



def _get_module(key):
    return {
        # 'timecycle': CycleTime,
        'corrflow-thumosimages': (
            CorrFlowModel, CorrFlowRepresentation, corrflow_img_loading_func, CorrFlowThumosDim
        ),
        'corrflow-gymnastics': (
            CorrFlowModel, CorrFlowRepresentation, corrflow_img_loading_func, CorrFlowGymnasticsDim
        ),
        'corrflow-activitynet': (
            CorrFlowModel, CorrFlowRepresentation, corrflow_img_loading_func, CorrFlowActivitynetDim
        ),
        'ccc-thumosimages': (
            CCCModel, CCCRepresentation, ccc_img_loading_func, CCCThumosDim
        ),
        'ccc-gymnastics': (
            CCCModel, CCCRepresentation, ccc_img_loading_func, CCCGymnasticsDim
        ),
        'ccc-activitynet': (
            CCCModel, CCCRepresentation, ccc_img_loading_func, CCCActivitynetDim
        ),
        'resnet-thumosimages': (
            ResnetModel, ResnetRepresentation, resnet_img_loading_func, ResnetThumosDim
        ),
        'resnet-gymnastics': (
            ResnetModel, ResnetRepresentation, resnet_img_loading_func, ResnetGymnasticsDim
        ),
        'resnet-activitynet': (
            ResnetModel, ResnetRepresentation, resnet_img_loading_func, ResnetActivitynetDim
        ),
        'amdim-thumosimages': (
            AMDIMModel, AMDIMRepresentation, amdim_img_loading_func, AMDIMThumosDim
        ),
        'amdim-gymnastics': (
            AMDIMModel, AMDIMRepresentation, amdim_img_loading_func, AMDIMGymnasticsDim
        ),
        'amdim-activitynet': (
            AMDIMModel, AMDIMRepresentation, amdim_img_loading_func, AMDIMActivityNetDim
        )        
    }.get(key)


def get_img_loader(opt):
    key = '%s-%s' % (opt['representation_module'], opt['dataset'])
    _, _, img_loading_func, _ = _get_module(key)
    return img_loading_func


def get_video_transforms(representation_module, do_augment):
    augment = 'augment' if do_augment else 'regular'
    key = '%s-%s' % (representation_module, augment)
    return {
        'amdim-augment': amdim_augment_transforms,
        'amdim-regular': amdim_regular_transforms,
        'ccc-augment': ccc_augment_transforms,
        'ccc-regular': ccc_regular_transforms,
        'corrflow-augment': corrflow_augment_transforms,
        'corrflow-regular': corrflow_regular_transforms,
        'resnet-augment': resnet_augment_transforms,
        'resnet-regular': resnet_regular_transforms,
    }.get(key)



def partial_load(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = model.translate(pretrained_dict)
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


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
            tags_csv = opt['representation_tags']
            if tags_csv:
                hparams = load_hparams_from_tags_csv(tags_csv)
                hparams.__setattr__('on_gpu', False)
                self.representation_model = model(hparams)
            else:
                self.representation_model = model(opt)
                
            if self.do_feat_conversion:
                self.representation_mapping = representation(opt)
            else:
                self.feat_dim = representation_dim
                
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

    def _get_representation(self, x):
        # Input is [bs, num_videoframes, 3, 256, 448]
        with torch.no_grad():
            x = self.representation_model(x)
            x = x.detach()

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
        return x
            
    def forward(self, x):
        if self.do_representation:
            x = self._get_representation(x)
        else:
            x = x.transpose(1, 2)

        if self.do_gradient_checkpointing:
            x = F.relu(checkpoint(self.conv1, x))
            x = F.relu(checkpoint(self.conv2, x))
            x = checkpoint(self.conv3, x)
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)
            
        return torch.sigmoid(self.nonlinear_factor * x)


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


def load_hparams_from_tags_csv(tags_csv):
    from argparse import Namespace
    import pandas as pd

    tags_df = pd.read_csv(tags_csv)
    dic = tags_df.to_dict(orient='records')
    
    ns_dict = {row['key']: convert(row['value']) for row in dic}
    
    ns = Namespace(**ns_dict)
    return ns


def convert(val):
    constructors = [int, float, str]
    
    if type(val) is str:
        if val.lower() == 'true':
            return True
        if val.lower() == 'false':
            return False
        
    for c in constructors:
        try:
            return c(val)
        except ValueError:
            pass
    return val
