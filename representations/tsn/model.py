import cv2
import numpy as np
import mmcv
from . import TSN2D as tsn_helper


GYMNASTICS_OUTPUT_DIM = 1024


def img_loading_func(path, do_augment=False):
    mean=[104, 117, 128]
    std=[1, 1, 1]    
    path = str(path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    scale = (np.Inf, 256)
    img_group = [img]
    tuple_list = [mmcv.imrescale(
        img, scale, return_scale=True) for img in img_group]
    img_group, scale_factors = list(zip(*tuple_list))
    scale_factor = scale_factors[0]
    
    op_crop = GroupCenterCrop(224)
    img_group, crop_quadruple = op_crop(img_group, is_flow=False)
    img_shape = img_group[0].shape    
    if do_augment and np.random.rand() < 0.5:
        img_group = [mmcv.imflip(img) for img in img_group]

    img_group = [
        mmcv.imnormalize(img, mean, std, to_rgb=False)
        for img in img_group
    ]
    img_group = [img.transpose(2, 0, 1) for img in img_group]
    return img_group[0]
    

def tsn_model(opt):
    cfg = mmcv.Config.fromfile(opt['tsn_config'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if cfg.data.test.oversample == 'three_crop':
        cfg.model.spatial_temporal_module.spatial_size = 8

    model = tsn_helper.build_recognizer(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    return model


class GroupCenterCrop(object):
    def __init__(self, size):
        self.size = size if not isinstance(size, int) else (size, size)

    def __call__(self, img_group, is_flow=False):
        h = img_group[0].shape[0]
        w = img_group[0].shape[1]
        tw, th = self.size
        x1 = (w - tw) // 2
        y1 = (h - th) // 2
        box = np.array([x1, y1, x1+tw-1, y1+th-1])
        return ([mmcv.imcrop(img, box) for img in img_group],
                np.array([x1, y1, tw, th], dtype=np.float32))

