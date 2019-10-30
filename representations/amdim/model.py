import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import pdb

from . import amdim_nets as amdim_utils
from .. import video_transforms, functional_video


# NOTE: This should include jitter and random gray too.
transforms_augment_video = transforms.Compose([
    video_transforms.ToTensorVideo(),
    video_transforms.RandomHorizontalFlipVideo(p=0.5),
    video_transforms.RandomResizedCropVideo(
        128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
        interpolation_mode='bicubic'
    ),
    # jitter,
    # rndgray,
    video_transforms.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
])
transforms_regular_video = transforms.Compose([
    video_transforms.ToTensorVideo(),
    video_transforms.ResizeVideo((146, 146), interpolation='bicubic'),
    video_transforms.CenterCropVideo(128),
    video_transforms.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
])

def load_image(img_path):
    img = np.load(img_path)
    img = Image.fromarray(img)
    return img


def img_loading_func(path, do_augment=False):
    img = load_image(path)
    post_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if do_augment:
        flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = transforms.RandomResizedCrop(
            128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
            interpolation=3)
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        transform = transforms.Compose([
            flip_lr, rand_crop, col_jitter, rnd_gray, post_transform
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(146, interpolation=3),
            transforms.CenterCrop(128),
            post_transform
        ])

    img = transform(img)
    return img


class Model(torch.nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        dummy_batch = torch.zeros((2, 3, hparams.image_height, hparams.image_height))

        self.encoder = amdim_utils.Encoder(
            dummy_batch,
            num_channels=3,
            ndf=hparams.ndf,
            n_rkhs=hparams.n_rkhs,
            n_depth=hparams.n_depth,
            encoder_size=hparams.image_height,
            use_bn=hparams.use_bn
        )
        self.encoder.init_weights()

        # the loss has learnable parameters
        self.nce_loss = amdim_utils.LossMultiNCE(tclip=self.hparams.tclip)

        # self.tng_split = None
        # self.val_split = None

    def forward(self, imgs):
        # NOTE: imgs come in as [bs, nf, 3, 128, 128]
        bs, nf, ch, h, w = imgs.size()
        imgs = imgs.reshape([bs * nf, ch, h, w])
        
        # feats for img 1
        # r1 = last layer out
        # r5 = last layer with (b, c, 5, 5) size
        # r7 = last layer with (b, c, 7, 7) size
        r1_x1, r5_x1, r7_x1 = self.encoder(imgs)
        # per img: r1_x1 - 2560, r5_x1 - 2560x5x5, r7x1 - 2560x7x7
        # that's a total of 192000
        r1_x1 = r1_x1.reshape([bs*nf, 2560, np.prod(r1_x1.shape[2:])])
        r5_x1 = r5_x1.reshape([bs*nf, 2560, np.prod(r5_x1.shape[2:])])
        r7_x1 = r7_x1.reshape([bs*nf, 2560, np.prod(r7_x1.shape[2:])])
        representation = torch.cat([r1_x1, r5_x1, r7_x1], axis=2)
        return representation

    # @pl.data_loader
    def tng_dataloader(self):
        if self.hparams.dataset_name == 'imagenet_128':
            train_transform = amdim_utils.TransformsImageNet128()
            dataset = UnlabeledImagenet(self.hparams.imagenet_data_files_tng,
                                        nb_classes=self.hparams.nb_classes,
                                        split='train',
                                        transform=train_transform)

            dist_sampler = None
            if self.trainer.use_ddp:
                dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
                sampler=dist_sampler
            )

            return loader

    # @pl.data_loader
    def val_dataloader(self):
        if self.hparams.dataset_name == 'imagenet_128':
            train_transform = amdim_utils.TransformsImageNet128()
            dataset = UnlabeledImagenet(self.hparams.imagenet_data_files_val,
                                        nb_classes=self.hparams.nb_classes,
                                        split='val',
                                        transform=train_transform)

            dist_sampler = None
            if self.trainer.use_ddp:
                dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
                sampler=dist_sampler
            )

            return loader

    @staticmethod
    def translate(pretrained):
        return {
            'representation_model.%s' % k: v
            for k, v in pretrained.items()
        }
