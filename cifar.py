from collections import namedtuple
import os
import cv2
import time
import math
import pickle
import random
import argparse
import datetime
import numpy as np
from glob import glob
from PIL import Image

from comet_ml import Experiment as CometExperiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import run_cifar_jobs
import opts
from models import load_hparams_from_tags_csv
from representations.amdim.model import Model as AMDIMModel
from representations.amdim.representation import THUMOS_OUTPUT_DIM as AMDIM_OUTPUT_DIM
from representations.ccc.model import Model as CCCModel
# from representations.ccc.representation import ACTIVITYNET_OUTPUT_DIM as CCC_OUTPUT_DIM
CCC_OUTPUT_DIM = 8192 # 184832
from representations.corrflow.model import Model as CORRFLOWModel
CORRFLOW_OUTPUT_DIM = 64 * 8 * 8
RESNET_OUTPUT_DIM = 2048

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="run", help="run, array, or job")
parser.add_argument("--not_pretrain", action="store_true", default=False)
parser.add_argument('--name',
                    type=str,
                    help='the identifying name of this experiment.',
                    default=None)
parser.add_argument('--counter',
                    type=int,
                    help='the integer counter of this experiment. '
                    'defaults to None because Cinjon is likely the '
                    'only one who is going to use it.')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--lr_interval", type=int, default=20)
parser.add_argument("--min_lr", type=float, default=0.0003)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument(
    "--model", type=str, help="amdim, ccc, corrflow or resnet")
parser.add_argument(
    "--num_classes",
    type=int,
    help="10 for CIFAR10 or 100 for CIFAR100")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#######################################################
# AMDIM Image loading
#######################################################

def amdim_img_loading_func(img, do_augment=False):
    img = Image.fromarray(img)
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
            # transforms.Resize(146, interpolation=3),
            # transforms.CenterCrop(128),
            transforms.Resize(128, interpolation=3),
            post_transform
        ])

    img = transform(img)
    return img


#######################################################
# CCC Image loading
#######################################################

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    return img

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)

def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    img = cv2.resize( img, (owidth, oheight) )
    img = im_to_torch(img)
    return img

def ccc_img_loading_func(img, do_augment=False):
    imgSize = 32 # 256

    img = img.astype(np.float32) / 255.0
    img = im_to_torch(img.copy())
    ht, wd = img.size(1), img.size(2)
    if ht <= wd:
        ratio  = float(wd) / float(ht)
        # width, height
        img = resize(img, int(imgSize * ratio), imgSize)
    else:
        ratio  = float(ht) / float(wd)
        # width, height
        img = resize(img, imgSize, int(imgSize * ratio))

    if do_augment:
        if random.random() > 0.5:
            img = torch.from_numpy(fliplr(img.numpy())).float()

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img = color_normalize(img, mean, std)
    return img


#######################################################
# CorrFlow Image loading
#######################################################

def rgb_preprocess_jitter(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    # image = transforms.Resize(224)(image)
    image = transforms.ColorJitter(0.1,0.1,0.1,0.1)(image)
    image = transforms.ToTensor()(image)
    return image

def corrflow_img_loading_func(img, do_augment=False):
    # This should output RGB.
    img = img / 255.
    img = img.astype(np.float32)

    M = 8
    h, w = img.shape[0], img.shape[1]
    if w % M != 0: img = img[:, :-(w % M)]
    if h % M != 0: img = img[:-(h % M),]
    r = np.random.random()
    # The random was originally 0.1.
    if not do_augment or np.random.random() > 0.1:
        return transforms.ToTensor()(img)
        # img = Image.fromarray(img)
        # img = transforms.Resize(224)(img)
        # img = transforms.ToTensor()(img)
        return img
    return rgb_preprocess_jitter(img)

#######################################################
# Resnet Image loading
#######################################################

def resnet_img_loading_func(img, do_augment=False):
    img = Image.fromarray(img)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    if do_augment:
        transforms_ = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transforms_ = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms_(img)


#######################################################
# CIFAR Dataset
#######################################################

def unpickle(file):
    with open(file, "rb") as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

class CIFAR_dataset(data.Dataset):
    def __init__(self, paths, num_classes, model_name, do_augment):
        self.paths = paths
        self.num_classes = num_classes
        self.do_augment = do_augment
        self.model_name = model_name

        self.imgs = []
        self.lbls = []
        for path in self.paths:
            data_dic = unpickle(path)
            self.imgs.append(data_dic[b"data"])
            if self.num_classes == 10:
                self.lbls.append(data_dic[b"labels"])
            if self.num_classes == 100:
                self.lbls.append(data_dic[b"fine_labels"])

        self.imgs = np.concatenate(self.imgs, 0)
        self.imgs = np.reshape(self.imgs, (-1, 3, 32, 32))
        self.imgs = np.transpose(self.imgs, (0, 2, 3, 1))
        self.lbls = np.concatenate(self.lbls, 0)
        assert len(self.imgs) == len(self.lbls), "imgs and lbls not match"

    def __getitem__(self, index):
        if self.model_name == "amdim":
            img = amdim_img_loading_func(
                self.imgs[index], do_augment=self.do_augment)
        elif self.model_name == "ccc":
            img = ccc_img_loading_func(
                self.imgs[index], do_augment=self.do_augment)
        elif self.model_name == "corrflow":
            img = corrflow_img_loading_func(
                self.imgs[index], do_augment=self.do_augment)
        elif self.model_name == "resnet":
            img = resnet_img_loading_func(
                self.imgs[index], do_augment=self.do_augment)
        lbl = torch.LongTensor([self.lbls[index]])
        return img, lbl

    def __len__(self):
        return len(self.imgs)


#######################################################
# Linear model
#######################################################

class LinearModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LinearModel, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.linear = nn.Linear(self.in_channels, self.num_classes)
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.zero_()

        # self.fc1 = nn.Linear(self.in_channels, 512)
        # self.fc2 = nn.Linear(512, self.num_classes)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc1.weight.data.normal_(0, 0.01)
        # self.fc1.bias.data.zero_()
        # self.fc2.weight.data.normal_(0, 0.01)
        # self.fc2.bias.data.zero_()


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        # x = self.relu(self.fc1(x))
        # x = self.fc2(x)
        return x

#######################################################
# Main
#######################################################

def main(args):
    print('Pretrain? ', not args.not_pretrain)
    print(args.model)

    comet_exp = CometExperiment(api_key="hIXq6lDzWzz24zgKv7RYz6blo",
                                project_name="selfcifar",
                                workspace="cinjon",
                                auto_metric_logging=True,
                                auto_output_logging=None,
                                auto_param_logging=False)
    comet_exp.log_parameters(vars(args))
    comet_exp.set_name(args.name)

    # Build model
    path = "/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/bsn"
    if args.model == "amdim":
        hparams = load_hparams_from_tags_csv(os.path.join(path, "meta_tags.csv"))
        model = AMDIMModel(hparams)
        if not args.not_pretrain:
            model.load_state_dict(
                torch.load(os.path.join(path, "_ckpt_epoch_434.ckpt"))["state_dict"])
        else:
            print("AMDIM not loading checkpoint") # Debug
        linear_model = LinearModel(AMDIM_OUTPUT_DIM, args.num_classes)
    elif args.model == "ccc":
        model = CCCModel(None)
        if not args.not_pretrain:
            checkpoint = torch.load(os.path.join(path, "TimeCycleCkpt14.pth"))
            base_dict = {
                '.'.join(k.split('.')[1:]): v
                for k, v in list(checkpoint['state_dict'].items())}
            model.load_state_dict(base_dict)
        else:
            print("CCC not loading checkpoint") # Debug
        linear_model = LinearModel(CCC_OUTPUT_DIM, args.num_classes).to(device)
    elif args.model == "corrflow":
        model = CORRFLOWModel(None)
        if not args.not_pretrain:
            checkpoint = torch.load(os.path.join(path, "corrflow.kineticsmodel.pth"))
            base_dict = {
                '.'.join(k.split('.')[1:]): v
                for k, v in list(checkpoint['state_dict'].items())}
            model.load_state_dict(base_dict)
        else:
            print("CorrFlow not loading checkpoing") # Debug
        linear_model = LinearModel(CORRFLOW_OUTPUT_DIM, args.num_classes)
    elif args.model == "resnet":
        if not args.not_pretrain:
            resnet = torchvision.models.resnet50(pretrained=True)
        else:
            resnet = torchvision.models.resnet50(pretrained=False)
            print("ResNet not loading checkpoint") # Debug
        modules = list(resnet.children())[:-1]
        model = nn.Sequential(*modules)
        linear_model = LinearModel(RESNET_OUTPUT_DIM, args.num_classes)
    else:
        raise Exception("model type has to be amdim, ccc, corrflow or resnet")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
        linear_model = nn.DataParallel(linear_model).to(device)
    else:
        model = model.to(device)
        linear_model = linear_model.to(device)
    # model = model.to(device)
    # linear_model = linear_model.to(device)

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    if args.optimizer == "Adam":
        optimizer = optim.Adam(
            linear_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)
        print("Optimizer: Adam with weight decay: {}".format(
            args.weight_decay))
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(
            linear_model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        print("Optimizer: SGD with weight decay: {} momentum: {}".format(
            args.weight_decay, args.momentum))
    else:
        raise Exception("optimizer should be Adam or SGD")
    optimizer.zero_grad()

    # Set up log dir
    now = datetime.datetime.now()
    log_dir = "{}{:%Y%m%dT%H%M}".format(args.model, now)
    log_dir = os.path.join("weights", log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("Saving to {}".format(log_dir))

    batch_size = args.batch_size * torch.cuda.device_count()
    # CIFAR-10
    if args.num_classes == 10:
        data_path = ("/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/"
                     "bsn/data/cifar-10-batches-py")
        _train_dataset = CIFAR_dataset(
            glob(os.path.join(data_path, "data*")),
            args.num_classes,
            args.model,
            True)
        # _train_acc_dataset = CIFAR_dataset(
        #     glob(os.path.join(data_path, "data*")),
        #     args.num_classes,
        #     args.model,
        #     False)
        train_dataloader = data.DataLoader(
            _train_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers)
        # train_split = int(len(_train_dataset) * 0.8)
        # train_dev_split = int(len(_train_dataset) - train_split)
        # train_dataset, train_dev_dataset = data.random_split(
        #     _train_dataset, [train_split, train_dev_split])
        # train_acc_dataloader = data.DataLoader(
        #     train_dataset, shuffle=False, batch_size=batch_size, num_workers=args.num_workers)
        # train_dev_acc_dataloader = data.DataLoader(
        #     train_dev_dataset, shuffle=False, batch_size=batch_size, num_workers=args.num_workers)
        # train_dataset = data.Subset(_train_dataset, list(range(train_split)))
        # train_dataloader = data.DataLoader(
        #     train_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers)
        # train_acc_dataset = data.Subset(
        #     _train_acc_dataset, list(range(train_split)))
        # train_acc_dataloader = data.DataLoader(
        #     train_acc_dataset, shuffle=False, batch_size=batch_size, num_workers=args.num_workers)
        # train_dev_acc_dataset = data.Subset(
        #     _train_acc_dataset, list(range(train_split, len(_train_acc_dataset))))
        # train_dev_acc_dataloader = data.DataLoader(
        #     train_dev_acc_dataset, shuffle=False, batch_size=batch_size, num_workers=args.num_workers)

        _val_dataset = CIFAR_dataset(
            [os.path.join(data_path, "test_batch")],
            args.num_classes,
            args.model,
            False)
        val_dataloader = data.DataLoader(
            _val_dataset, shuffle=False, batch_size=batch_size, num_workers=args.num_workers)
        # val_split = int(len(_val_dataset) * 0.8)
        # val_dev_split = int(len(_val_dataset) - val_split)
        # val_dataset, val_dev_dataset = data.random_split(
        #     _val_dataset, [val_split, val_dev_split])
        # val_dataloader = data.DataLoader(
        #     val_dataset, shuffle=False, batch_size=batch_size, num_workers=args.num_workers)
        # val_dev_dataloader = data.DataLoader(
        #     val_dev_dataset, shuffle=False, batch_size=batch_size, num_workers=args.num_workers)
    # CIFAR-100
    elif args.num_classes == 100:
        data_path = ("/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/zeping/"
                     "bsn/data/cifar-100-python")
        _train_dataset = CIFAR_dataset(
            [os.path.join(data_path, "train")],
            args.num_classes,
            args.model,
            True)
        train_dataloader = data.DataLoader(
            _train_dataset, shuffle=True, batch_size=batch_size)
        _val_dataset = CIFAR_dataset(
            [os.path.join(data_path, "test")],
            args.num_classes,
            args.model,
            False)
        val_dataloader = data.DataLoader(
            _val_dataset, shuffle=False, batch_size=batch_size)
    else:
        raise Exception("num_classes should be 10 or 100")

    best_acc = 0.0
    best_epoch = 0

    # Training
    for epoch in range(1, args.epochs+1):
        current_lr = max(3e-4, args.lr *\
            math.pow(0.5, math.floor(epoch / args.lr_interval)))
        linear_model.train()
        if args.optimizer == "Adam":
            optimizer = optim.Adam(
                linear_model.parameters(),
                lr=current_lr,
                weight_decay=args.weight_decay)
        elif args.optimizer == "SGD":
            optimizer = optim.SGD(
                linear_model.parameters(),
                lr=current_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,)

        ####################################################
        # Train
        t = time.time()
        train_acc = 0
        train_loss_sum = 0.0
        for iter, input in enumerate(train_dataloader):
            imgs = input[0].to(device)
            if args.model != "resnet":
                imgs = imgs.unsqueeze(1)
            lbls = input[1].flatten().to(device)

            # output = model(imgs)
            # output = linear_model(output)
            output = linear_model(model(imgs))
            loss = F.cross_entropy(output, lbls)
            train_loss_sum += float(loss.data)
            train_acc += int(sum(torch.argmax(output, dim=1) == lbls))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log_text = "train epoch {}/{}\titer {}/{} loss:{} {:.3f}s/iter"
            if iter % 1500 == 0:
                log_text = "train epoch {}/{}\titer {}/{} loss:{}"
                print(log_text.format(
                    epoch,
                    args.epochs,
                    iter+1,
                    len(train_dataloader),
                    loss.data,
                    time.time() - t),
                      flush=False)
                t = time.time()

        train_acc /= len(_train_dataset)
        train_loss_sum /= len(train_dataloader)
        with comet_exp.train():
            comet_exp.log_metrics({'acc': train_acc, 'loss': train_loss_sum}, step=(epoch + 1) * len(train_dataloader), epoch=epoch + 1)
        print("train acc epoch {}/{} loss:{} train_acc:{}".format(
            epoch, args.epochs, train_loss_sum, train_acc), flush=True)

        #######################################################################
        # Train acc
        # linear_model.eval()
        # train_acc = 0
        # train_loss_sum = 0.0
        # for iter, input in enumerate(train_acc_dataloader):
        #     imgs = input[0].to(device)
        #     if args.model != "resnet":
        #         imgs = imgs.unsqueeze(1)
        #     lbls = input[1].flatten().to(device)
        #
        #     # output = model(imgs)
        #     # output = linear_model(output)
        #     output = linear_model(model(imgs))
        #     loss = F.cross_entropy(output, lbls)
        #     train_loss_sum += float(loss.data)
        #     train_acc += int(sum(torch.argmax(output, dim=1) == lbls))
        #
        #     print("train acc epoch {}/{}\titer {}/{} loss:{} {:.3f}s/iter".format(
        #         epoch,
        #         args.epochs,
        #         iter+1,
        #         len(train_acc_dataloader),
        #         loss.data,
        #         time.time() - t),
        #         flush=True)
        #     t = time.time()
        #
        #
        # train_acc /= len(train_acc_dataset)
        # train_loss_sum /= len(train_acc_dataloader)
        # print("train acc epoch {}/{} loss:{} train_acc:{}".format(
        #     epoch, args.epochs, train_loss_sum, train_acc), flush=True)

        #######################################################################
        # Train dev acc
        # # linear_model.eval()
        # train_dev_acc = 0
        # train_dev_loss_sum = 0.0
        # for iter, input in enumerate(train_dev_acc_dataloader):
        #     imgs = input[0].to(device)
        #     if args.model != "resnet":
        #         imgs = imgs.unsqueeze(1)
        #     lbls = input[1].flatten().to(device)
        #
        #     output = model(imgs)
        #     output = linear_model(output)
        #     # output = linear_model(model(imgs))
        #     loss = F.cross_entropy(output, lbls)
        #     train_dev_loss_sum += float(loss.data)
        #     train_dev_acc += int(sum(torch.argmax(output, dim=1) == lbls))
        #
        #     print("train dev acc epoch {}/{}\titer {}/{} loss:{} {:.3f}s/iter".format(
        #         epoch,
        #         args.epochs,
        #         iter+1,
        #         len(train_dev_acc_dataloader),
        #         loss.data,
        #         time.time() - t),
        #         flush=True)
        #     t = time.time()
        #
        # train_dev_acc /= len(train_dev_acc_dataset)
        # train_dev_loss_sum /= len(train_dev_acc_dataloader)
        # print("train dev epoch {}/{} loss:{} train_dev_acc:{}".format(
        #     epoch, args.epochs, train_dev_loss_sum, train_dev_acc), flush=True)

        #######################################################################
        # Val dev
        # # linear_model.eval()
        # val_dev_acc = 0
        # val_dev_loss_sum = 0.0
        # for iter, input in enumerate(val_dev_dataloader):
        #     imgs = input[0].to(device)
        #     if args.model != "resnet":
        #         imgs = imgs.unsqueeze(1)
        #     lbls = input[1].flatten().to(device)
        #
        #     output = model(imgs)
        #     output = linear_model(output)
        #     loss = F.cross_entropy(output, lbls)
        #     val_dev_loss_sum += float(loss.data)
        #     val_dev_acc += int(sum(torch.argmax(output, dim=1) == lbls))
        #
        #     print("val dev epoch {}/{} iter {}/{} loss:{} {:.3f}s/iter".format(
        #         epoch,
        #         args.epochs,
        #         iter+1,
        #         len(val_dev_dataloader),
        #         loss.data,
        #         time.time() - t),
        #         flush=True)
        #     t = time.time()
        #
        # val_dev_acc /= len(val_dev_dataset)
        # val_dev_loss_sum /= len(val_dev_dataloader)
        # print("val dev epoch {}/{} loss:{} val_dev_acc:{}".format(
        #     epoch, args.epochs, val_dev_loss_sum, val_dev_acc), flush=True)

        #######################################################################
        # Val
        linear_model.eval()
        val_acc = 0
        val_loss_sum = 0.0
        for iter, input in enumerate(val_dataloader):
            imgs = input[0].to(device)
            if args.model != "resnet":
                imgs = imgs.unsqueeze(1)
            lbls = input[1].flatten().to(device)

            output = model(imgs)
            output = linear_model(output)
            loss = F.cross_entropy(output, lbls)
            val_loss_sum += float(loss.data)
            val_acc += int(sum(torch.argmax(output, dim=1) == lbls))

            # log_text = "val epoch {}/{} iter {}/{} loss:{} {:.3f}s/iter"
            if iter % 1500 == 0:
                log_text = "val epoch {}/{} iter {}/{} loss:{}"
                print(log_text.format(
                    epoch,
                    args.epochs,
                    iter+1,
                    len(val_dataloader),
                    loss.data,
                    time.time() - t),
                      flush=False)
                t = time.time()

        val_acc /= len(_val_dataset)
        val_loss_sum /= len(val_dataloader)
        print("val epoch {}/{} loss:{} val_acc:{}".format(
            epoch, args.epochs, val_loss_sum, val_acc))
        with comet_exp.test():
            comet_exp.log_metrics({'acc': val_acc, 'loss': val_loss_sum}, step=(epoch + 1) * len(train_dataloader), epoch=epoch + 1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            save_path = os.path.join(log_dir, "{}.pth".format(epoch))
            torch.save(linear_model.state_dict(), save_path)

        # Check bias and variance
        print("Epoch {} lr {} total: train_loss:{} train_acc:{} val_loss:{} val_acc:{}".format(
            epoch, current_lr, train_loss_sum, train_acc, val_loss_sum, val_acc), flush=True)
        # print("Epoch {} lr {} total: train_acc:{} train_dev_acc:{} val_dev_acc:{} val_acc:{}".format(
        #     epoch, current_lr, train_acc, train_dev_acc, val_dev_acc, val_acc), flush=True)

    print("The best epoch: {} acc: {}".format(best_epoch, best_acc))


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


if __name__ == "__main__":
    if args.mode == "run":
        main(args)
    elif args.mode == "array":
        jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        if not jobid:
            raise
        counter, job = run_cifar_jobs.do(find_counter=jobid, do_job=False)
        opt = vars(args)
        print(counter, job, '\n', opt)
        opt.update(job)
        print(opt)
        args = Struct(**opt)
        main(args)

