from collections import defaultdict
from comet_ml import Experiment as CometExperiment, OfflineExperiment
import sys
sys.dont_write_bytecode = True
import os
import time
import json
import torch
import torchvision
import torch.nn.parallel
import torch.optim as optim
from torchsummary import summary
import numpy as np
import opts
from dataset import ThumosFeatures, ThumosImages, ProposalDataSet, GymnasticsSampler, GymnasticsImages, GymnasticsFeatures, ProposalSampler, VideoDataset
from models import TEM, PEM, partial_load, get_img_loader, AMDIMModel, get_video_transforms
from loss_function import TEM_loss_function, PEM_loss_function
import pandas as pd
from pgm import PGM_proposal_generation, PGM_feature_generation
from post_processing import BSN_post_processing
from post_processing2 import BSN_post_processing as BSN_post_processing2
from eval import evaluation_proposal
from eval2 import evaluation_proposal as evaluation_proposal2
import tem_jobs


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def compute_metrics(sums, loss, count):
    values = {k: loss[k].cpu().detach().numpy()
              for k in sums if k not in ['entries']}
    for key in ['entries']:
        if key in loss:
            values[key] = loss[key]
    new_sums = {k: v + sums[k] for k, v in values.items()}
    avg = {k: v / count for k, v in new_sums.items()}
    return new_sums, avg


def train_TEM(data_loader, model, optimizer, epoch, global_step, comet_exp, opt):
    model.train()
    if opt['do_representation'] and not opt['no_freeze']:
        model.module.set_eval_representation()
        
    count = 1
    keys = ['action_loss', 'start_loss', 'end_loss', 'total_loss', 'cost', 'current_l2', 'action_positive', 'start_positive', 'end_positive', 'entries']
    epoch_sums = {k: 0 for k in keys}
    
    if comet_exp:
        with comet_exp.train():
            comet_exp.log_current_epoch(epoch)

    start = time.time()
    for n_iter, (input_data, label_action, label_start,
                 label_end) in enumerate(data_loader):
        if time.time() - opt['start_time'] > opt['time']*3600 - 10 and comet_exp is not None:
            comet_exp.end()
            sys.exit(-1)

        # for thumosimages, input_data shape: [bs, 100, 3, 176, 320]
        # print('Just before tem input: ', type(input_data))
        # print(input_data[0])
        TEM_output = model(input_data)
        loss = TEM_loss_function(label_action, label_start, label_end,
                                 TEM_output, opt)
        l2 = sum([(W**2).sum() for W in model.module.parameters()])
        l2 = l2.sum() / 2
        l2 = opt['tem_l2_loss'] * l2
        loss['current_l2'] = l2
        total_loss = loss['cost'] + l2
        loss["total_loss"] = total_loss
        optimizer.zero_grad()
        if opt['do_gradient_checkpointing']:
            model.zero_grad()
        total_loss.backward()
        # print(model.module.representation_model.backbone.inception_5b_3x3.weight[0][0])        
        optimizer.step()
        global_step += 1

        if n_iter % opt['tem_compute_loss_interval'] == 0:
            epoch_sums, epoch_avg = compute_metrics(epoch_sums, loss, count)
                        
            count += 1
            steps_per_second = 0
            if n_iter > 10:
                steps_per_second = (n_iter+1) / (time.time() - start)
                epoch_avg['steps_per_second'] = steps_per_second
            epoch_avg['current_lr'] = get_lr(optimizer)
            # print({k: type(v) for k, v in epoch_avg.items()})
            print('\nEpoch %d, S/S %.3f, Global Step %d, Local Step %d / %d.' % (epoch, steps_per_second, global_step, n_iter, len(data_loader)), flush=True)
            s = ", ".join(['%s --> %.6f' % (key, epoch_avg[key]) for key in epoch_avg])
            print("TEM avg: %s." % s, flush=True)
            if comet_exp:
                with comet_exp.train():
                    comet_exp.log_metrics(epoch_avg, step=global_step, epoch=epoch)

    epoch_sums, epoch_avg = compute_metrics(epoch_sums, loss, count)
    # steps_per_second = (n_iter+1) / (time.time() - start)
    # epoch_avg['steps_per_second'] = steps_per_second
    print('\n***End of Epoch %d***\nLearningRate: %.4f' % (epoch, get_lr(optimizer)), flush=True)
    s = ", ".join(['%s --> %.6f' % (key.replace('_loss', '').replace('current_', '').capitalize(), epoch_avg[key]) for key in sorted(epoch_avg.keys())])
    print("Train: %s." % s, flush=True)
    if comet_exp:
        with comet_exp.train():
            comet_exp.log_metrics(epoch_avg, step=global_step, epoch=epoch)                    
            comet_exp.log_epoch_end(epoch)
                                
    return global_step + 1


def test_TEM(data_loader, model, optimizer, epoch, global_step, comet_exp, opt):
    model.eval()
    
    keys = ['action_loss', 'start_loss', 'end_loss', 'total_loss', 'cost', 'action_positive', 'start_positive', 'end_positive', 'entries', 'current_l2']
    epoch_sums = {k: 0 for k in keys}
    
    for n_iter, (input_data, label_action, label_start,
                 label_end) in enumerate(data_loader):
        if time.time() - opt['start_time'] > opt['time']*3600 - 10 and comet_exp is not None:
            comet_exp.end()
            sys.exit(-1)

        TEM_output = model(input_data)
        loss = TEM_loss_function(label_action, label_start, label_end,
                                 TEM_output, opt)
        l2 = sum([(W**2).sum() for W in model.module.parameters()])
        l2 = l2.sum() / 2
        l2 = opt['tem_l2_loss'] * l2
        loss['current_l2'] = l2
        total_loss = loss['cost'] + l2
        loss["total_loss"] = total_loss
        for k in keys:
            if k == 'entries':
                epoch_sums[k] += loss[k]
            else:
                epoch_sums[k] += loss[k].cpu().detach().numpy()
                
        # if n_iter % opt['tem_compute_loss_interval'] == 0:
        #     print('\nTest - Local Step %d / %d.' % (n_iter, len(data_loader)))

    epoch_values = {k: v / (n_iter + 1) for k, v in epoch_sums.items()}
    if comet_exp:
        with comet_exp.test():
            comet_exp.log_metrics(epoch_values, step=global_step, epoch=epoch)

    s = ", ".join(['%s --> %.6f' % (key.replace('_loss', '').replace('current_', '').capitalize(), epoch_values[key]) for key in sorted(epoch_values.keys())])
    print("Test %s." % s, flush=True)
    
    state = {
        'epoch': epoch, 'global_step': global_step, 'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict()
    }
    save_dir = os.path.join(opt["checkpoint_path"], opt['name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    total_loss = epoch_values['total_loss']
    if total_loss < model.module.tem_best_loss:
        model.module.tem_best_loss = total_loss
        save_path = os.path.join(save_dir, 'tem_checkpoint.%d.pth' % epoch)
        torch.save(state, save_path)
        # save_path = os.path.join(save_dir, 'tem_best.%d.pth % epoch')
        # torch.save(state, save_path)
        


def train_PEM(data_loader, model, optimizer, epoch, global_step, comet_exp, opt):
    model.train()

    count = 1
    keys = ['iou_loss', 'current_l2', 'total_loss']
    epoch_sums = {k: 0 for k in keys}

    start = time.time()
    for n_iter, (input_data, label_iou) in enumerate(data_loader):
        if time.time() - opt['start_time'] > opt['time']*3600 - 10 and comet_exp is not None:
            comet_exp.end()
            sys.exit(-1)

        PEM_output = model(input_data)
        loss = PEM_loss_function(PEM_output, label_iou, opt)
        l2 = sum([(W**2).sum() for W in model.module.parameters()])
        l2 = l2.sum() / 2
        l2 = opt['pem_l2_loss'] * l2
        loss['current_l2'] = l2
        loss['iou_loss'] *= 10
        iou_loss = loss['iou_loss']
        total_loss = iou_loss + l2
        loss['total_loss'] = total_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        global_step += 1
        
        if n_iter % opt['pem_compute_loss_interval'] == 0:
            epoch_sums, epoch_avg = compute_metrics(epoch_sums, loss, count)
            count += 1
            steps_per_second = 0
            if n_iter > 10:
                steps_per_second = (n_iter+1) / (time.time() - start)
                epoch_avg['steps_per_second'] = steps_per_second
            epoch_avg['current_lr'] = get_lr(optimizer)
            # print('\nEpoch %d, S/S %.3f, Global Step %d, Local Step %d / %d.' % (epoch, steps_per_second, global_step, n_iter, len(data_loader)))
            # s = ", ".join(['%s --> %.6f' % (key, epoch_avg[key]) for key in epoch_avg])
            # print("PEM avg so far this epoch: %s." % s)
            if comet_exp:
                with comet_exp.train():
                    comet_exp.log_metrics(epoch_avg, step=global_step, epoch=epoch)

    epoch_sums, epoch_avg = compute_metrics(epoch_sums, loss, count)
    steps_per_second = (n_iter+1) / (time.time() - start)
    # epoch_avg['steps_per_second'] = steps_per_second
    print('\n***End of Epoch %d***\nLearningRate: %.4f' % (epoch, get_lr(optimizer)))
    s = ", ".join(['%s --> %.6f' % (key.replace('current_', '').replace('_loss', '').capitalize(), epoch_avg[key]) for key in sorted(epoch_avg.keys())])
    print("Train: %s." % s)
    if comet_exp:
        with comet_exp.train():
            comet_exp.log_metrics(epoch_avg, step=global_step, epoch=epoch)
            comet_exp.log_epoch_end(epoch)
                                
    return global_step + 1


def test_PEM(data_loader, model, epoch, global_step, comet_exp, opt):
    model.eval()
    keys = ['iou_loss', 'current_l2', 'total_loss']
    epoch_sums = {k: 0 for k in keys}
    
    for n_iter, (input_data, label_iou) in enumerate(data_loader):
        if time.time() - opt['start_time'] > opt['time']*3600 - 10 and comet_exp is not None:
            comet_exp.end()
            sys.exit(-1)
            
        PEM_output = model(input_data)
        loss = PEM_loss_function(PEM_output, label_iou, opt)
        l2 = sum([(W**2).sum() for W in model.module.parameters()])
        l2 = l2.sum() / 2
        l2 = opt['pem_l2_loss'] * l2
        loss['current_l2'] = l2
        loss['iou_loss'] *= 10
        iou_loss = loss['iou_loss']
        total_loss = iou_loss + l2
        loss['total_loss'] = total_loss        
        for k in keys:
            epoch_sums[k] += loss[k].cpu().detach().numpy()
            
    epoch_values = {k : v / (n_iter + 1) for k, v in epoch_sums.items()}
    if comet_exp:
        with comet_exp.test():
            comet_exp.log_metrics(epoch_values, step=global_step, epoch=epoch)

    s = ", ".join(['%s --> %.06f' % (k.replace('current_', '').replace('_loss', '').capitalize(), epoch_values[k]) for k in sorted(keys)])
    print("Test: %s." % s)
    state = {'epoch': epoch, 'global_step': global_step, 'state_dict': model.state_dict()}

    save_dir = os.path.join(opt["checkpoint_path"], opt['name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    save_path = os.path.join(save_dir, 'pem_checkpoint.%d.pth' % epoch)
    torch.save(state, save_path)
    
    iou_loss = epoch_values['iou_loss']
    if iou_loss < model.module.pem_best_loss:
        model.module.pem_best_loss = iou_loss
        save_path = os.path.join(save_dir, 'pem_best.pth')
        torch.save(state, save_path)


def _maybe_load_checkpoint(model, optimizer, global_step, epoch, directory):
    if not os.path.exists(directory):
        print('NOT LAODING CHECKPOINT (nonexistent dir)')
        return global_step, epoch
    
    ckpts = os.listdir(directory)
    if not ckpts:
        print('NOT LAODING CHECKPOINT (empty dir)')
        return global_step, epoch

    print('HAVE ckpts: ', ckpts)
    best_ckpt = sorted(ckpts, key=lambda k: int(k.split('.')[-2]), reverse=True)[0]
    print('LOADING from ckpt!', directory, best_ckpt)
    checkpoint = torch.load(os.path.join(directory, best_ckpt))
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    print('Epoch/Gss: ', epoch, global_step)
    return epoch, global_step

    
def BSN_Train_TEM(opt):
    global_step = 0
    epoch = 0
    if opt['do_representation']:
        model = TEM(opt)
        optimizer = optim.Adam(model.parameters(),
                               lr=opt["tem_training_lr"],
                               weight_decay=opt["tem_weight_decay"])
        global_step, epoch = _maybe_load_checkpoint(
            model, optimizer, global_step, epoch,
            os.path.join(opt["checkpoint_path"], opt['name']))
        if opt['representation_checkpoint']:
            # print(model.representation_model.backbone.inception_5b_3x3.weight[0][0])
            if opt['do_random_model']:
                print('DOING RANDOM MDOEL!!!')
            else:
                print('DOING Pretrianed modelll!!!')                
                partial_load(opt['representation_checkpoint'], model)
            # print(model.representation_model.backbone.inception_5b_3x3.weight[0][0])
        if not opt['no_freeze']:
            for param in model.representation_model.parameters():
                param.requires_grad = False
        print(len([p for p in model.representation_model.parameters()]))
    else:
        model = TEM(opt)
        optimizer = optim.Adam(model.parameters(),
                               lr=opt["tem_training_lr"],
                               weight_decay=opt["tem_weight_decay"])
        global_step, epoch = _maybe_load_checkpoint(
            model, optimizer, global_step, epoch,
            os.path.join(opt["checkpoint_path"], opt['name']))
        
    model = torch.nn.DataParallel(model).cuda()
    # summary(model, (2, 3, 224, 224))
    
    print('    Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters()) / 1000000.0))

    if opt['dataset'] == 'gymnastics':
        # default image_dir is '/checkpoint/cinjon/spaceofmotion/sep052019/rawframes.426x240.12'
        img_loading_func = get_img_loader(opt)
        train_data_set = GymnasticsImages(
            opt, subset='Train', img_loading_func=img_loading_func,
            image_dir=opt['gym_image_dir'],
            video_info_path = os.path.join(opt['video_info'], 'Train_Annotation.csv')
        )
        train_sampler = GymnasticsSampler(train_data_set, opt['sampler_mode'])
        test_data_set = GymnasticsImages(
            opt, subset="Val", img_loading_func=img_loading_func,
            image_dir=opt['gym_image_dir'],
            video_info_path = os.path.join(opt['video_info'], 'Val_Annotation.csv')
        )
    elif opt['dataset'] == 'gymnasticsfeatures':
        # feature_dirs should roughly look like:
        # /checkpoint/cinjon/spaceofmotion/sep052019/tsn.1024.426x240.12.no-oversample/csv/rgb,/checkpoint/cinjon/spaceofmotion/sep052019/tsn.1024.426x240.12.no-oversample/csv/flow
        feature_dirs = opt['feature_dirs'].split(',')
        train_data_set = GymnasticsFeatures(opt, subset='Train', feature_dirs=feature_dirs, video_info_path = os.path.join(opt['video_info'], 'Train_Annotation.csv'))
        test_data_set = GymnasticsFeatures(opt, subset='Val', feature_dirs=feature_dirs, video_info_path = os.path.join(opt['video_info'], 'Val_Annotation.csv'))
        train_sampler = None
    elif opt['dataset'] == 'thumosfeatures':
        feature_dirs = opt['feature_dirs'].split(',')
        train_data_set = ThumosFeatures(opt, subset='Val', feature_dirs=feature_dirs)
        test_data_set = ThumosFeatures(opt, subset="Test", feature_dirs=feature_dirs)
        train_sampler = None
    elif opt['dataset'] == 'thumosimages':
        img_loading_func = get_img_loader(opt)
        train_data_set = ThumosImages(
            opt, subset='Val', img_loading_func=img_loading_func,
            image_dir='/checkpoint/cinjon/thumos/rawframes.TH14_validation_tal.30',
            video_info_path = os.path.join(opt['video_info'], 'Val_Annotation.csv')
        )
        test_data_set = ThumosImages(
            opt, subset='Test', img_loading_func=img_loading_func,
            image_dir='/checkpoint/cinjon/thumos/rawframes.TH14_test_tal.30',
            video_info_path = os.path.join(opt['video_info'], 'Test_Annotation.csv')            
        )
        train_sampler = None
    elif opt['dataset'] == 'activitynet':
        train_sampler = None
        representation_module = opt['representation_module']
        train_transforms = get_video_transforms(
            representation_module, opt['do_augment'])
        test_transforms = get_video_transforms(
            representation_module, False)
        train_data_set = VideoDataset(opt, train_transforms, subset='train', fraction=0.3)
        # We use val because we don't have annotations for test.
        test_data_set = VideoDataset(opt, test_transforms, subset='val', fraction=0.3)

    print('train_loader / val_loader sizes: ', len(train_data_set), len(test_data_set))
    train_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=model.module.batch_size,
        shuffle=False if train_sampler else True,
        sampler=train_sampler,
        num_workers=opt['data_workers'],
        pin_memory=True,
        drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_data_set,
        batch_size=model.module.batch_size,
        shuffle=False,
        num_workers=opt['data_workers'],
        pin_memory=True,
        drop_last=False)
    # test_loader = None

    milestones = [int(k) for k in opt['tem_lr_milestones'].split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=opt['tem_step_gamma'])

    if opt['log_to_comet']:
        comet_exp = CometExperiment(api_key="hIXq6lDzWzz24zgKv7RYz6blo",
                                    project_name="bsn",
                                    workspace="cinjon",
                                    auto_metric_logging=True,
                                    auto_output_logging=None,
                                    auto_param_logging=False)
    elif opt['local_comet_dir']:
        comet_exp = OfflineExperiment(
            api_key="hIXq6lDzWzz24zgKv7RYz6blo",
            project_name="bsn",
            workspace="cinjon",
            auto_metric_logging=True,
            auto_output_logging=None,
            auto_param_logging=False,
            offline_directory=opt['local_comet_dir'])
    else:
        comet_exp = None

    if comet_exp:
        comet_exp.log_parameters(opt)
        comet_exp.set_name(opt['name'])

    # test_TEM(test_loader, model, optimizer, 0, 0, comet_exp, opt)
    for epoch in range(epoch+1, opt["tem_epoch"] + 1):
        global_step = train_TEM(train_loader, model, optimizer, epoch, global_step, comet_exp, opt)
        test_TEM(test_loader, model, optimizer, epoch, global_step, comet_exp, opt)
        if opt['dataset'] == 'activitynet':
            test_loader.dataset._subset_dataset(.3)
            train_loader.dataset._subset_dataset(.3)
        scheduler.step()
        

def BSN_Train_PEM(opt):
    model = PEM(opt)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=opt["pem_training_lr"],
                           weight_decay=opt["pem_weight_decay"])

    print('Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    def collate_fn(batch):
        batch_data = torch.cat([x[0] for x in batch])
        batch_iou = torch.cat([x[1] for x in batch])
        return batch_data, batch_iou

    train_dataset = ProposalDataSet(opt, subset="train")
    train_sampler = ProposalSampler(train_dataset.proposals, train_dataset.indices, max_zero_weight=opt['pem_max_zero_weight'])
    
    global_step = 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=model.module.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=opt['data_workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn if not opt['pem_do_index'] else None)

    subset = "validation" if opt['dataset'] == 'activitynet' else "test"
    test_loader = torch.utils.data.DataLoader(
        ProposalDataSet(opt, subset=subset),
        batch_size=model.module.batch_size,
        shuffle=True,
        num_workers=opt['data_workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn if not opt['pem_do_index'] else None)

    milestones = [int(k) for k in opt['pem_lr_milestones'].split(',')]    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=opt['pem_step_gamma'])

    if opt['log_to_comet']:
        comet_exp = CometExperiment(api_key="hIXq6lDzWzz24zgKv7RYz6blo",
                                    project_name="bsnpem",
                                    workspace="cinjon",
                                    auto_metric_logging=True,
                                    auto_output_logging=None,
                                    auto_param_logging=False)
    elif opt['local_comet_dir']:
        comet_exp = OfflineExperiment(
            api_key="hIXq6lDzWzz24zgKv7RYz6blo",
            project_name="bsnpem",
            workspace="cinjon",
            auto_metric_logging=True,
            auto_output_logging=None,
            auto_param_logging=False,
            offline_directory=opt['local_comet_dir'])
    else:
        comet_exp = None

    if comet_exp:
        comet_exp.log_parameters(opt)
        comet_exp.set_name(opt['name'])    

    test_PEM(test_loader, model, -1, -1, comet_exp, opt)
    for epoch in range(opt["pem_epoch"]):
        global_step = train_PEM(train_loader, model, optimizer, epoch, global_step, comet_exp, opt)
        test_PEM(test_loader, model, epoch, global_step, comet_exp, opt)
        scheduler.step()


def BSN_inference_TEM(opt):
    output_dir = os.path.join(opt['tem_results_dir'], opt['checkpoint_path'].split('/')[-1])
    print(sorted(opt.items()), flush=True)
        
    model = TEM(opt)
    checkpoint_epoch = opt['checkpoint_epoch']
    if checkpoint_epoch is not None:
        checkpoint_path = os.path.join(opt['checkpoint_path'], 'tem_checkpoint.%d.pth' % checkpoint_epoch)
        output_dir = os.path.join(output_dir, 'ckpt.%d' % checkpoint_epoch)
    else:
        checkpoint_path = os.path.join(opt['checkpoint_path'], 'tem_best.pth')
        output_dir = os.path.join(output_dir, 'ckpt.best')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print('Checkpoint path is ', checkpoint_path, flush=True)
    checkpoint = torch.load(checkpoint_path)
    base_dict = {
        '.'.join(k.split('.')[1:]): v
        for k, v in list(checkpoint['state_dict'].items())
    }
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    if opt['dataset'] == 'gymnastics':
        img_loading_func = get_img_loader(opt)
        dataset = GymnasticsImages(
            opt,
            subset=opt['tem_results_subset'].title(),
            img_loading_func=img_loading_func,
            image_dir=opt['gym_image_dir'],
            video_info_path = os.path.join(opt['video_info'], 'Full_Annotation.csv')            
        )
    elif opt['dataset'] == 'gymnasticsfeatures':
        # feature_dirs should roughly look like:
        # /checkpoint/cinjon/spaceofmotion/sep052019/tsn.1024.426x240.12.no-oversample/csv/rgb,/checkpoint/cinjon/spaceofmotion/sep052019/tsn.1024.426x240.12.no-oversample/csv/flow
        feature_dirs = opt['feature_dirs'].split(',')
        dataset = GymnasticsFeatures(
            opt, subset=opt['tem_results_subset'].title(),
            feature_dirs=feature_dirs,
            video_info_path = os.path.join(opt['video_info'], 'Full_Annotation.csv'))
    elif opt['dataset'] == 'thumosfeatures':
        feature_dirs = opt['feature_dirs'].split(',')
        dataset = ThumosFeatures(opt, subset=opt['tem_results_subset'].title(), feature_dirs=feature_dirs,
            video_info_path=os.path.join(opt['video_info'], 'Full_Annotation.csv')
        )
    elif opt['dataset'] == 'thumosimages':
        img_loading_func = get_img_loader(opt)
        dataset = ThumosImages(
            opt,
            subset=opt['tem_results_subset'].title(),
            img_loading_func=img_loading_func,
            image_dir='/checkpoint/cinjon/thumos/rawframes.TH14_%s_tal.30' % opt['tem_results_subset'],
            video_info_path=os.path.join(opt['video_info'], 'Full_Annotation.csv')
        )
    elif opt['dataset'] == 'activitynet':
        representation_module = opt['representation_module']
        test_transforms  = get_video_transforms(
            representation_module, False)
        dataset = VideoDataset(opt, test_transforms, subset='full', fraction=1.0)
                
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=model.module.batch_size,
        shuffle=False,
        num_workers=opt['data_workers'],
        pin_memory=True,
        drop_last=False)

    columns = ["action", "start", "end", "frames"]

    all_vids = defaultdict(int)
    current_video = None
    current_start = defaultdict(float)
    current_end = defaultdict(float)
    current_action = defaultdict(float)    
    calc_time_list = defaultdict(int)
    num_videoframes = opt['num_videoframes']
    skip_videoframes = opt['skip_videoframes']
    print('About to start enumerating', flush=True)
    for test_idx, (index_list, input_data, video_name, snippets) in enumerate(test_loader):
        if test_idx == 0:
            print('Started enumerating!', flush=True)        
        # The data should be coming back s.t. consecutive data are from the same video.
        # until there is a breakpoint and it starts a new video.

        TEM_output = model(input_data).detach().cpu().numpy()
        batch_action = TEM_output[:, 0, :]
        batch_start = TEM_output[:, 1, :]
        batch_end = TEM_output[:, 2, :]

        index_list = index_list.numpy()
        for batch_idx, full_idx in enumerate(index_list):
            item_video = video_name[batch_idx]
            all_vids[item_video] += 1
            item_snippets = snippets[batch_idx]
            if not current_video:
                print('First video: ', item_video, flush=True)
                current_video = item_video
                current_start = defaultdict(float)
                current_end = defaultdict(float)
                current_action = defaultdict(float)    
                calc_time_list = defaultdict(int)
            elif item_video != current_video:
                print('Next video: ', item_video, full_idx, flush=True)
                column_frames = sorted(calc_time_list.keys())
                column_action = [current_action[k] * 1. / calc_time_list[k] for k in column_frames]
                column_start = [current_start[k] * 1. / calc_time_list[k] for k in column_frames]
                column_end = [current_end[k] * 1. / calc_time_list[k] for k in column_frames]
                video_result = np.stack([column_action, column_start, column_end], axis=1)
                column_frames = np.reshape(column_frames, [-1, 1])
                
                video_result = np.concatenate([video_result, column_frames], axis=1)
                video_df = pd.DataFrame(video_result, columns=columns)
                path = os.path.join(output_dir, '%s.csv' % current_video)
                video_df.to_csv(path, index=False)
                current_video = item_video
                current_start = defaultdict(float)
                current_end = defaultdict(float)
                current_action = defaultdict(float)    
                calc_time_list = defaultdict(int)
                
            for snippet_, action_, start_, end_ in zip(
                    item_snippets, batch_action[batch_idx], batch_start[batch_idx], batch_end[batch_idx]):
                frame = snippet_.item()
                calc_time_list[frame] += 1
                current_action[frame] += action_
                current_start[frame] += start_
                current_end[frame] += end_
                    
    if len(calc_time_list):
        column_frames = sorted(calc_time_list.keys())
        column_action = [current_action[k] * 1. / calc_time_list[k] for k in column_frames]
        column_start = [current_start[k] * 1. / calc_time_list[k] for k in column_frames]
        column_end = [current_end[k] * 1. / calc_time_list[k] for k in column_frames]
        video_result = np.stack([column_action, column_start, column_end], axis=1)
        print(video_result.shape, flush=True)
        
        video_result = np.concatenate([video_result, np.reshape(column_frames, [-1, 1])], axis=1)
        video_df = pd.DataFrame(video_result, columns=columns)
        path = os.path.join(output_dir, '%s.csv' % current_video)
        video_df.to_csv(path, index=False)
    print(len(all_vids))
             

def BSN_inference_PEM(opt):
    output_dir = os.path.join(opt['pem_inference_results_dir'], opt['checkpoint_path'].split('/')[-1])
    checkpoint_epoch = opt['checkpoint_epoch']
    if checkpoint_epoch is not None:
        checkpoint_path = os.path.join(opt['checkpoint_path'], 'pem_checkpoint.%d.pth' % checkpoint_epoch)
        output_dir = os.path.join(output_dir, 'ckpt.%d' % checkpoint_epoch)
    else:
        checkpoint_path = os.path.join(opt['checkpoint_path'], 'pem_best.pth')
        output_dir = os.path.join(output_dir, 'ckpt.best')

    print('Checkpoint path is ', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    base_dict = {
        '.'.join(k.split('.')[1:]): v
        for k, v in list(checkpoint['state_dict'].items())
    }

    model = PEM(opt)
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    test_loader = torch.utils.data.DataLoader(
        ProposalDataSet(opt, subset=opt["pem_inference_subset"]),
        batch_size=model.module.batch_size,
        shuffle=False,
        num_workers=opt['data_workers'],
        pin_memory=True,
        drop_last=False)

    current_video = None
    columns = ["xmin", "xmax", "xmin_score", "xmax_score", "iou_score"]
    for idx, (index_list, video_feature, video_xmin, video_xmax, video_xmin_score,
              video_xmax_score) in enumerate(test_loader):
        video_conf = model(video_feature).view(-1).detach().cpu().numpy()
        video_xmin = video_xmin.view(-1).cpu().numpy()
        video_xmax = video_xmax.view(-1).cpu().numpy()
        video_xmin_score = video_xmin_score.view(-1).cpu().numpy()
        video_xmax_score = video_xmax_score.view(-1).cpu().numpy()
        
        index_list = index_list.numpy()
        for batch_idx, full_idx in enumerate(index_list):
            video, frame = test_loader.dataset.indices[full_idx]
            if not current_video:
                print('First video: ', video, full_idx)
                current_video = video
                current_data = [[] for _ in range(len(columns))]
            elif video != current_video:
                print('Changing from video %s to video %s: %d' % (current_video, video, full_idx))
                video_result = np.stack(current_data, axis=1)
                video_df = pd.DataFrame(video_result, columns=columns)
                path = os.path.join(output_dir, '%s.csv' % current_video)
                video_df.to_csv(path, index=False)
                current_video = video
                current_data = [[] for _ in range(len(columns))]

            current_data[0].append(video_xmin[batch_idx])
            current_data[1].append(video_xmax[batch_idx])
            current_data[2].append(video_xmin_score[batch_idx])
            current_data[3].append(video_xmax_score[batch_idx])
            current_data[4].append(video_conf[batch_idx])

    if current_data[0]:
        video_result = np.stack(current_data, axis=1)
        video_df = pd.DataFrame(video_result, columns=columns)
        path = os.path.join(output_dir, '%s.csv' % current_video)
        video_df.to_csv(path, index=False)

        
def main(opt):        
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_gpus = opt['num_gpus']
    if opt['module'] == 'TEM':
        opt['base_training_lr'] = opt['tem_training_lr']
        opt['base_batch_size'] = opt['tem_batch_size']
        opt['tem_batch_size'] *= num_gpus
        opt['tem_training_lr'] *= num_gpus
    elif opt['module'] == 'PEM':
        opt['base_training_lr'] = opt['pem_training_lr']
        opt['base_batch_size'] = opt['pem_batch_size']
        opt['pem_batch_size'] *= num_gpus
        opt['pem_training_lr'] *= num_gpus
    print(opt, flush=True)

    if opt["module"] == "TEM":
        if opt["mode"] == "train":
            print("TEM training start")
            BSN_Train_TEM(opt)
            print("TEM training finished")
        elif opt["mode"] == "inference":
            print("TEM inference start")
            BSN_inference_TEM(opt)
            print("TEM inference finished")
        else:
            print("Wrong mode. TEM has two modes: train and inference")
    elif opt["module"] == "PGM":
        print("PGM: start generate proposals")
        PGM_proposal_generation(opt)
        print("PGM: finish generate proposals")

        print("PGM: start generate BSP feature")
        PGM_feature_generation(opt)
        print("PGM: finish generate BSP feature")
    elif opt["module"] == "PEM":
        if opt["mode"] == "train":
            print("PEM training start")
            BSN_Train_PEM(opt)
            print("PEM training finished")
        elif opt["mode"] == "inference":
            print("PEM inference start")
            BSN_inference_PEM(opt)
            print("PEM inference finished")
        else:
            print("Wrong mode. PEM has two modes: train and inference")

    elif opt["module"] == "Post_processing":
        print("Post processing start")
        BSN_post_processing2(opt)
        print("Post processing finished")
        if opt['do_eval_after_postprocessing']:
            evaluation_proposal2(opt)
            
    elif opt["module"] == "Evaluation":
        evaluation_proposal2(opt)
    print("")


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    opt['start_time'] = time.time()
    
    # When we use jobarray, we need to get the opt.
    mode = opt['mode']
    if 'jobarray_train' in opt['mode']:
        jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        if not jobid:
            raise
        counter, job = tem_jobs.run(find_counter=jobid)
        print(counter, job, '\n', opt)
        opt.update(job)
        print(sorted(opt.items()), flush=True)
        print('\n***\n%s\n***\n' % opt['do_feat_conversion'])
        if 'debug' in mode:
            opt.update({'num_gpus': 2, 'data_workers': 12,
                        'name': 'dbg', 'counter': 0,
                        'tem_batch_size': 1, 'do_feat_conversion': True,
                        # 'gym_image_dir': '/checkpoint/cinjon/spaceofmotion/sep052019/rawframes.426x240.12',
                        'local_comet_dir': None,
                        'dataset': 'thumosimages',
                        'video_info': '/private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/thumos14_annotations',
                        'ccc_img_size': 128,
                        # 'do_random_model': True
            })

    if 'debugrun' not in mode:
        main(opt)
