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
import numpy as np
from tensorboardX import SummaryWriter
import opts
from dataset import VideoDataSet, ProposalDataSet, GymnasticsSampler
from models import TEM, PEM, partial_load, get_img_loader
from loss_function import TEM_loss_function, PEM_loss_function
import pandas as pd
from pgm import PGM_proposal_generation, PGM_feature_generation
from post_processing import BSN_post_processing
from eval import evaluation_proposal


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def compute_metrics(sums, loss, count):
    values = {k: loss[k].cpu().detach().numpy()
              for k in sums if k != 'entries'}
    values['entries'] = loss['entries']
    new_sums = {k: v + sums[k] for k, v in values.items()}
    avg = {k: v / count for k, v in new_sums.items()}
    return new_sums, avg


def train_TEM(data_loader, model, optimizer, epoch, global_step, comet_exp, opt):
    model.train()
    if opt['do_representation']:
        model.module.set_eval_representation()
        
    count = 1
    keys = ['action_loss', 'start_loss', 'end_loss', 'total_loss', 'action_l1', 'start_l1', 'end_l1', 'action_positive', 'start_positive', 'end_positive', 'entries']
    epoch_sums = {k: 0 for k in keys}
    
    if comet_exp:
        with comet_exp.train():
            comet_exp.log_current_epoch(epoch)

    start = time.time()
    for n_iter, (input_data, label_action, label_start,
                 label_end) in enumerate(data_loader):
        if n_iter == 0:
            print('Training')
            
        if time.time() - opt['start_time'] > opt['time']*3600 - 10 and comet_exp is not None:
            comet_exp.end()
            sys.exit(-1)
        
        TEM_output = model(input_data)
        loss = TEM_loss_function(label_action, label_start, label_end,
                                 TEM_output, opt)
        total_loss = loss["total_loss"]
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        global_step += 1

        if n_iter % opt['tem_compute_loss_interval'] == 0:
            epoch_sums, epoch_avg = compute_metrics(epoch_sums, loss, count)
            count += 1
            if n_iter > 10:
                steps_per_second = (n_iter+1) / (time.time() - start)
                epoch_avg['steps_per_second'] = steps_per_second
            epoch_avg['current_lr'] = get_lr(optimizer)
            print('\nEpoch %d, S/S %.3f, Global Step %d, Local Step %d / %d.' % (epoch, steps_per_second, global_step, n_iter, len(data_loader)))
            s = ", ".join(['%s --> %.4f' % (key, epoch_avg[key]) for key in epoch_avg])
            print("TEM avg so far this epoch: %s." % s)
            if comet_exp:
                with comet_exp.train():
                    comet_exp.log_metrics(epoch_avg, step=global_step, epoch=epoch)

    print('Count: ', count)
    epoch_sums, epoch_avg = compute_metrics(epoch_sums, loss, count)
    steps_per_second = (n_iter+1) / (time.time() - start)
    epoch_avg['steps_per_second'] = steps_per_second
    print('\n***End of Epoch %d***\nS/S %.3f, Global Step %d, Local Step %d / %d.' % (epoch, steps_per_second, global_step, n_iter, len(data_loader)))
    s = ", ".join(['%s --> %.3f' % (key, epoch_avg[key]) for key in epoch_avg])
    print("TEM avg: %s." % s)
    if comet_exp:
        with comet_exp.train():
            comet_exp.log_metrics(epoch_avg, step=global_step, epoch=epoch)
                    
    if comet_exp:
        with comet_exp.train():
            comet_exp.log_epoch_end(epoch)
                                
    return global_step + 1


def test_TEM(data_loader, model, epoch, global_step, comet_exp, opt):
    model.eval()
    
    keys = ['action_loss', 'start_loss', 'end_loss', 'total_loss', 'action_l1', 'start_l1', 'end_l1', 'action_positive', 'start_positive', 'end_positive', 'entries']
    epoch_sums = {k: 0 for k in keys}
    
    for n_iter, (input_data, label_action, label_start,
                 label_end) in enumerate(data_loader):
        if time.time() - opt['start_time'] > opt['time']*3600 - 10 and comet_exp is not None:
            comet_exp.end()
            sys.exit(-1)
            
        TEM_output = model(input_data)
        loss = TEM_loss_function(label_action, label_start, label_end,
                                 TEM_output, opt)
        for k in keys:
            if k == 'entries':
                epoch_sums[k] += loss[k]
            else:
                epoch_sums[k] += loss[k].cpu().detach().numpy()
                
        if n_iter % opt['tem_compute_loss_interval'] == 0:
            print('\nTest - Local Step %d / %d.' % (n_iter, len(data_loader)))

    epoch_values = {k: v / (n_iter + 1) for k, v in epoch_sums.items()}
    if comet_exp:
        with comet_exp.test():
            comet_exp.log_metrics(epoch_values, step=global_step, epoch=epoch)

    s = ", ".join(['%s --> %.3f' % (k, epoch_values[k]) for k in keys])
    print("TEM avg test on epoch %d: %s." % (epoch, s))
    state = {'epoch': epoch, 'global_step': global_step, 'state_dict': model.state_dict()}
    save_dir = os.path.join(opt["checkpoint_path"], '%d.%s' % (opt['counter'], opt['name']))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    save_path = os.path.join(save_dir, 'tem_checkpoint.%d.pth' % epoch)
    torch.save(state, save_path)

    total_loss = epoch_values['total_loss']
    if total_loss < model.module.tem_best_loss:
        model.module.tem_best_loss = total_loss
        save_path = os.path.join(save_dir, 'tem_best.pth')
        torch.save(state, save_path)


def train_PEM(data_loader, model, optimizer, epoch, writer, opt):
    model.train()
    epoch_iou_loss = 0

    for n_iter, (input_data, label_iou) in enumerate(data_loader):
        PEM_output = model(input_data)
        iou_loss = PEM_loss_function(PEM_output, label_iou, model, opt)
        optimizer.zero_grad()
        iou_loss.backward()
        optimizer.step()
        epoch_iou_loss += iou_loss.cpu().detach().numpy()

    writer.add_scalars('data/iou_loss',
                       {'train': epoch_iou_loss / (n_iter + 1)}, epoch)

    print("PEM training loss(epoch %d): iou - %.04f" % (epoch, epoch_iou_loss /
                                                        (n_iter + 1)))


def test_PEM(data_loader, model, epoch, writer, opt):
    model.eval()
    epoch_iou_loss = 0

    for n_iter, (input_data, label_iou) in enumerate(data_loader):
        PEM_output = model(input_data)
        iou_loss = PEM_loss_function(PEM_output, label_iou, model, opt)
        epoch_iou_loss += iou_loss.cpu().detach().numpy()

    writer.add_scalars('data/iou_loss', {'test': epoch_iou_loss / (n_iter + 1)},
                       epoch)

    print("PEM testing  loss(epoch %d): iou - %.04f" % (epoch, epoch_iou_loss /
                                                        (n_iter + 1)))

    state = {'epoch': epoch+1, 'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/pem_checkpoint.pth.tar")
    if epoch_iou_loss < model.module.pem_best_loss:
        model.module.pem_best_loss = np.mean(epoch_iou_loss)
        torch.save(state, opt["checkpoint_path"] + "/pem_best.pth.tar")


def BSN_Train_TEM(opt):
    if opt['do_representation']:
        model = TEM(opt)
        img_loading_func = get_img_loader(opt)
        partial_load(opt['representation_checkpoint'], model)
        for param in model.representation_model.parameters():
            param.requires_grad = False
    else:
        model = TEM(opt)

    model = torch.nn.DataParallel(model).cuda()    
    global_step = 0

    print('    Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = optim.Adam(model.parameters(),
                           lr=opt["tem_training_lr"],
                           weight_decay=opt["tem_weight_decay"])

    train_data_set = VideoDataSet(opt, subset=opt['tem_train_subset'], img_loading_func=img_loading_func, overlap_windows=True)
    train_sampler = GymnasticsSampler(train_data_set.video_dict, train_data_set.frame_list)
    train_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=model.module.batch_size,
        sampler=train_sampler,
        num_workers=opt['data_workers'],
        pin_memory=True,
        drop_last=True)

    test_data_set = VideoDataSet(opt, subset="test", img_loading_func=img_loading_func)
    test_loader = torch.utils.data.DataLoader(
        test_data_set,
        batch_size=model.module.batch_size,
        shuffle=False,
        num_workers=opt['data_workers'],
        pin_memory=True,
        drop_last=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=opt["tem_step_size"],
                                                gamma=opt["tem_step_gamma"])

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

    for epoch in range(opt["tem_epoch"]):
        global_step = train_TEM(train_loader, model, optimizer, epoch, global_step, comet_exp, opt)
        scheduler.step()
        test_TEM(test_loader, model, epoch, global_step, comet_exp, opt)
        

def BSN_Train_PEM(opt):
    model = PEM(opt)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=opt["pem_training_lr"],
                           weight_decay=opt["pem_weight_decay"])

    def collate_fn(batch):
        batch_data = torch.cat([x[0] for x in batch])
        batch_iou = torch.cat([x[1] for x in batch])
        return batch_data, batch_iou

    train_loader = torch.utils.data.DataLoader(
        ProposalDataSet(opt, subset="train"),
        batch_size=model.module.batch_size,
        shuffle=True,
        num_workers=opt['data_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(
        ProposalDataSet(opt, subset="test"),
        batch_size=model.module.batch_size,
        shuffle=True,
        num_workers=opt['data_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=opt["pem_step_size"],
                                                gamma=opt["pem_step_gamma"])

    for epoch in range(opt["pem_epoch"]):
        scheduler.step()
        train_PEM(train_loader, model, optimizer, epoch, writer, opt)
        test_PEM(test_loader, model, epoch, writer, opt)

    writer.close()


def BSN_inference_TEM(opt):
    model = TEM(opt)
    img_loading_func = get_img_loader(opt)    
    checkpoint_path = os.path.join(opt['checkpoint_path'], 'tem_best.pth')
    checkpoint = torch.load(checkpoint_path)
    base_dict = {
        '.'.join(k.split('.')[1:]): v
        for k, v in list(checkpoint['state_dict'].items())
    }
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    output_dir = os.path.join(opt['tem_results_dir'], opt['checkpoint_path'].split('/')[-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset = VideoDataSet(opt, subset=opt['tem_results_subset'], img_loading_func=img_loading_func)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=model.module.batch_size,
        shuffle=False,
        num_workers=opt['data_workers'],
        pin_memory=True,
        drop_last=False)

    columns = ["action", "start", "end", "xmin", "xmax"]
    if opt['do_representation']:
        columns.append('frames')
        
    current_video = None
    current_data = [[] for _ in range(len(columns))]
    for test_idx, (index_list, input_data, anchor_xmin, anchor_xmax) in enumerate(test_loader):
        # The data should be coming back s.t. consecutive data are from the same video.
        # until there is a breakpoint and it starts a new video.
        TEM_output = model(input_data).detach().cpu().numpy()
        batch_action = TEM_output[:, 0, :]
        batch_start = TEM_output[:, 1, :]
        batch_end = TEM_output[:, 2, :]

        index_list = index_list.numpy()
        anchor_xmin = np.array([x.numpy()[0] for x in anchor_xmin])
        anchor_xmax = np.array([x.numpy()[0] for x in anchor_xmax])

        for batch_idx, full_idx in enumerate(index_list):
            if opt['do_representation']:
                video, frame = dataset.frame_list[full_idx]
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

                # NOTE: Here, we are extending but not saying where in the file it is.
                #
                start_frame = frame
                end_frame = start_frame + opt['num_videoframes']*opt['skip_videoframes']
                frames = range(start_frame, end_frame, opt['skip_videoframes'])
                current_data[0].extend(batch_action[batch_idx])
                current_data[1].extend(batch_start[batch_idx])
                current_data[2].extend(batch_end[batch_idx])
                current_data[3].extend(anchor_xmin)
                current_data[4].extend(anchor_xmax)
                current_data[5].extend(list(frames))
            else:
                video = dataset.video_list[full_idx]
                video_action = batch_action[batch_idx]
                video_start = batch_start[batch_idx]
                video_end = batch_end[batch_idx]
                video_result = np.stack((video_action, video_start, video_end,
                                         anchor_xmin, anchor_xmax),
                                        axis=1)
                video_df = pd.DataFrame(video_result, columns=columns)
                path = os.path.join(output_dir, '%s.csv' % video)
                video_df.to_csv(path, index=False)

    if current_data[0]:
        video_result = np.stack(current_data, axis=1)
        video_df = pd.DataFrame(video_result, columns=columns)
        path = os.path.join(output_dir, '%s.csv' % current_video)
        video_df.to_csv(path, index=False)
             

def BSN_inference_PEM(opt):
    model = PEM(opt)
    checkpoint = torch.load(opt["checkpoint_path"] + "/pem_best.pth.tar")
    base_dict = {
        '.'.join(k.split('.')[1:]): v
        for k, v in list(checkpoint['state_dict'].items())
    }
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.eval()

    test_loader = torch.utils.data.DataLoader(
        ProposalDataSet(opt, subset=opt["pem_inference_subset"]),
        batch_size=model.module.batch_size,
        shuffle=False,
        num_workers=opt['data_workers'],
        pin_memory=True,
        drop_last=False)

    for idx, (video_feature, video_xmin, video_xmax, video_xmin_score,
              video_xmax_score) in enumerate(test_loader):
        video_name = test_loader.dataset.video_list[idx]
        video_conf = model(video_feature).view(-1).detach().cpu().numpy()
        video_xmin = video_xmin.view(-1).cpu().numpy()
        video_xmax = video_xmax.view(-1).cpu().numpy()
        video_xmin_score = video_xmin_score.view(-1).cpu().numpy()
        video_xmax_score = video_xmax_score.view(-1).cpu().numpy()

        df = pd.DataFrame()
        df["xmin"] = video_xmin
        df["xmax"] = video_xmax
        df["xmin_score"] = video_xmin_score
        df["xmax_score"] = video_xmax_score
        df["iou_score"] = video_conf

        df.to_csv("./output/PEM_results/" + video_name + ".csv", index=False)


def main(opt):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_gpus = opt['num_gpus']
    opt['base_training_lr'] = opt['tem_training_lr']
    opt['base_batch_size'] = opt['tem_batch_size']
    opt['tem_batch_size'] *= num_gpus
    opt['tem_training_lr'] *= num_gpus
    print(opt)
            
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
            if not os.path.exists("output/PEM_results"):
                os.makedirs("output/PEM_results")
            print("PEM inference start")
            BSN_inference_PEM(opt)
            print("PEM inference finished")
        else:
            print("Wrong mode. PEM has two modes: train and inference")

    elif opt["module"] == "Post_processing":
        print("Post processing start")
        BSN_post_processing(opt)
        print("Post processing finished")

    elif opt["module"] == "Evaluation":
        evaluation_proposal(opt)
    print("")


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    opt['start_time'] = time.time()
    if os.path.exists(opt.get('checkpoint_path')):
        with open(opt["checkpoint_path"] + "/opts.json", "w") as f:
            json.dump(opt, f)
    main(opt)
