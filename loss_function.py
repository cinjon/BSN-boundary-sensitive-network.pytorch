# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F


def bi_loss(scores, anchors, opt):
    scores = scores.view(-1).cuda()
    anchors = anchors.contiguous().view(-1)
    l1 = torch.mean(torch.abs(scores - anchors))
    print(scores[:10], anchors[:10])

    pmask = (scores > opt["tem_match_thres"]).float().cuda()
    num_positive = torch.sum(pmask)
    num_entries = len(scores)
    # I made it do the +1 below and the ratio + 1e-6
    ratio = (num_entries + 1) / (num_positive + 1)
    ratio += 1e-6

    coef_0 = 0.5 * (ratio) / (ratio - 1)
    coef_1 = coef_0 * (ratio - 1)
    loss = coef_1 * pmask * torch.log(anchors + 0.00001) + coef_0 * (
        1.0 - pmask) * torch.log(1.0 - anchors + 0.00001)
    loss = -torch.mean(loss)
    num_sample = [num_positive, ratio, num_entries]
    return loss, num_sample, l1


def TEM_loss_calc(anchors_action, anchors_start, anchors_end,
                  match_scores_action, match_scores_start, match_scores_end,
                  opt):

    action_loss, num_sample_action, action_l1 = bi_loss(match_scores_action,
                                                        anchors_action, opt)
    start_loss, num_sample_start, start_l1 = bi_loss(match_scores_start,
                                                     anchors_start, opt)
    end_loss, num_sample_end, end_l1 = bi_loss(match_scores_end,
                                               anchors_end, opt)

    loss_dict = {
        "action_loss": action_loss,
        "action_positive": num_sample_action[0],
        "action_l1": action_l1,
        "start_loss": start_loss,
        "start_positive": num_sample_start[0],
        "start_l1": start_l1,
        "end_loss": end_loss,
        "end_positive": num_sample_end[0],
        "end_l1": end_l1,
        "entries": num_sample_action[2]
    }
    return loss_dict


def TEM_loss_function(y_action, y_start, y_end, TEM_output, opt):
    anchors_action = TEM_output[:, 0, :]
    anchors_start = TEM_output[:, 1, :]
    anchors_end = TEM_output[:, 2, :]
    loss_dict = TEM_loss_calc(anchors_action, anchors_start, anchors_end,
                              y_action, y_start, y_end, opt)

    total_loss = 2 * loss_dict["action_loss"] + loss_dict["start_loss"] + loss_dict[
        "end_loss"]
    loss_dict["total_loss"] = total_loss
    return loss_dict


def PEM_loss_function(anchors_iou, match_iou, model, opt):
    match_iou = match_iou.cuda()
    anchors_iou = anchors_iou.view(-1)
    u_hmask = (match_iou > opt["pem_high_iou_thres"]).float()
    u_mmask = ((match_iou <= opt["pem_high_iou_thres"]) &
               (match_iou > opt["pem_low_iou_thres"])).float()
    u_lmask = (match_iou < opt["pem_low_iou_thres"]).float()

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = model.module.u_ratio_m * num_h / (num_m)
    r_m = torch.min(r_m, torch.Tensor([1.0]).cuda())[0]
    u_smmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).cuda()
    u_smmask = u_smmask * u_mmask
    u_smmask = (u_smmask > (1. - r_m)).float()

    r_l = model.module.u_ratio_l * num_h / (num_l)
    r_l = torch.min(r_l, torch.Tensor([1.0]).cuda())[0]
    u_slmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).cuda()
    u_slmask = u_slmask * u_lmask
    u_slmask = (u_slmask > (1. - r_l)).float()

    iou_weights = u_hmask + u_smmask + u_slmask
    iou_loss = F.smooth_l1_loss(anchors_iou, match_iou)
    iou_loss = torch.sum(iou_loss * iou_weights) / torch.sum(iou_weights)

    return iou_loss
