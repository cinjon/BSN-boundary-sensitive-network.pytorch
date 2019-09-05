"""
Example run:
python main.py --module PGM --do_representation --num_videoframes 25 --tem_results_dir /checkpoint/cinjon/spaceofmotion/bsn/teminf/101.2019.8.30-00101.1 --pgm_proposals_dir /checkpoint/cinjon/spaceofmotion/bsn/pgmprops --video_anno /private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/gymnastics_annotations/anno_fps12.on.json --video_info /private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/gymnastics_annotations/video_info_new.csv --pgm_score_threshold 0.25
"""

# -*- coding: utf-8 -*-
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from scipy.interpolate import interp1d


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def generate_proposals_repr(opt, video_list, video_dict):
    start_time = time.time()

    tem_results_dir = opt['tem_results_dir']
    proposals_dir = os.path.join(opt['pgm_proposals_dir'], tem_results_dir.split('/')[-1])
        
    for video_name in video_list:
        print('Starting %s' % video_name)
        results_path = os.path.join(tem_results_dir, '%s.csv' % video_name)
        if not os.path.exists(results_path):
            print('Skipping %s because %s is not a path.' % (video_name, results_path))
            continue
        
        tdf = pd.read_csv(results_path)
        start_scores = tdf.start.values[:]
        end_scores = tdf.end.values[:]
        frame_list = tdf.frames.values[:]

        start_bins = np.zeros(len(start_scores))
        start_bins[[0, -1]] = 1
        for idx in range(1, len(start_scores) - 1):
            if start_scores[idx] > start_scores[
                    idx + 1] and start_scores[idx] > start_scores[idx - 1]:
                if start_scores[idx] > 0.9:
                    start_bins[[idx, idx - 1, idx + 1]] = 1
                else:
                    start_bins[idx] = 1
                    
        end_bins = np.zeros(len(end_scores))
        end_bins[[0, -1]] = 1
        for idx in range(1, len(start_scores) - 1):
            if end_scores[idx] > end_scores[
                    idx + 1] and end_scores[idx] > end_scores[idx - 1]:
                if end_scores[idx] > 0.9:
                    end_bins[[idx, idx - 1, idx + 1]] = 1
                else:
                    end_bins[idx] = 1

        xmin_list = []
        xmin_score_list = []
        xmax_list = []
        xmax_score_list = []
        for index in range(len(start_bins)):
            if start_bins[index] == 1:
                xmin_list.append(int(frame_list[index]))
                xmin_score_list.append(start_scores[index])
            if end_bins[index] == 1:
                xmax_list.append(int(frame_list[index]))
                xmax_score_list.append(end_scores[index])

        print('Doing new_props')
        new_props = []
        for ii in range(len(xmax_list)):
            if ii % 5000 == 0:
                print('Done %d / %d'  % (ii, len(xmax_list)))
            tmp_xmax = xmax_list[ii]
            tmp_xmax_score = xmax_score_list[ii]

            for ij in range(len(xmin_list)):
                tmp_xmin = xmin_list[ij]
                tmp_xmin_score = xmin_score_list[ij]
                if tmp_xmax - tmp_xmin < 10:
                    break
                if tmp_xmax - tmp_xmin > 300:
                    continue
                new_props.append(
                    [tmp_xmin, tmp_xmax, tmp_xmin_score, tmp_xmax_score])
        new_props = np.stack(new_props)

        col_name = ["xmin", "xmax", "xmin_score", "xmax_score"]
        new_df = pd.DataFrame(new_props, columns=col_name)
        new_df["score"] = new_df.xmin_score * new_df.xmax_score
        print('Filtering from ', len(new_df))
        new_df = new_df[new_df.score > opt['pgm_score_threshold']]
        print('... To ', len(new_df))

        path = os.path.join(proposals_dir, '%s.proposals.csv' % video_name)
        print('saving preliminary to %s' % path)
        new_df.to_csv(path, index=False)
        
        # new_df = new_df.sort_values(by="score", ascending=False)
        print('Doing gt max')
        video_info = video_dict[video_name]
        video_fps = video_info['fps']

        gt_xmins = []
        gt_xmaxs = []
        annos = video_info["annotations"]
        for idx in range(len(annos)):
            if annos[idx]['label'] == 'off':
                continue
            gt_xmins.append(annos[idx]["segment"][0] * video_fps)
            gt_xmaxs.append(annos[idx]["segment"][1] * video_fps)
            
        print('GT Xmins and Xmaxs')
        print(gt_xmins)
        print(gt_xmaxs)
        new_iou_list = []
        match_xmin_list = []
        match_xmax_list = []
        print('Doing iou and xmin lists.')        
        for j in range(len(new_df)):
            tmp_new_iou = list(
                iou_with_anchors(new_df.xmin.values[j],
                                 new_df.xmax.values[j], gt_xmins, gt_xmaxs))
            new_iou_list.append(max(tmp_new_iou))
            match_xmin_list.append(gt_xmins[tmp_new_iou.index(max(tmp_new_iou))])
            match_xmax_list.append(gt_xmaxs[tmp_new_iou.index(max(tmp_new_iou))])

        new_ioa_list = []
        print('Doing ioa max')        
        for j in range(len(new_df)):
            tmp_new_ioa = max(
                ioa_with_anchors(new_df.xmin.values[j],
                                 new_df.xmax.values[j], gt_xmins, gt_xmaxs))
            new_ioa_list.append(tmp_new_ioa)
            
        new_df["match_iou"] = new_iou_list
        new_df["match_ioa"] = new_ioa_list
        new_df["match_xmin"] = match_xmin_list
        new_df["match_xmax"] = match_xmax_list
        path = os.path.join(proposals_dir, '%s.proposals.csv' % video_name)
        print('saving to %s' % path)
        new_df.to_csv(path, index=False)
        print('Video %s took %.4f time' % (video_name, time.time() - start_time))
    print('Total time was %.4f' % (time.time() - start_time))

def generate_proposals(opt, video_list, video_dict):
    tscale = opt["temporal_scale"]
    tgap = 1. / tscale
    peak_thres = opt["pgm_threshold"]

    tem_results_dir = opt['tem_results_dir']
    proposals_dir = os.path.join(opt['pgm_proposals_dir'], tem_results_dir.split('/')[-1])
    if os.path.exists(proposals_dir):
        shutil.rmtree(proposals_dir)
    if not os.path.exists(proposals_dir):
        os.makedirs(proposals_dir)
        
    for video_name in video_list:
        results_path = os.path.join(tem_results_dir, '%s.csv' % video_name)
        tdf = pd.read_csv(results_path)
        start_scores = tdf.start.values[:]
        end_scores = tdf.end.values[:]

        max_start = max(start_scores)
        max_end = max(end_scores)

        start_bins = np.zeros(len(start_scores))
        start_bins[[0, -1]] = 1
        for idx in range(1, tscale - 1):
            if start_scores[idx] > start_scores[
                    idx + 1] and start_scores[idx] > start_scores[idx - 1]:
                start_bins[idx] = 1
            elif start_scores[idx] > (peak_thres * max_start):
                start_bins[idx] = 1

        end_bins = np.zeros(len(end_scores))
        end_bins[[0, -1]] = 1
        for idx in range(1, tscale - 1):
            if end_scores[idx] > end_scores[
                    idx + 1] and end_scores[idx] > end_scores[idx - 1]:
                end_bins[idx] = 1
            elif end_scores[idx] > (peak_thres * max_end):
                end_bins[idx] = 1

        xmin_list = []
        xmin_score_list = []
        xmax_list = []
        xmax_score_list = []
        for j in range(tscale):
            if start_bins[j] == 1:
                xmin_list.append(tgap / 2 + tgap * j)
                xmin_score_list.append(start_scores[j])
            if end_bins[j] == 1:
                xmax_list.append(tgap / 2 + tgap * j)
                xmax_score_list.append(end_scores[j])

        new_props = []
        for ii in range(len(xmax_list)):
            tmp_xmax = xmax_list[ii]
            tmp_xmax_score = xmax_score_list[ii]

            for ij in range(len(xmin_list)):
                tmp_xmin = xmin_list[ij]
                tmp_xmin_score = xmin_score_list[ij]
                if tmp_xmin >= tmp_xmax:
                    break
                new_props.append(
                    [tmp_xmin, tmp_xmax, tmp_xmin_score, tmp_xmax_score])
        new_props = np.stack(new_props)

        col_name = ["xmin", "xmax", "xmin_score", "xmax_score"]
        new_df = pd.DataFrame(new_props, columns=col_name)
        new_df["score"] = new_df.xmin_score * new_df.xmax_score

        new_df = new_df.sort_values(by="score", ascending=False)

        video_info = video_dict[video_name]
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second

        try:
            gt_xmins = []
            gt_xmaxs = []
            for idx in range(len(video_info["annotations"])):
                gt_xmins.append(video_info["annotations"][idx]["segment"][0] /
                                corrected_second)
                gt_xmaxs.append(video_info["annotations"][idx]["segment"][1] /
                                corrected_second)
            new_iou_list = []
            for j in range(len(new_df)):
                tmp_new_iou = max(
                    iou_with_anchors(new_df.xmin.values[j],
                                     new_df.xmax.values[j], gt_xmins, gt_xmaxs))
                new_iou_list.append(tmp_new_iou)

            new_ioa_list = []
            for j in range(len(new_df)):
                tmp_new_ioa = max(
                    ioa_with_anchors(new_df.xmin.values[j],
                                     new_df.xmax.values[j], gt_xmins, gt_xmaxs))
                new_ioa_list.append(tmp_new_ioa)
            new_df["match_iou"] = new_iou_list
            new_df["match_ioa"] = new_ioa_list
        except:
            pass

        path = os.path.join(proposals_dir, '%s.proposals.csv' % video_name)
        print('saving to %s' % path)
        new_df.to_csv(path, index=False)


def getDatasetDict(opt):
    df = pd.read_csv(opt["video_info"])
    json_data = load_json(opt["video_anno"])
    database = json_data
    video_dict = {}
    keys = ['duration_frame', 'duration_second', 'feature_frame', 'annotations', 'fps']
    for i in range(len(df)):
        video_name = df.video.values[i]
        video_info = database[video_name]
        video_new_info = {k: video_info[k] for k in keys}
        video_new_info['subset'] = df.subset.values[i]
        video_dict[video_name] = video_new_info
    return video_dict


def bookend_zeros(arr, num):
    return np.concatenate([np.zeros([num]), arr, np.zeros([num])])

    
def generate_features_repr(opt, video_list, video_dict):
    num_sample_start = opt["num_sample_start"]
    num_sample_end = opt["num_sample_end"]
    num_sample_action = opt["num_sample_action"]
    num_sample_interpld = opt["num_sample_interpld"]

    tem_results_dir = opt['tem_results_dir']
    model = tem_results_dir.split('/')[-1]
    proposals_dir = os.path.join(opt['pgm_proposals_dir'], model)
    features_dir = os.path.join(opt['pgm_features_dir'], model)

    start_time = time.time()
    for video_name in video_list:
        s0 = time.time()
        tem_path = os.path.join(tem_results_dir, video_name + ".csv")
        if not os.path.exists(tem_path):
            print("NOT generating features for %s because features don't exist." % video_name)
            continue        
        adf = pd.read_csv(tem_path)
        
        proposals_path = os.path.join(proposals_dir, '%s.proposals.csv' % video_name)
        if not os.path.exists(proposals_path):
            print("NOT generating features for %s because proposals don't exist." % video_name)
            continue        
        pdf = pd.read_csv(proposals_path)

        print('Doing %s with paths %s and %s' % (video_name, tem_path, proposals_path))
        
        score_action = bookend_zeros(adf.action.values[:], 20)
        score_end = bookend_zeros(adf.end.values[:], 20)
        score_start = bookend_zeros(adf.start.values[:], 20)
        snippets = [5*i - 87 for i in range(20)] + list(adf.frames.values[:]) + [5*i + 5 + adf.frames.values[:][-1] for i in range(20)]
        print('Computing the interp1ds')
        f_action = interp1d(snippets, score_action, axis=0)
        f_start = interp1d(snippets, score_start, axis=0)
        f_end = interp1d(snippets, score_end, axis=0)
        print('Done ciomputing interp1ds')
        
        feature_bsp = []
        s1 = time.time()
        for idx in range(len(pdf)):
            if idx % 1000 == 0 and idx > 0:
                print('At %d of %d. S/S: %.4f' % (idx, len(pdf), idx / (time.time() - s1)))
                
            xmin = pdf.xmin.values[idx]
            xmax = pdf.xmax.values[idx]
            xlen = xmax - xmin
            xmin_0 = xmin - xlen * opt["bsp_boundary_ratio"]
            xmin_1 = xmin + xlen * opt["bsp_boundary_ratio"]
            xmax_0 = xmax - xlen * opt["bsp_boundary_ratio"]
            xmax_1 = xmax + xlen * opt["bsp_boundary_ratio"]
            
            #start
            plen_start = (xmin_1 - xmin_0) / (num_sample_start - 1)
            tmp_x_new = [xmin_0 + plen_start * ii for ii in range(num_sample_start)]
            tmp_y_new_start = np.concatenate((f_action(tmp_x_new),
                                              f_start(tmp_x_new)))
            
            #end
            plen_end = (xmax_1 - xmax_0) / (num_sample_end - 1)
            tmp_x_new = [xmin_0 + plen_end * ii for ii in range(num_sample_end)]
            tmp_y_new_end = np.concatenate((f_action(tmp_x_new),
                                            f_end(tmp_x_new)))
            
            #action
            plen_action = (xmax - xmin) / (num_sample_action - 1)
            tmp_x_new = [xmin_0 + plen_action * ii for ii in range(num_sample_action)]
            tmp_y_new_action = f_action(tmp_x_new)
            tmp_y_new_action = np.reshape(tmp_y_new_action, [-1])

            # make the feature bsp
            feature_bsp.append(np.concatenate(
                [tmp_y_new_action, tmp_y_new_start, tmp_y_new_end]))
            
        feature_bsp = np.array(feature_bsp)
        path = os.path.join(
            features_dir, "%s.features.npy" % video_name)
        print("Size of feature_bsp: ", feature_bsp.shape, len(adf), len(pdf), video_name)
        np.save(path, feature_bsp)
        print('Time from start to finish for video with adf len %d and pdf len %d was %.4f.', (len(adf), len(pdf), time.time() - s0))
    print('Total time was ', time.time() - start_time)


def generate_features(opt, video_list, video_dict):
    num_sample_start = opt["num_sample_start"]
    num_sample_end = opt["num_sample_end"]
    num_sample_action = opt["num_sample_action"]
    num_sample_interpld = opt["num_sample_interpld"]

    tem_results_dir = opt['tem_results_dir']    
    for video_name in video_list:
        tem_path = os.path.join(tem_results_dir, video_name + ".csv")
        adf = pd.read_csv(tem_path)
        score_action = adf.action.values[:]
        seg_xmins = adf.xmin.values[:]
        seg_xmaxs = adf.xmax.values[:]
        video_scale = len(adf)
        video_gap = seg_xmaxs[0] - seg_xmins[0]
        video_extend = video_scale / 4 + 10
        pdf = pd.read_csv("./output/PGM_proposals/" + video_name + ".csv")
        video_subset = video_dict[video_name]['subset']
        if video_subset == "training":
            pdf = pdf[:opt["pem_top_K"]]
        else:
            pdf = pdf[:opt["pem_top_K_inference"]]
        tmp_zeros = np.zeros([video_extend])
        score_action = np.concatenate((tmp_zeros, score_action, tmp_zeros))
        tmp_cell = video_gap
        tmp_x = [-tmp_cell/2-(video_extend-1-ii)*tmp_cell for ii in range(video_extend)] + \
                 [tmp_cell/2+ii*tmp_cell for ii in range(video_scale)] + \
                  [tmp_cell/2+seg_xmaxs[-1] +ii*tmp_cell for ii in range(video_extend)]
        f_action = interp1d(tmp_x, score_action, axis=0)
        feature_bsp = []

        for idx in range(len(pdf)):
            xmin = pdf.xmin.values[idx]
            xmax = pdf.xmax.values[idx]
            xlen = xmax - xmin
            xmin_0 = xmin - xlen * opt["bsp_boundary_ratio"]
            xmin_1 = xmin + xlen * opt["bsp_boundary_ratio"]
            xmax_0 = xmax - xlen * opt["bsp_boundary_ratio"]
            xmax_1 = xmax + xlen * opt["bsp_boundary_ratio"]
            #start
            plen_start = (xmin_1 - xmin_0) / (num_sample_start - 1)
            plen_sample = plen_start / num_sample_interpld
            tmp_x_new = [
                xmin_0 - plen_start / 2 + plen_sample * ii
                for ii in range(num_sample_start * num_sample_interpld + 1)
            ]
            tmp_y_new_start_action = f_action(tmp_x_new)
            tmp_y_new_start = [
                np.mean(
                    tmp_y_new_start_action[ii * num_sample_interpld:(ii + 1) *
                                           num_sample_interpld + 1])
                for ii in range(num_sample_start)
            ]
            #end
            plen_end = (xmax_1 - xmax_0) / (num_sample_end - 1)
            plen_sample = plen_end / num_sample_interpld
            tmp_x_new = [
                xmax_0 - plen_end / 2 + plen_sample * ii
                for ii in range(num_sample_end * num_sample_interpld + 1)
            ]
            tmp_y_new_end_action = f_action(tmp_x_new)
            tmp_y_new_end = [
                np.mean(
                    tmp_y_new_end_action[ii * num_sample_interpld:(ii + 1) *
                                         num_sample_interpld + 1])
                for ii in range(num_sample_end)
            ]
            #action
            plen_action = (xmax - xmin) / (num_sample_action - 1)
            plen_sample = plen_action / num_sample_interpld
            tmp_x_new = [
                xmin - plen_action / 2 + plen_sample * ii
                for ii in range(num_sample_action * num_sample_interpld + 1)
            ]
            tmp_y_new_action = f_action(tmp_x_new)
            tmp_y_new_action = [
                np.mean(tmp_y_new_action[ii * num_sample_interpld:(ii + 1) *
                                            num_sample_interpld + 1])
                for ii in range(num_sample_action)
            ]
            tmp_feature = np.concatenate(
                [tmp_y_new_action, tmp_y_new_start, tmp_y_new_end])
            feature_bsp.append(tmp_feature)
        feature_bsp = np.array(feature_bsp)
        np.save("./output/PGM_feature/" + video_name, feature_bsp)


def PGM_proposal_generation(opt):
    pgm_dir = opt["pgm_proposals_dir"]
    if not os.path.exists(pgm_dir):
        os.makedirs(pgm_dir, exist_ok=True)

    tem_results_dir = opt['tem_results_dir']
    pgm_dir = os.path.join(pgm_dir, tem_results_dir.split('/')[-1])
    if os.path.exists(pgm_dir):
        shutil.rmtree(pgm_dir)
    if not os.path.exists(pgm_dir):
        os.makedirs(pgm_dir)
        
    video_dict = load_json(opt["video_anno"])
    video_list = sorted(video_dict.keys())  #[:199]
    # NOTE: change this back.
    # video_list = [k for k in video_list if '12.4.18-Part-1' in k]
    # video_list = sorted(video_list, key=lambda k: ('12.4.18' in k, k))
    num_videos = len(video_list)
    num_threads = min(num_videos, opt['pgm_thread'])
    num_videos_per_thread = int(num_videos / num_threads)
    processes = []
    func = generate_proposals_repr if opt['do_representation'] else generate_proposals
    
    for tid in range(num_threads - 1):
        tmp_video_list = video_list[tid * num_videos_per_thread:(tid + 1) * num_videos_per_thread]
        p = mp.Process(target=func,
                       args=(
                           opt,
                           tmp_video_list,
                           video_dict,
                       ))
        p.start()
        processes.append(p)

    tmp_video_list = video_list[(num_threads - 1) *
                                num_videos_per_thread:]
    print(tmp_video_list)
    p = mp.Process(target=func,
                   args=(
                       opt,
                       tmp_video_list,
                       video_dict,
                   ))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()


def PGM_feature_generation(opt):
    model = opt['tem_results_dir'].split('/')[-1]
    features_dir = os.path.join(opt['pgm_features_dir'], model)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    
    video_dict = getDatasetDict(opt)
    video_list = sorted(video_dict.keys())
    func = generate_features_repr if opt['do_representation'] else generate_features
    num_videos = len(video_list)
    num_threads = min(num_videos, opt['pgm_thread'])
    num_videos_per_thread = int(num_videos / opt["pgm_thread"])
    processes = []
    print('\n***\n')
    print(opt['pgm_thread'], num_videos_per_thread, video_list)
    for tid in range(opt["pgm_thread"] - 1):
        tmp_video_list = video_list[tid * num_videos_per_thread:(tid + 1) *
                                    num_videos_per_thread]
        p = mp.Process(target=func,
                       args=(
                           opt,
                           tmp_video_list,
                           video_dict,
                       ))
        p.start()
        processes.append(p)

    tmp_video_list = video_list[(opt["pgm_thread"] - 1) *
                                num_videos_per_thread:]
    p = mp.Process(target=func,
                   args=(
                       opt,
                       tmp_video_list,
                       video_dict,
                   ))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
