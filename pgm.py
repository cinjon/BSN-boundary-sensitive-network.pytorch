"""
Example run:
python main.py --module PGM --do_representation --num_videoframes 25 --tem_results_dir /checkpoint/cinjon/spaceofmotion/bsn/teminf/101.2019.8.30-00101.1 --pgm_proposals_dir /checkpoint/cinjon/spaceofmotion/bsn/pgmprops --video_anno /private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/gymnastics_annotations/anno_fps12.on.json --video_info /private/home/cinjon/Code/BSN-boundary-sensitive-network.pytorch/data/gymnastics_annotations/video_info_new.csv --pgm_score_threshold 0.25
"""

# -*- coding: utf-8 -*-
import json
import os
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


def generate_proposals(opt, video_list, video_data):
    start_time = time.time()

    # NOTE: Hackity hack hack.
    video_dict = None
    anno_df = None
    if type(video_data) == dict:
        video_dict = video_data
    else:
        anno_df = video_data

    tem_results_dir = opt['tem_results_dir']
    proposals_dir = opt['pgm_proposals_dir']
    skipped_paths = []
        
    for video_name in video_list:
        print('Starting %s' % video_name)
        results_path = os.path.join(tem_results_dir, '%s.csv' % video_name)
        if not os.path.exists(results_path):
            print('Skipping %s because %s is not a path.' % (video_name, results_path))
            skipped_paths.append(results_path)
            continue

        anno_df_ = anno_df[anno_df.video == video_name]
        
        tdf = pd.read_csv(results_path)
        start_scores = tdf.start.values[:]
        end_scores = tdf.end.values[:]
        try:
            frame_list = tdf.frames.values[:]
        except Exception as e:
            frame_list = tdf.frame.values[:]

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
        for index in range(len(start_scores)):
            if start_bins[index] == 1:
                xmin_list.append(int(frame_list[index]))
                xmin_score_list.append(start_scores[index])
            if end_bins[index] == 1:
                xmax_list.append(int(frame_list[index]))
                xmax_score_list.append(end_scores[index])

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
        # new_df["score"] = new_df.xmin_score * new_df.xmax_score
        # print('Filtering from ', len(new_df))
        # new_df = new_df[new_df.score > opt['pgm_score_threshold']]
        # print('... To ', len(new_df))

        # path = os.path.join(proposals_dir, '%s.proposals.csv' % video_name)
        # print('saving preliminary to %s' % path)
        # new_df.to_csv(path, index=False)
        
        if video_dict is not None:
            video_info = video_dict[video_name]
            video_fps = video_info['fps']
            annos = video_info["annotations"]
            gt_xmins = []
            gt_xmaxs = []
            for idx in range(len(annos)):
                if annos[idx]['label'] == 'off':
                    continue

                gt_xmins.append(annos[idx]["segment"][0] * video_fps)
                gt_xmaxs.append(annos[idx]["segment"][1] * video_fps)
        elif anno_df_ is not None:
            gt_xmins = anno_df_.startFrame.values[:]
            gt_xmaxs = anno_df_.endFrame.values[:]

        # Ok, so all of these gt_xmins and gt_xmaxs are the same ...
        # ... As are the xmin and xmax values in the DFs.
        
        new_iou_list = []
        match_xmin_list = []
        match_xmax_list = []
        for j in range(len(new_df)):
            tmp_new_iou = list(
                iou_with_anchors(
                    new_df.xmin.values[j],
                    new_df.xmax.values[j],
                    gt_xmins,
                    gt_xmaxs)
            )
            new_iou_list.append(max(tmp_new_iou))
            match_xmin_list.append(gt_xmins[tmp_new_iou.index(max(tmp_new_iou))])
            match_xmax_list.append(gt_xmaxs[tmp_new_iou.index(max(tmp_new_iou))])

        new_ioa_list = []
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
    print('Total time was %.4f' % (time.time() - start_time), skipped_paths)


def getDatasetDict(opt):
    print(opt['video_info'])
    print(opt['video_anno'])
    video_info = opt['video_info']
    if 'thumos' in opt['dataset']:
        video_info = os.path.join(video_info, 'Full_Annotation.csv')
    print(video_info)
    df = pd.read_csv(video_info)
    
    database = load_json(opt["video_anno"])
    video_dict = {}
    keys = ['duration_frame', 'duration_second', 'feature_frame', 'annotations', 'fps']
    for i in range(len(df)):
        video_name = df.video.values[i]
        video_info = database[video_name]
        video_new_info = {k: video_info[k] for k in keys}
        if 'thumos' in opt['dataset']:
            video_new_info['subset'] = video_name.split('_')[1]
        else:
            video_new_info['subset'] = df.subset.values[i]            
        video_dict[video_name] = video_new_info
    return video_dict


def bookend_zeros(arr, num):
    return np.concatenate([np.zeros([num]), arr, np.zeros([num])])

    
def generate_features(opt, video_list, video_dict):
    num_sample_start = opt["num_sample_start"]
    num_sample_end = opt["num_sample_end"]
    num_sample_action = opt["num_sample_action"]
    num_videoframes = opt["num_videoframes"]
    skip_videoframes = opt["skip_videoframes"]
    bookend_num = int(num_videoframes / skip_videoframes)
    normalizer = skip_videoframes*(bookend_num - 1) - (skip_videoframes + int((skip_videoframes + 1)/2))

    tem_results_dir = opt['tem_results_dir']
    proposals_dir = opt['pgm_proposals_dir']
    features_dir = opt['pgm_features_dir']

    start_time = time.time()
    for video_name in video_list:
        s0 = time.time()
        tem_path = os.path.join(tem_results_dir, video_name + ".csv")
        print(tem_path)
        if not os.path.exists(tem_path):
            print("NOT generating features for %s because features don't exist." % video_name)
            continue        
        adf = pd.read_csv(tem_path)
        try:
            adf_frames = adf.frames.values[:]
        except Exception as e:
            adf_frames = adf.frame.values[:]
        
        proposals_path = os.path.join(proposals_dir, '%s.proposals.csv' % video_name)
        if not os.path.exists(proposals_path):
            print("NOT generating features for %s because proposals don't exist." % video_name)
            continue        
        pdf = pd.read_csv(proposals_path)

        print('Doing %s with paths %s and %s' % (video_name, tem_path, proposals_path))
        
        score_action = bookend_zeros(adf.action.values[:], bookend_num)
        score_end = bookend_zeros(adf.end.values[:], bookend_num)
        score_start = bookend_zeros(adf.start.values[:], bookend_num)
        # 
        snippets = [skip_videoframes*i - normalizer for i in range(bookend_num)] + list(adf_frames) + [skip_videoframes*i + skip_videoframes + adf_frames[-1] for i in range(bookend_num)]
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
            
            # I originall had the following (see xmin_0)
            # tmp_x_new = [xmin_0 + plen_action * ii for ii in range(num_sample_action)]
            # But they have this:
            tmp_x_new = [xmin + plen_action * ii for ii in range(num_sample_action)]            
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
        print('Time from start to finish for video with adf len %d and pdf len %d was %.4f.' % (len(adf), len(pdf), time.time() - s0))
    print('Total time was ', time.time() - start_time)


def PGM_proposal_generation(opt):
    if 'thumos' in opt['dataset']:
        video_data = pd.read_csv(os.path.join(opt["video_info"], 'Full_Annotation.csv'))
        video_list = sorted(list(set(video_data.video.values[:])))
        # video_list = [k for k in video_list if 'video_validation_0000053' in k]
        # print(video_list)
    else:
        video_data = load_json(opt["video_anno"])
        video_list = sorted(video_data.keys())  #[:199]
    
    # NOTE: change this back.
    # video_list = [k for k in video_list if '12.18.18' in k or '12.5.18' in k]
    
    num_videos = len(video_list)
    num_threads = min(num_videos, opt['pgm_thread'])
    num_videos_per_thread = int(num_videos / num_threads)
    processes = []
    func = generate_proposals
    
    for tid in range(num_threads - 1):
        tmp_video_list = video_list[tid * num_videos_per_thread:(tid + 1) * num_videos_per_thread]
        p = mp.Process(target=func,
                       args=(
                           opt,
                           tmp_video_list,
                           video_data,
                       ))
        p.start()
        processes.append(p)

    tmp_video_list = video_list[(num_threads - 1) *
                                num_videos_per_thread:]
    p = mp.Process(target=func,
                   args=(
                       opt,
                       tmp_video_list,
                       video_data,
                   ))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()


def PGM_feature_generation(opt):
    video_dict = getDatasetDict(opt)
    video_list = sorted(video_dict.keys())
    # NOTE: change this back.
    # video_list = [k for k in video_list if '12.18.18' in k or '12.5.18' in k]
    
    func = generate_features
    num_videos = len(video_list)
    num_threads = min(num_videos, opt['pgm_thread'])
    num_videos_per_thread = int(num_videos / opt["pgm_thread"])
    processes = []
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
