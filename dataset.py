# -*- coding: utf-8 -*-
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import pickle

import numpy as np
import pandas as pd
import torch.utils.data as data
import torch


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores
    

class Thumos(data.Dataset):
    def __init__(self, opt, subset=None, feature_dirs=[], fps=30, image_dir=None, img_loading_func=None):
        self.subset = subset
        self.mode = opt["mode"]
        self.boundary_ratio = opt['boundary_ratio']
        self.skip_videoframes = opt['skip_videoframes']
        self.num_videoframes = opt['num_videoframes']
        self.img_loading_func = img_loading_func

        # A list of paths to directories containing csvs of the
        # features per video. We will concatenate tehse togehter.
        self.feature_dirs = feature_dirs

        # A path to directories containing npys of the images in each video.
        # We assume that these are just rgb for now.
        self.image_dir = image_dir
        self.fps = fps
        
        # e.g. /data/thumos14_annotations/Test_Annotation.csv
        self.video_info_path = os.path.join(opt["video_info"], '%s_Annotation.csv' % self.subset)
        self._get_data()

    def _get_data(self):
        anno_df = pd.read_csv(self.video_info_path)
        video_name_list = list(set(anno_df.video.values[:]))
        if self.feature_dirs:
            list_data = []
            
        list_anchor_xmins = []
        list_anchor_xmaxs = []
        list_gt_bbox = []
        list_videos = []
        list_indices = []
        
        num_videoframes = self.num_videoframes
        skip_videoframes = self.skip_videoframes
        start_snippet = int((skip_videoframes + 1) / 2)
        stride = int(num_videoframes / 2)
        
        for video_name in video_name_list:
            anno_df_video = anno_df[anno_df.video == video_name]
            if self.mode == 'train':
                gt_xmins = anno_df_video.startFrame.values[:]
                gt_xmaxs = anno_df_video.endFrame.values[:]

            # NOTE: num_snippet is the number of snippets in this video.
            if self.image_dir:
                image_dir = os.path.join(self.image_dir, video_name)
                num_snippet = len(os.listdir(image_dir))
                num_snippet = int((num_snippet - start_snippet) / skip_videoframes)
            elif self.feature_dirs:
                feature_dfs = [
                    pd.read_csv(os.path.join(feature_dir, '%s.csv' % video_name))
                    for feature_dir in self.feature_dirs
                ]
                num_snippet = min([len(df) for df in feature_dfs])
                df_data = np.concatenate([df.values[:num_snippet, :]
                                          for df in feature_dfs],
                                         axis=1)

            df_snippet = [start_snippet + skip_videoframes*i for i in range(num_snippet)]
            num_windows = int((num_snippet + stride - num_videoframes) / stride)
            windows_start = [i* stride for i in range(num_windows)]
            if num_snippet < num_videoframes:
                windows_start = [0]
                if self.feature_dirs:
                    # Add on a bunch of zero data if there aren't enough windows.
                    tmp_data = np.zeros((num_videoframes - num_snippet, 400))
                    df_data = np.concatenate((df_data, tmp_data), axis=0)
                df_snippet.extend([
                    df_snippet[-1] + skip_videoframes*(i+1)
                    for i in range(num_videoframes - num_snippet)
                ])
            elif num_snippet - windows_start[-1] - num_videoframes > int(num_videoframes / skip_videoframes):
                windows_start.append(num_snippet - num_videoframes)

            for start in windows_start:
                if self.feature_dirs:
                    tmp_data = df_data[start:start + num_videoframes, :]
                    
                tmp_snippets = np.array(df_snippet[start:start + num_videoframes])
                if self.mode == 'train':
                    tmp_anchor_xmins = tmp_snippets - skip_videoframes/2.
                    tmp_anchor_xmaxs = tmp_snippets + skip_videoframes/2.
                    tmp_gt_bbox = []
                    tmp_ioa_list = []
                    for idx in range(len(gt_xmins)):
                        tmp_ioa = ioa_with_anchors(gt_xmins[idx], gt_xmaxs[idx],
                                                   tmp_anchor_xmins[0],
                                                   tmp_anchor_xmaxs[-1])
                        tmp_ioa_list.append(tmp_ioa)
                        if tmp_ioa > 0:
                            tmp_gt_bbox.append([gt_xmins[idx], gt_xmaxs[idx]])
                        
                    if len(tmp_gt_bbox) > 0 and max(tmp_ioa_list) > 0.9:
                        list_gt_bbox.append(tmp_gt_bbox)
                        list_anchor_xmins.append(tmp_anchor_xmins)
                        list_anchor_xmaxs.append(tmp_anchor_xmaxs)
                        list_videos.append(video_name)
                        list_indices.append(tmp_snippets)
                        if self.feature_dirs:
                            list_data.append(np.array(tmp_data).astype(np.float32))
                elif self.mode == 'inference':
                    list_videos.append(video_name)
                    list_indices.append(tmp_snippets)
                    if self.feature_dirs:
                        list_data.append(np.array(tmp_data).astype(np.float32))
                                
        print("List of videos: ", len(set(list_videos)), flush=True)
        self.data = {
            'video_names': list_videos,
            'indices': list_indices
        }
        if self.mode == 'train':
            self.data.update({
                'gt_bbox': list_gt_bbox,
                'anchor_xmins': list_anchor_xmins,
                'anchor_xmaxs': list_anchor_xmaxs,
            }
        if self.feature_dirs:
            self.data['video_data'] = list_data
        print('Size of data: ', len(self.data['video_names']), flush=True)                             

    def __getitem__(self, index):
        video_data = self._get_video_data(self.data, index)
        if self.mode == "train":
            anchor_xmin = self.data['anchor_xmins'][index]
            anchor_xmax = self.data['anchor_xmaxs'][index]
            gt_bbox = self.data['gt_bbox'][index]        
            match_score_action, match_score_start, match_score_end = self._get_train_label(gt_bbox, anchor_xmin, anchor_xmax)
            return video_data, match_score_action, match_score_start, match_score_end
        else:
            video_name = self.data['video_names'][index]
            snippets = self.data['indices'][index]
            return index, video_data, video_name, snippets

    def _get_train_label(self, gt_bbox, anchor_xmin, anchor_xmax):
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        # same as gt_len but using the thumos code repo :/.
        gt_duration = gt_xmaxs - gt_xmins
        gt_duration_boundary = np.maximum(
            self.skip_videoframes, gt_duration * self.boundary_ratio)
        gt_start_bboxs = np.stack(
            (gt_xmins - gt_duration_boundary / 2, gt_xmins + gt_duration_boundary / 2),
            axis=1
        )
        gt_end_bboxs = np.stack(
            (gt_xmaxs - gt_duration_boundary / 2, gt_xmaxs + gt_duration_boundary / 2),
            axis=1
        )

        match_score_action = [
            np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                       gt_xmins, gt_xmaxs))
            for jdx in range(len(anchor_xmin))
        ]

        match_score_start = [
            np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                       gt_start_bboxs[:, 0], gt_start_bboxs[:, 1]))
            for jdx in range(len(anchor_xmin))
        ]

        match_score_end = [
            np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                       gt_end_bboxs[:, 0], gt_end_bboxs[:, 1]))
            for jdx in range(len(anchor_xmin))
        ]
        
        return torch.Tensor(match_score_action), torch.Tensor(match_score_start), torch.Tensor(match_score_end)
        
    def __len__(self):
        return len(self.data['video_names'])

    
class ThumosFeatures(Thumos):

    def __init__(self, opt, subset=None, feature_dirs=[]):
        super(ThumosFeatures, self).__init__(opt, subset, feature_dirs, fps=None, image_dir=None, img_loading_func=None)

    def _get_video_data(self, data, index):
        return data['video_data'][index]
    

class ThumosImages(Thumos):

    def __init__(self, opt, subset=None, fps=30, image_dir=None, img_loading_func=None):
        self.do_augment = opt['do_augment'] and subset == 'train'
        super(ThumosImages, self).__init__(opt, subset, feature_dirs=None, fps=fps, image_dir=image_dir, img_loading_func=img_loading_func)        

    def _get_video_data(self, data, index):
        indices = data['indices'][index]
        name = data['video_names'][index]
        path = os.path.join(self.image_dir, name)
        path = Path(path)
        paths = [path / ('%010.4f.npy' % (i / self.fps)) for i in indices]
        imgs = [self.img_loading_func(p.absolute(), do_augment=self.do_augment)
                for p in paths if p.exists()]
            
        if type(imgs[0]) == np.array:
            video_data = np.array(imgs)
            video_data = torch.Tensor(video_data)
        elif type(imgs[0]) == torch.Tensor:
            video_data = torch.stack(imgs)

        if len(video_data) < self.num_videoframes:
            shape = [self.num_videoframes - len(video_data)]
            shape += list(video_data.shape[1:])
            zeros = torch.zeros(*shape)
            video_data = torch.cat([video_data, zeros], axis=0)
        return video_data
    

class GymnasticsSampler(data.WeightedRandomSampler):
    def __init__(self, video_dict, frame_list):
        """
        Args:
          video_dict: A dict of key to video_info.
          frame_list: A list of (key, index into that key's video). This is what the Dataset is using
            and what we need to give sample weights for.
        """
        per_video_frame_count = {k: int(v['duration_frame']) for k, v in video_dict.items()}
        total_frame_count = sum(list(per_video_frame_count.values()))
        # Initial weight count is inversely proportional to the number of frames in that video.
        # The fewer the number of frames, the higher chance there is of selecting from that video.
        weights = [total_frame_count * 1. / per_video_frame_count[k] for k, _ in frame_list]

        on_indices = {k: set() for k in video_dict.keys()}
        for k, video_info in video_dict.items():
            fps = video_info['fps']
            for anno in video_info['annotations']:
                if anno['label'] == 'on':
                    end_frame = int(anno['segment'][1] * fps)
                    start_frame = int(anno['segment'][0] * fps) + 1
                    on_indices[k].update(range(start_frame, end_frame))

        total_on_count = sum([len(v) for k, v in on_indices.items()])
        total_on_ratio = 1. * total_on_count / total_frame_count
        print('total on ratio: ', total_on_ratio, total_on_count, total_frame_count)
        for num, (k, frame) in enumerate(frame_list):
            if frame in on_indices[k]:
                weights[num] *= 0.5 / total_on_ratio
            else:
                weights[num] *= 0.5 / (1 - total_on_ratio)

        super(GymnasticsSampler, self).__init__(weights, len(weights), replacement=True)

        
class GymnasticsDataSet(data.Dataset):

    def __init__(self, opt, subset="train", img_loading_func=None, overlap_windows=False):
        self.subset = subset
        self.mode = opt["mode"]
        self.boundary_ratio = opt["boundary_ratio"]
        self.num_videoframes = opt['num_videoframes']
        self.skip_videoframes = opt['skip_videoframes']
        self.img_loading_func = img_loading_func

        self.overlap_windows = overlap_windows
        self.do_augment = opt['do_augment'] and subset == 'train'
            
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        if self.mode == 'train':
            self._get_data()
        elif self.mode == 'inference':
            self._get_inference_data()
        else:
            raise

    def _get_data(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = self.video_dict.keys()
        
        # Frame list is used when do_representation
        # NOTE: We restrict frame_list to have a fraction of the total number of frames
        # It is probably the case that we can expand the dataset a ton by not doing this
        # but then most of the examples are really correlated.
        # Instead, starting from frame 0, we select every skip'th frame.
        stride = self.skip_videoframes * self.num_videoframes
        if self.overlap_windows:
            stride = int(stride / 2)
            
        self.frame_list = []
        for k, v in sorted(self.video_dict.items()):
            num_frames = max(v['feature_frame'] - stride, 1)
            start_frames = [(k, i) for i in range(0, num_frames, stride)]
            # print('Video Dict: %s / total frames %d, frames after skipping %d' %  (k, num_frames, len(video_frames)))
            self.frame_list.extend(start_frames)

        print("%s subset video numbers: %d" %
              (self.subset, len(self.video_list)))
        print("%s subset frame numbers: %d" %
              (self.subset, len(self.frame_list)))

    def _get_inference_data(self):
        raise
    
    def _get_indices(self, index):
        video_name, frame_num = self.frame_list[index]
        start = frame_num
        end = start + self.num_videoframes * self.skip_videoframes
        return start, end

    def __getitem__(self, index):
        start, end = self._get_indices(index)
        video_data, anchor_xmin, anchor_xmax = self._get_base_data(
            index, start, end)
        if self.mode == "train":
            match_score_action, match_score_start, match_score_end = self._get_train_label(
                index, anchor_xmin, anchor_xmax, start, end)
            return video_data, match_score_action, match_score_start, match_score_end
        else:
            # TODO: Fix this.
            return index, video_data, anchor_xmin, anchor_xmax, 'dummy'

    def _get_base_data(self, index, start=None, end=None):
        anchor_xmin = [
            self.skip_videoframes * i for i in range(self.num_videoframes)
        ]
        anchor_xmax = [
            self.skip_videoframes * i for i in range(1, self.num_videoframes + 1)
        ]

        # Instead of passing back the features here, we need to pass back the images.
        # We get these only between frames start and end.
        video_name, _ = self.frame_list[index]
        video_info = self.video_dict[video_name]
        fps = video_info['fps']
        max_frames = video_info['duration_frame']
        path = Path(video_info['abspath'])
        paths = [
            path / '{:010.4f}.npy'.format(i / fps)
            for i in range(start, end, self.skip_videoframes) \
            if i < max_frames
        ]

        imgs = [self.img_loading_func(p.absolute(), do_augment=self.do_augment)
                for p in paths if p.exists()]

        diff = len(list(range(start, end, self.skip_videoframes))) - len(imgs)
        if type(imgs[0]) == np.array:
            if diff > 0:
                imgs.extend([np.zeros(imgs[0].shape) for _ in range(diff)])
            video_data = np.array(imgs)
            video_data = torch.Tensor(video_data)
        elif type(imgs[0]) == torch.Tensor:
            if diff > 0:
                imgs.extend([torch.zeros(imgs[0].shape) for _ in range(diff)])
            video_data = torch.stack(imgs)
        # NOTE: video_data is [num_videoframes, 3, 426, 240]
        return video_data, anchor_xmin, anchor_xmax

    def _get_train_label(self,
                         index,
                         anchor_xmin,
                         anchor_xmax,
                         start=None,
                         end=None):
        video_name, _ = self.frame_list[index]
        video_info = self.video_dict[video_name]
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        fps = video_info['fps']
        corrected_second = float(feature_frame) / video_frame * video_second
        video_labels = video_info['annotations']
        if start is not None and end is not None:
            video_labels = [
                anno for anno in video_labels
                if start <= fps * anno['segment'][1]
            ]
            video_labels = [
                anno for anno in video_labels if end >= fps * anno['segment'][0]
            ]
            corrected_second = (end - start) / fps

        gt_bbox = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            if tmp_info['label'] == 'off':
                continue
            tmp_start = max(min(1, (tmp_info['segment'][0] - start*1./fps) / corrected_second),
                            0)
            tmp_end = max(min(1, (tmp_info['segment'][1] - start*1./fps) / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        if len(gt_bbox) == 0:
            # Only off in this segment.
            match_score_action = torch.Tensor([0 for _ in range(len(anchor_xmin))])
            match_score_start = torch.Tensor([0 for _ in range(len(anchor_xmin))])
            match_score_end = torch.Tensor([0 for _ in range(len(anchor_xmin))])
            return torch.Tensor(match_score_action), torch.Tensor(match_score_start), torch.Tensor(match_score_end)
            
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]

        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = np.maximum(self.skip_videoframes,
                                  self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack(
            (gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack(
            (gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        match_score_action = []
        for jdx in range(len(anchor_xmin)):
            match_score_action.append(
                np.max(
                    ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                           gt_xmins, gt_xmaxs)))
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(
                np.max(
                    ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                           gt_start_bboxs[:, 0],
                                           gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(
                np.max(
                    ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                           gt_end_bboxs[:, 0],
                                           gt_end_bboxs[:, 1])))

        match_score_action = torch.Tensor(match_score_action)
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        return match_score_action, match_score_start, match_score_end

    def __len__(self):
        return len(self.frame_list)
    

class ProposalSampler(data.WeightedRandomSampler):
    def __init__(self, proposals, frame_list, max_zero_weight=0.25):
        """
        We are jsut trying to even out the 0 samples from everything else. We don't want those to dominate.

        Args:
          proposals: A dict of video_name key to pandas data frame.
          indices: A list of (key, index into that key's data frame). 
          This is what the Dataset is using and what we need to give sample weights for.
        """
        video_zero_indices = {k: set() for k in proposals}
        video_total_counts = {k: 0 for k in proposals}
        for video_name, pdf in proposals.items():
            video_total_counts[video_name] = len(pdf)
            video_zero_indices[video_name] = set([
                num for num, iou in enumerate(pdf.match_iou.values[:]) \
                if iou == 0
            ])
            
        weights = []
        curr_vid = None
        switched_counts = {k: 0 for k in proposals}
        for num, (video_name, pdf_num) in enumerate(frame_list):
            count_zeros = len(video_zero_indices[video_name]) * 1.
            count_total = video_total_counts[video_name]
            percent = count_zeros / count_total
            if curr_vid is None or video_name != curr_vid:
                # print('switching to %s with percent %.04f' % (video_name, percent))
                curr_vid = video_name
                
            if percent < max_zero_weight:
                # We don't care if there aren't many zeros.
                weights.append(1)
                continue

            # Otherwise, we roughly want there to be 10% zeros at most.
            # Say the original count of zeros and nonzeros is x, y.
            # We want the final distro to be .1 / .9, so weight the
            # zeros by w = .1/x and the nonzeros by .9/y. This yields
            # a prob of .1/x for each zero, which then yields .1 total.
            if pdf_num in video_zero_indices[video_name]:
                # Weight zero classes by (1 - percent)
                weights.append(max_zero_weight / count_zeros)
            else:
                # Weight non-zero classes by percent.
                weights.append((1 - max_zero_weight) / (count_total - count_zeros))

        super(ProposalSampler, self).__init__(weights, len(weights), replacement=True)

        
class ProposalDataSet(data.Dataset):

    def __init__(self, opt, subset="train"):

        self.subset = subset
        self.opt = opt
        self.mode = opt["mode"]
        if self.mode == "train":
            self.top_K = opt["pem_top_K"]
        else:
            self.top_K = opt["pem_top_K_inference"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self._getDatasetDict()

    def _exists(self, video_name):
        pgm_proposals_path = os.path.join(self.opt['pgm_proposals_dir'], '%s.proposals.csv' % video_name)
        pgm_features_path = os.path.join(self.opt['pgm_features_dir'], '%s.features.npy' % video_name)
        return os.path.exists(pgm_proposals_path) and os.path.exists(pgm_features_path)
        
    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            
            if 'thumos' in self.opt['dataset']:
                video_subset = video_name.split('_')[1].replace('validation', 'train')
            else:
                video_subset = anno_df.subset.values[i]
                
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = sorted(self.video_dict.keys())
        self.video_list = [k for k in self.video_list if self._exists(k)]

        if self.opt['pem_do_index']:
            self.features = {}
            self.proposals = {}
            self.indices = []
            for video_name in self.video_list:
                pgm_proposals_path = os.path.join(self.opt['pgm_proposals_dir'], '%s.proposals.csv' % video_name)
                pgm_features_path = os.path.join(self.opt['pgm_features_dir'], '%s.features.npy' % video_name)
                pdf = pd.read_csv(pgm_proposals_path)                    
                video_feature = np.load(pgm_features_path)
                pre_count = len(pdf)
                if self.top_K > 0:
                    try:
                        pdf = pdf.sort_values(by="score", ascending=False)
                    except KeyError:
                        pdf['score'] = pdf.xmin_score * pdf.xmax_score
                        pdf = pdf.sort_values(by="score", ascending=False)
                    pdf = pdf[:self.top_K]
                    video_feature = video_feature[pdf.index]
                    
                # print(video_name, pre_count, len(pdf), video_feature.shape, pgm_proposals_path, pgm_features_path)
                self.proposals[video_name] = pdf
                self.features[video_name] = video_feature
                self.indices.extend([(video_name, i) for i in range(len(pdf))])
            print('Num indices: ', len(self.indices))

    def __len__(self):
        if self.opt['pem_do_index'] > 0:
            return len(self.indices)
        else:
            return len(self.video_list)

    def __getitem__(self, index):
        if self.opt['pem_do_index']:
            video_name, video_index = self.indices[index]
            video_feature = self.features[video_name][video_index]
            video_feature = torch.Tensor(video_feature)
            pdf = self.proposals[video_name]
            match_iou = pdf.match_iou.values[video_index:video_index+1]
            video_match_iou = torch.Tensor(match_iou)
            if self.mode == 'train':
                return video_feature, video_match_iou
            else:
                video_xmin = pdf.xmin.values[video_index:video_index+1]
                video_xmax = pdf.xmax.values[video_index:video_index+1]
                video_xmin_score = pdf.xmin_score.values[video_index:video_index+1]
                video_xmax_score = pdf.xmax_score.values[video_index:video_index+1]
                return index, video_feature, video_xmin, video_xmax, video_xmin_score, video_xmax_score
        else:
            video_name = self.video_list[index]
            pgm_proposals_path = os.path.join(self.opt['pgm_proposals_dir'], '%s.proposals.csv' % video_name)
            pgm_features_path = os.path.join(self.opt['pgm_features_dir'], '%s.features.npy' % video_name)
        
            pdf = pd.read_csv(pgm_proposals_path)
            # I added in this:
            # ***
            pdf = pdf.sort_values(by="score", ascending=False)
            # ***
            video_feature = np.load(pgm_features_path)
            if self.top_K > 0:
                pdf = pdf[:self.top_K]
                video_feature = video_feature[:self.top_K, :]
                
            video_feature = torch.Tensor(video_feature)

            if self.mode == "train":
                video_match_iou = torch.Tensor(pdf.match_iou.values[:])
                return video_feature, video_match_iou
            else:
                video_xmin = pdf.xmin.values[:]
                video_xmax = pdf.xmax.values[:]
                video_xmin_score = pdf.xmin_score.values[:]
                video_xmax_score = pdf.xmax_score.values[:]
                return video_feature, video_xmin, video_xmax, video_xmin_score, video_xmax_score


def make_on_anno_files(mmd, videotable):
    regex = re.compile('https://storage.googleapis.com/spaceofmotion/(.*)-(\d{2}\.\d{2}\.\d{2}\.\d{3})-(\d{2}\.\d{2}\.\d{2}\.\d{3}).*.comp.mp4')
    path = Path('.')
    newmmd = {k: {'abspath': (path / k).absolute(), 'threads': []} for k in videotable.values()}
    for motion in mmd:
        match = reg.match(motion['video_location'])
        if not match:
            for thread in motion['threads']:
                thread['motion_video_location'] = motion['video_location']
                thread['motion_master_video'] = motion['master_video']
                newmmd[video]['threads'].append(thread)
            continue
        
        videokey, start, end = match.groups()
        video = videotable[videokey]
        sh, sm, ss, sms = start.split('.')
        eh, em, es, ems = end.split('.')
        start = int(sh)*3600 + int(sm)*60 + int(ss) + int(sms)*1./1000
        end = int(eh)*3600 + int(em)*60 + int(es) + int(ems)*1./1000
        for thread in motion['threads']:
            newthread = {}
            for thk, thv in thread.items():
                if thk == 'start_time':
                    newthread[thk] = thv + start
                elif thk == 'end_time':
                    newthread[thk] = thv + start
                elif thk == 'remarks':
                    newremarks = []
                    for remark in thv:
                        newrem = {}
                        for remk, remv in remark.items():
                            if remk == 'start_time':
                                newrem[remk] = remv + start
                            elif remk == 'end_time':
                                newrem[remk] = remv + start
                            else:
                                newrem[remk] = remv
                        newremarks.append(newrem)
                    newthread[thk] = newremarks
                else:
                    newthread[thk] = thv
            # if thread['thread_slug'] == 'accomplish-gain-do-thread-1':
            #     print(motion, start, end, newthread['start_time'], newthread['end_time'], thread['start_time'], thread['end_time'])
            newthread['motion_video_location'] = motion['video_location']
            newthread['motion_master_video'] = motion['master_video']
            newmmd[video]['threads'].append(newthread)

                        
    on_anno = {}
    for k, v in mmd.items():
        current_start = None
        current_end = None
        wv = {i:j for i, j in v.items()}
        wv['annotations'] = []
        for anno in v['annotations']:
            s, e = anno['segment']
            if current_start is None:
                current_start = s
                current_end = e
            elif s <= current_end:
                current_end = max(e, current_end)
            else:
                wv['annotations'].append({'label': 'on', 'segment': [current_start, current_end]})
                current_start = s
                current_end = e
        wv['annotations'].append({'label': 'on', 'segment': [current_start, current_end]})
        onmmd_anno[k] = wv
        
            
            
            
            
