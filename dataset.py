# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

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
        self.temporal_gap = 5
        self.window_size = 100
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
        
        window_size = self.window_size
        temporal_gap = self.temporal_gap
        start_snippet = int((temporal_gap + 1) / 2)
        
        for video_name in video_name_list:
            anno_df_video = anno_df[anno_df.video == video_name]
            gt_xmins = anno_df_video.startFrame.values[:]
            gt_xmaxs = anno_df_video.endFrame.values[:]

            # NOTE: num_snippet is the number of snippets in this video.
            if self.image_dir:
                image_dir = os.path.join(self.image_dir, video_name)
                num_snippet = len(os.listdir(image_dir))
                num_snippet = int((num_snippet - start_snippet) / temporal_gap)
            elif self.feature_dirs:
                feature_dfs = [
                    pd.read_csv(os.path.join(feature_dir, '%s.csv' % video_name))
                    for feature_dir in self.feature_dirs
                ]
                num_snippet = min([len(df) for df in feature_dfs])
                df_data = np.concatenate([df.values[:num_snippet, :]
                                          for df in feature_dfs],
                                         axis=1)
                
            df_snippet = [start_snippet + temporal_gap*i for i in range(num_snippet)]
            stride = int(window_size / 2)
            num_windows = int((num_snippet + stride - window_size) / stride)
            windows_start = [i* stride for i in range(num_windows)]
            if num_snippet < window_size:
                windows_start = [0]
                df_snippet.extend([
                    df_snippet[-1] + temporal_gap*(i+1)
                    for i in range(window_size - num_snippet)
                ])
            elif num_snippet - windows_start[-1] - window_size > int(window_size / temporal_gap):
                windows_start.append(num_snippet - window_size)

            for start in windows_start:
                # if start + window_size > num_snippet:
                #     continue

                if self.feature_dirs:
                    tmp_data = df_data[start:start + window_size, :]                    
                tmp_snippets = np.array(df_snippet[start:start + window_size])
                tmp_anchor_xmins = tmp_snippets - temporal_gap/2.
                tmp_anchor_xmaxs = tmp_snippets + temporal_gap/2.
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

        print('Size of data: ', len(self.data['gt_bbox']))
        self.data = {
            'gt_bbox': list_gt_bbox,
            'anchor_xmins': list_anchor_xmins,
            'anchor_xmaxs': list_anchor_xmaxs,
            'video_names': list_videos,
            'indices': list_indices
        }
        if self.feature_dirs:
            self.data['video_data'] = list_data

    def __getitem__(self, index):
        anchor_xmin = self.data['anchor_xmins'][index]
        anchor_xmax = self.data['anchor_xmaxs'][index]
        video_data = self._get_video_data(self.data, index)
        
        if self.mode == "train":
            gt_bbox = self.data['gt_bbox'][index]        
            match_score_action, match_score_start, match_score_end = self._get_train_label(gt_bbox, anchor_xmin, anchor_xmax)
            return video_data, match_score_action, match_score_start, match_score_end
        else:
            return index, video_data, anchor_xmin, anchor_xmax

    def _get_train_label(self, gt_bbox, anchor_xmin, anchor_xmax):
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        # same as gt_len but using the thumos code repo :/.
        gt_duration = gt_xmaxs - gt_xmins
        gt_duration_boundary = np.maximum(
            self.temporal_gap, gt_duration * self.boundary_ratio)
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
        return len(self.data['gt_bbox'])

    
class ThumosFeatures(Thumos):

    def __init__(self, opt, subset=None, feature_dirs=[]):
        super(ThumosFeatures, self).__init__(opt, subset, feature_dirs, fps=None, image_dir=None, img_loading_func=None)

    def _get_video_data(self, data, index):
        return data['video_data'][index]
    

class ThumosImages(data.Dataset):

    def __init__(self, opt, subset=None, fps=30, image_dir=None, img_loading_func=None):
        super(ThumosImages, self).__init__(opt, subset, feature_dirs=None, fps=fps, image_dir=image_dir, img_loading_func=img_loading_func)        

    def _get_video_data(self, data, index):
        indices = data['indices'][index]
        name = data['video_names'][index]
        path = os.path.join(self.image_dir, video_name)
        path = Path(path)
        paths = [path / ('%010.4f.npy' % (i / self.fps)) for i in indices]
        imgs = [self.img_loading_func(p.absolute()) for p in paths]
        if type(imgs[0]) == np.array:
            video_data = np.array(imgs)
            video_data = torch.Tensor(video_data)
        elif type(imgs[0]) == torch.Tensor:
            video_data = torch.stack(imgs)
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

        per_video_frame_off_count = {k: 0 for k in video_dict.keys()}
        off_indices = {k: set() for k in video_dict.keys()}
        on_indices = {k: set() for k in video_dict.keys()}
        for k, video_info in video_dict.items():
            fps = video_info['fps']
            for anno in video_info['annotations']:
                end_frame = round(anno['segment'][1] * fps)
                start_frame = round(anno['segment'][0] * fps)
                count = end_frame - start_frame
                if anno['label'] == 'off':
                    per_video_frame_off_count[k] += count
                    off_indices[k].update(range(start_frame, end_frame))
                else:
                    on_indices[k].update(range(start_frame, end_frame))                    
        per_video_frame_off_ratio = {k: per_video_frame_off_count[k] * 1. / per_video_frame_count[k]
                                     for k in per_video_frame_off_count.keys()}

        for num, (k, frame) in enumerate(frame_list):
            if frame in off_indices[k]:
                weights[num] *= 0.5 / per_video_frame_off_ratio[k]
            else:
                weights[num] *= 0.5 / (1 - per_video_frame_off_ratio[k])

        super(GymnasticsSampler, self).__init__(weights, len(weights), replacement=True)

        
class GymnasticsDataSet(data.Dataset):

    def __init__(self, opt, subset="train", img_loading_func=None, overlap_windows=False):
        self.subset = subset
        self.mode = opt["mode"]
        self.img_loading_func = img_loading_func
        self.overlap_windows = overlap_windows

        self.num_videoframes = opt['num_videoframes']
        self.skip_videoframes = opt['skip_videoframes']
        self.temporal_scale = self.num_videoframes
        self.temporal_gap = 1. / self.temporal_scale
            
        self.boundary_ratio = opt["boundary_ratio"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self._getDatasetDict()

    def _getDatasetDict(self):
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
        skip = self.skip_videoframes * self.num_videoframes
        self.frame_list = []
        for k, v in sorted(self.video_dict.items()):
            num_frames = v['feature_frame'] - skip
            if self.overlap_windows:
                # In this scenario, we take overlapping windows. Used for training.
                video_frames = [(k, i) for i in range(0, num_frames, skip)]
            else:
                # In this scenario, windows do not overlap. Used for testing.
                video_frames = [(k, i) for i in range(0, num_frames, self.skip_videoframes)]                    
            print('Video Dict: %s / total frames %d, frames after skipping %d' %  (k, num_frames, len(video_frames)))
            self.frame_list.extend(video_frames)

        print("%s subset video numbers: %d" %
              (self.subset, len(self.video_list)))
        print("%s subset frame numbers: %d" %
              (self.subset, len(self.frame_list)))

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
            return index, video_data, anchor_xmin, anchor_xmax

    def _get_base_data(self, index, start=None, end=None):
        anchor_xmin = [
            self.temporal_gap * i for i in range(self.temporal_scale)
        ]
        anchor_xmax = [
            self.temporal_gap * i for i in range(1, self.temporal_scale + 1)
        ]

        # Instead of passing back the features here, we need to pass back the images.
        # We get these only between frames start and end.
        video_name, _ = self.frame_list[index]
        video_info = self.video_dict[video_name]
        fps = video_info['fps']
        path = Path(video_info['abspath'])
        paths = [
            path / '{0:.4f}.npy'.format(i / fps)
            for i in range(start, end, self.skip_videoframes)
        ]
        imgs = [self.img_loading_func(p.absolute()) for p in paths]
        if type(imgs[0]) == np.array:
            video_data = np.array(imgs)
            video_data = torch.Tensor(video_data)
        elif type(imgs[0]) == torch.Tensor:
            video_data = torch.stack(imgs)
        # NOTE: video_data is [num_videoframes, 3, 256, 448]
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
        gt_len_small = np.maximum(self.temporal_gap,
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
                print('switching to %s with percent %.04f' % (video_name, percent))
                curr_vid = video_name
                
            if percent < max_zero_weight:
                # We don't care if there aren't many zeros.
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
        print(pgm_proposals_path)
        print(pgm_features_path)
        return os.path.exists(pgm_proposals_path) and os.path.exists(pgm_features_path)
        
    def _getDatasetDict(self):
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
        self.video_list = sorted(self.video_dict.keys())
        self.video_list = [k for k in self.video_list if self._exists(k)]
        print('\n***\n')
        print(self.subset)
        print(self.video_list)

        if self.opt['pem_do_index']:
            self.features = {}
            self.proposals = {}
            self.indices = []
            for video_name in self.video_list:
                pgm_proposals_path = os.path.join(self.opt['pgm_proposals_dir'], '%s.proposals.csv' % video_name)
                pgm_features_path = os.path.join(self.opt['pgm_features_dir'], '%s.features.npy' % video_name)
                pdf = pd.read_csv(pgm_proposals_path)
                pdf = pdf.sort_values(by="score", ascending=False)
                video_feature = np.load(pgm_features_path)
                video_feature = video_feature[pdf[:self.top_K].index]
                pre_count = len(pdf)
                pdf = pdf[:self.top_K]
                print(video_name, pre_count, len(pdf), video_feature.shape)
                print('Num zeros in match_iou: ', len(pdf[pdf.match_iou == 0]))
                print('')
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
            pdf = pdf[:self.top_K]
            
            video_feature = np.load(pgm_features_path)
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
