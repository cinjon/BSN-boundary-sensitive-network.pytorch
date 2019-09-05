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


class VideoDataSet(data.Dataset):

    def __init__(self, opt, subset="train", img_loading_func=None, overlap_windows=False):
        self.temporal_scale = opt["temporal_scale"]
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]
        self.img_loading_func = img_loading_func
        self.overlap_windows = overlap_windows

        # feature path is where the features are stored.
        # If we are using a representation_module, then this shoudl
        # be the image paths instead.
        self.do_representation = opt['do_representation']
        if self.do_representation:
            self.num_videoframes = opt['num_videoframes']
            self.skip_videoframes = opt['skip_videoframes']
            # lol, yeah.
            self.temporal_scale = self.num_videoframes
            self.temporal_gap = 1. / self.temporal_scale
        else:
            self.feature_path = opt["feature_path"]
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
        if self.do_representation:
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
        start = end = None
        if self.do_representation:
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

        if self.do_representation:
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
        else:
            video_name = self.video_list[index]
            video_df = pd.read_csv(self.feature_path + "csv_mean_" +
                                   str(self.temporal_scale) + "/" + video_name +
                                   ".csv")
            video_data = video_df.values[:, :]
            video_data = torch.Tensor(video_data)
            video_data = torch.transpose(video_data, 0, 1)
            ### NOTE: This was uncommented in original version.
            # video_data.float()
        return video_data, anchor_xmin, anchor_xmax

    def _get_train_label(self,
                         index,
                         anchor_xmin,
                         anchor_xmax,
                         start=None,
                         end=None):
        if self.do_representation:
            video_name, _ = self.frame_list[index]
        else:
            video_name = self.video_list[index]

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

        # NOTE: With the original code, there was an assumption that this was the entire video.
        # We don't make that assumption and so in training we need to correct for this by telling
        # the model to only predict over the range of time given.
        gt_bbox = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            if tmp_info['label'] == 'off':
                continue
            tmp_start = max(min(1, (tmp_info['segment'][0] - int(self.do_representation)*start*1./fps) / corrected_second),
                            0)
            tmp_end = max(min(1, (tmp_info['segment'][1] - int(self.do_representation)*start*1./fps) / corrected_second), 0)
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
                    self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                           gt_xmins, gt_xmaxs)))
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(
                np.max(
                    self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                           gt_start_bboxs[:, 0],
                                           gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(
                np.max(
                    self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                           gt_end_bboxs[:, 0],
                                           gt_end_bboxs[:, 1])))

        match_score_action = torch.Tensor(match_score_action)
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        return match_score_action, match_score_start, match_score_end

    def _ioa_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores

    def __len__(self):
        if self.do_representation:
            return len(self.frame_list)
        return len(self.video_list)


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
        # print(pgm_proposals_path, pgm_features_path)
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
        print('Bef Len vidlist: ', self.video_list)
        self.video_list = [k for k in self.video_list if self._exists(k)]
        print('Aft Len vidlist: ', self.video_list)        

        if self.opt['pem_top_threshold']:
            print('Doing top threshold...')
            self.features = {}
            self.proposals = {}
            self.indices = []
            for video_name in self.video_list:
                pgm_proposals_path = os.path.join(self.opt['pgm_proposals_dir'], '%s.proposals.csv' % video_name)
                pgm_features_path = os.path.join(self.opt['pgm_features_dir'], '%s.features.npy' % video_name)

                print(pgm_features_path)
                pdf = pd.read_csv(pgm_proposals_path)
                pdf = pdf.sort_values(by="score", ascending=False)
                pre_count = len(pdf)
                count = len(pdf[pdf.score > self.opt['pem_top_threshold']])
                video_feature = np.load(pgm_features_path)
                video_feature = video_feature[pdf[:count].index]
                pdf = pdf[:count]
                print(video_name, pre_count, len(pdf), video_feature.shape)
                print(pdf[:5])
                print('\n')
                
                self.proposals[video_name] = pdf
                self.features[video_name] = video_feature
                self.indices.extend([(video_name, i) for i in range(len(pdf))])
            print('Num indices: ', len(self.indices))

    def __len__(self):
        if self.opt['pem_top_threshold'] > 0:
            return len(self.indices)
        else:
            return len(self.video_list)

    def __getitem__(self, index):
        if self.opt['pem_top_threshold']:
            video_name, video_index = self.indices[index]
            video_feature = self.features[video_name][video_index]
            video_feature = torch.Tensor(video_feature)
            match_iou = self.proposals[video_name].match_iou.values[video_index:video_index+1]
            video_match_iou = torch.Tensor(match_iou)
            if self.mode == 'train':
                return video_feature, video_match_iou
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
                print('from getitiem')
                print(video_feature.shape)
                print(video_match_iou.shape)
                return video_feature, video_match_iou
            else:
                video_xmin = pdf.xmin.values[:]
                video_xmax = pdf.xmax.values[:]
                video_xmin_score = pdf.xmin_score.values[:]
                video_xmax_score = pdf.xmax_score.values[:]
                return video_feature, video_xmin, video_xmax, video_xmin_score, video_xmax_score
