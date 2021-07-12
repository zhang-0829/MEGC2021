# *_*coding:utf-8 *_
import os
import time
import random
import pickle
import glob
import random
import openpyxl
from collections import OrderedDict
from itertools import chain
import json
import numpy as np
import pandas as pd
import accimage
import torch
import torchvision
from torch.utils.data import Dataset as BaseDataset
from torchvision.datasets.folder import default_loader


class CASMEDataset(BaseDataset):
    def __init__(self, root_dir, partition, split, neg_factor=2, transform=None, img_ext='jpg',
                 n_frames=16, stride=1, pad_type='last'):
        # note: default padding mode is "last", not "cycle"; default img_ext is "jpg".
        self.root_dir = root_dir
        self.label_file = os.path.join(self.root_dir, 'CAS(ME)^2code_final.xlsx')
        self.img_dir = os.path.join(self.root_dir, 'longVideoFaceCropped')

        self.partition = partition
        assert split in range(1,23), 'Invalid split number for CASME dataset!'
        self.split = split
        self.neg_factor = neg_factor
        self.transform = transform
        self.img_ext = img_ext
        self.n_frames = n_frames
        self.stride = stride # sampling rate
        self.pad_type = pad_type
        self.img_loader = default_loader # read image

        self.win_len = 1 + (self.n_frames - 1) * self.stride
        self.hop_len = self.win_len // 2

        # self.label_dict, self.idx2sub, self.exp2id = meta_info['label_dict'], meta_info['idx2sub'], meta_info['exp2id']
        self.label_dict, self.idx2sub, self.exp2id, self.subjects = self._process_label_file(self.label_file)
        self.split_sub = self.subjects[self.split-1]
        # self.data = self._get_macro_train_val_data(self.partition, self.split, self.label_dict, self.idx2sub, self.exp2id)
        self.data = self._get_macro_train_val_data(self.partition, self.split_sub, self.label_dict, self.idx2sub, self.exp2id)
        print(f"Total samples for '{self.partition}' part of split '{self.split_sub}' ({self.split}/{len(self.subjects)}): {len(self.data)}.")
        self.label_distribution = self._get_label_distribution(self.data)
        print(f'Label distribution (0 for non-expression): {self.label_distribution}.')

    def _get_naming_rule(self, ws_naming1, ws_naming2):
        idx2sub = {}
        for row in ws_naming1.rows:
            vals = []
            for ele in row:
                vals.append(ele.value)
            idx2sub[vals[2]] = vals[1] # ex: 1 --> s15
        exp2id = {}
        for row in ws_naming2.rows:
            vals = []
            for ele in row:
                vals.append(ele.value)
            exp2id[vals[1]] = vals[0] # ex: anger1 --> 0401
        return idx2sub, exp2id

    def _process_label_file(self, label_file):
        wb = openpyxl.load_workbook(label_file)
        ws_label, ws_naming1, ws_naming2  = wb.worksheets
        idx2sub, exp2id = self._get_naming_rule(ws_naming1, ws_naming2)
        subjects = list(idx2sub.values())

        label_dict = {}
        for row in ws_label.rows:
            vals = []
            for ele in row:
                vals.append(ele.value)
            sub_idx, exp_name, onset, apex, offset, au, polarity, exp_type, exp = vals
            sample_dict = {'exp_name':exp_name, 'onset':onset, 'apex':apex,
                           'offset':offset, 'au':au, 'polarity':polarity, 'exp':exp}
            # sub_name = idx2sub[sub_idx]
            # exp_id = exp2id[exp_name]
            label_dict.setdefault(exp_type, {})
            label_dict[exp_type].setdefault(sub_idx, [])
            label_dict[exp_type][sub_idx].append(sample_dict)

        return label_dict, idx2sub, exp2id, subjects

    def _get_label_distribution(self, data):
        label_count = {}
        for sample in data:
            label = sample['label']
            label_count.setdefault(label, 0)
            label_count[label] += 1
        return label_count

    def _get_macro_train_val_data(self, partition, split_sub, label_dict, idx2sub, exp2id):
        # get marco-expression intervals
        macro_label_dict = label_dict['macro-expression']
        macro_intervals = {} # {sub_name: {video_dirname: [(start, end]), ...}, ...}
        for sub_idx in macro_label_dict.keys():
            sub_name = idx2sub[sub_idx]
            macro_intervals.setdefault(sub_name, {})
            for sample in macro_label_dict[sub_idx]:
                exp_name = sample['exp_name']
                video_name = exp_name.split('_')[0]

                vid = exp2id[video_name]
                sub_dir = os.path.join(self.img_dir, sub_name)
                video_dirname = ''
                for ele in os.listdir(sub_dir):
                    if vid in ele:
                        video_dirname = ele
                        break
                assert video_dirname, 'Error: could not find video dir!'

                macro_intervals[sub_name].setdefault(video_dirname, [])
                start = sample['onset']
                end = sample['offset'] if sample['offset'] != 0 else sample['apex'] # use apex instead if offset is 0
                macro_intervals[sub_name][video_dirname].append([start, end])

        # get non-marco-expression intervals
        total_videos = 0
        non_macro_intervals = {} # {sub_name: {video_dirname: [(start, end]), ...}, ...}
        for sub_name in self.subjects: # note that macro_label_dict.keys() is a subset 0f idx2sub.values(), because some subjects do not have macro-expressions
            sub_dir = os.path.join(self.img_dir, sub_name)
            non_macro_intervals.setdefault(sub_name, {})
            for video_dirname in os.listdir(sub_dir):
                total_videos += 1
                non_macro_intervals[sub_name].setdefault(video_dirname, [])
                video_dir = os.path.join(self.img_dir, sub_name, video_dirname)
                img_files = glob.glob(os.path.join(video_dir, f'*.{self.img_ext}'))
                global_interval = [1, len(img_files)]

                if sub_name in macro_intervals and video_dirname in macro_intervals[sub_name]:
                    start = global_interval[0]
                    for interval in macro_intervals[sub_name][video_dirname]:
                        end = interval[0] - 1
                        # if (end - start) > self.n_frames:
                        if (end - start + 1) > self.win_len:
                            non_macro_intervals[sub_name][video_dirname].append([start, end])
                        start = interval[1] + 1
                    end = global_interval[1]
                    # if (end - start) > self.n_frames:
                    if (end - start + 1) > self.win_len:
                        non_macro_intervals[sub_name][video_dirname].append([start, end])
                else:
                    non_macro_intervals[sub_name][video_dirname].append(global_interval)

        # construct macro-expression samples
        data = []
        for sub_name in macro_intervals.keys():
            # if partition == 'val' and sub_name != idx2sub[split]:
            if partition == 'val' and sub_name != split_sub:
                continue
            # if partition == 'train' and sub_name == idx2sub[split]:
            if partition == 'train' and sub_name == split_sub:
                continue
            if partition == 'train':
                for video_dirname in macro_intervals[sub_name].keys():
                    intervals = macro_intervals[sub_name][video_dirname]
                    video_dir = os.path.join(self.img_dir, sub_name, video_dirname)
                    for interval in intervals:
                        sample = {'label': 1, 'interval': interval, 'video_dir': video_dir} # 'marco-expression' : 1
                        data.append(sample)
            else: # 'val'
                for video_dirname in macro_intervals[sub_name].keys():
                    intervals = macro_intervals[sub_name][video_dirname]
                    video_dir = os.path.join(self.img_dir, sub_name, video_dirname)
                    for interval in intervals:
                        # old version
                        # for i in range(interval[0], interval[1] - win_len + 2, hop_len):
                        #     start, end = i, i + win_len - 1  # note: include the 'end' frame
                        #     sample = {'label': 1, 'interval': [start, end], 'video_dir': video_dir}  # 'marco-expression' : 1
                        #     data.append(sample)
                        # new version
                        start = interval[0]
                        end = start
                        while end < interval[1]:
                            end = min(start + self.win_len - 1, interval[1])
                            sample = {'label': 1, 'interval': [start, end], 'video_dir': video_dir}  # 'marco-expression' : 1
                            data.append(sample)
                            start = start + self.hop_len

        # construct non-macro-expression samples
        neg_data = []
        for sub_name in non_macro_intervals.keys():
            # if partition == 'val' and sub_name != idx2sub[split]:
            if partition == 'val' and sub_name != split_sub:
                continue
            # if partition == 'train' and sub_name == idx2sub[split]:
            if partition == 'train' and sub_name == split_sub:
                continue
            for video_dirname in non_macro_intervals[sub_name].keys():
                intervals = non_macro_intervals[sub_name][video_dirname]
                video_dir = os.path.join(self.img_dir, sub_name, video_dirname)

                # sample negative intervals (method 1)
                # weights = [len(interval) for interval in intervals]
                # sampled_intervals = random.choices(intervals, k=n_neg_per_video, weights=weights)
                # for interval in sampled_intervals:
                #     sample = {'label': 0, 'interval': interval, 'video_dir': video_dir}  # 'non-marco-expression' : 0
                #     data[partition].append(sample)

                # generate negative intervals (method 2)
                for interval in intervals:
                    start = interval[0]
                    end = start
                    while end < interval[1]:
                        end = min(start + self.win_len - 1, interval[1])
                        sample = {'label': 0, 'interval': [start, end], 'video_dir': video_dir} # 'non-marco-expression' : 0
                        neg_data.append(sample)
                        start = start + self.hop_len

        # sample negative samples (for method 2)
        n_pos = len(data)
        print('n_pos', n_pos)
        n_neg = max(int(n_pos * self.neg_factor), 15) # make sure that the number of non-macro-expression samples in 'val' >= 15 (15 ~= 300 / 22)
        print('n_neg', n_neg, 'total_neg_data', len(neg_data))
        sampled_neg_data = random.sample(neg_data, k=n_neg)
        data.extend(sampled_neg_data)

        return data

    def __len__(self):
        return len(self.data)

    # for train
    def load_one_clip(self, idx):
        clip = {} # {'video': [PIL_IMAGE, ...], 'label': int}
        sample = self.data[idx]
        # get frame files
        [start, end], video_dir, label = sample['interval'], sample['video_dir'], sample['label']
        frame_files = [os.path.join(video_dir, f'img_{i}.{self.img_ext}') for i in range(start, end+1)] # note that the end frame is included
        total_frames = len(frame_files)
        least_frames = (self.n_frames - 1) * self.stride + 1
        start_idx = 0 if total_frames <= least_frames else np.random.randint(0, total_frames - least_frames + 1) # randint:[low, high)
        
        frames = []
        for i in range(self.n_frames):
            index = start_idx + i * self.stride
            if self.pad_type == 'cycle':  # cycle
                index = index % total_frames
            else:  # repeat the last frame
                index = -1 if index >= total_frames else index
            frame = self.img_loader(frame_files[index])
            frames.append(frame)

        clip['video'] = frames
        clip['label'] = label

        return clip

    def __getitem__(self, idx):
        clip = self.load_one_clip(idx)
        video = self.transform(clip['video']) if self.transform is not None else clip['video']
        label = torch.tensor(clip['label'])
        return video, label


if __name__ == "__main__":
    from utils.data_util import get_transform
    transform  = get_transform(partition='val')

    import time
    s_t = time.time()
    root_dir = '/data5/sunlicai/Dataset/MEGC/2021/CAS(ME)2_longVideoFaceCropped'
    dataset = CASMEDataset(root_dir, partition='val', split=1, transform=transform)
    e_t = time.time()
    print(f'Time used: {e_t - s_t:.1f}s.')
    for sample in dataset.data:
        print(sample)

    video, label = dataset[0]
    print(video.size(), label)
