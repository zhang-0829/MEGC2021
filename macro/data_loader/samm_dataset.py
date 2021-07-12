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


class SAMMDataset(BaseDataset):
    def __init__(self, root_dir, partition, split, neg_factor=2, transform=None, img_ext='jpg',
                 n_frames=16, stride=1, pad_type='last'):
        # note: default padding mode is "last", not "cycle"; default img_ext is "jpg".
        self.root_dir = root_dir
        self.label_file = os.path.join(self.root_dir, 'SAMM_LongVideos_V2_Release.xlsx')
        self.img_dir = os.path.join(self.root_dir, 'SAMM_longvideos')

        self.partition = partition
        self.split = split
        self.neg_factor = neg_factor
        self.transform = transform
        self.img_ext = img_ext
        self.n_frames = n_frames
        self.stride = stride # sampling rate
        self.pad_type = pad_type
        self.img_loader = default_loader # read image

        self.win_len = 1 + (self.n_frames - 1) * self.stride
        self.hop_len = self.win_len // 4 # different from that for CASME (self.win_len // 2), 2021/06/30

        self.img_naming_format = self._get_img_naming_format()

        self.label_dict, self.subjects = self._process_label_file(self.label_file)
        self.split_sub = self.subjects[self.split-1]

        self.data = self._get_macro_train_val_data(self.partition, self.split_sub, self.label_dict)
        print(f"Total samples for '{self.partition}' part of split '{self.split_sub}' ({self.split}/{len(self.subjects)}): {len(self.data)}.")

        self.label_distribution = self._get_label_distribution(self.data)
        print(f'Label distribution (0 for non-expression): {self.label_distribution}.')

    def _get_img_naming_format(self):
        img_naming_format = {}
        for video_dirname in os.listdir(self.img_dir):
            video_dir = os.path.join(self.img_dir, video_dirname)
            img_filename = os.listdir(video_dir)[0]
            img_naming_format[video_dirname] = f"0{len(os.path.splitext(img_filename)[0].split('_')[-1])}d" # '04d', '05d'
        return img_naming_format

    def _process_label_file(self, label_file):
        df = pd.read_excel(label_file, sheet_name='FACS_Movement_Only', skiprows=9, engine='openpyxl')

        label_dict = {}
        subjects = []
        for idx, row in df.iterrows():
            sub_name, exp_name, exp_type, au = row['Subject'], row['Filename'], row['Type'], row['Action Units']
            onset, apex, offset = round(row['Onset']), round(row['Apex']), round(row['Offset']) # note that float type for these items in excel

            sub_name = f'{sub_name:03d}'
            sample_dict = {'exp_name':exp_name, 'onset':onset, 'apex':apex, 'offset':offset, 'au':au}

            label_dict.setdefault(exp_type, {})
            label_dict[exp_type].setdefault(sub_name, [])
            label_dict[exp_type][sub_name].append(sample_dict)
            if sub_name not in subjects:
                subjects.append(sub_name)

        return label_dict, subjects

    def _get_label_distribution(self, data):
        label_count = {}
        for sample in data:
            label = sample['label']
            label_count.setdefault(label, 0)
            label_count[label] += 1
        return label_count

    def _get_macro_train_val_data(self, partition, split_sub, label_dict):
        # get marco-expression intervals
        macro_label_dict = label_dict['Macro']
        macro_intervals = {} # {sub_name: {video_dirname: [(start, end]), ...}, ...}
        for sub_name in macro_label_dict.keys():
            macro_intervals.setdefault(sub_name, {})
            for sample in macro_label_dict[sub_name]:
                exp_name = sample['exp_name']
                video_dirname = '_'.join(exp_name.split('_')[:2])
                macro_intervals[sub_name].setdefault(video_dirname, [])

                start = sample['onset'] if sample['onset'] > 0 else sample['apex'] # use apex instead if offset is 0
                end = sample['offset']

                # handle offset overflow problem
                """
                020_6_5002-5476.jpg does not exist for '020_6_5' ([3532,5476])
                036_7_8202-8298.jpg does not exist for '036_7_4' ([5873,8298])
                """
                if exp_name == '020_6_5':
                    end = 5001
                elif exp_name == '036_7_4':
                    end = 8201

                # check offset
                # video_dir = os.path.join(self.img_dir, video_dirname)
                # for i in range(start, end+1):
                #     img_file = os.path.join(video_dir, f'{video_dirname}_{i:{self.img_naming_format[video_dirname]}}.{self.img_ext}')
                #     if not os.path.exists(img_file):
                #         print(f"Warning: {img_file} for '{exp_name}' ([{start},{end}]) macro-expression does not exist!")

                macro_intervals[sub_name][video_dirname].append([start, end])

        # get non-marco-expression intervals
        total_videos = 0
        non_macro_intervals = {} # {sub_name: {video_dirname: [(start, end]), ...}, ...}
        for video_dirname in os.listdir(self.img_dir):
            total_videos += 1
            sub_name = video_dirname.split('_')[0]
            non_macro_intervals.setdefault(sub_name, {})
            non_macro_intervals[sub_name].setdefault(video_dirname, [])
            video_dir = os.path.join(self.img_dir, video_dirname) # note: no subject dir for SAMM dataset.
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
            if partition == 'val' and sub_name != split_sub:
                continue
            if partition == 'train' and sub_name == split_sub:
                continue
            if partition == 'train':
                for video_dirname in macro_intervals[sub_name].keys():
                    intervals = macro_intervals[sub_name][video_dirname]
                    video_dir = os.path.join(self.img_dir, video_dirname)
                    for interval in intervals:
                        sample = {'label': 1, 'interval': interval, 'video_dir': video_dir} # 'marco-expression' : 1
                        data.append(sample)
            else: # 'val'
                for video_dirname in macro_intervals[sub_name].keys():
                    intervals = macro_intervals[sub_name][video_dirname]
                    video_dir = os.path.join(self.img_dir, video_dirname)
                    for interval in intervals:
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
            if partition == 'val' and sub_name != split_sub:
                continue
            if partition == 'train' and sub_name == split_sub:
                continue
            for video_dirname in non_macro_intervals[sub_name].keys():
                intervals = non_macro_intervals[sub_name][video_dirname]
                video_dir = os.path.join(self.img_dir, video_dirname)

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
        n_neg = max(int(n_pos * self.neg_factor), 12) # make sure that the number of non-macro-expression samples in 'val' >= 12 (12 ~= 343 / 30)
        if n_neg > len(neg_data):
            n_neg = len(neg_data) # for random.sample (split 29: ValueError: Sample larger than population or is negative)
            print('Warning: n_neg (expected sampled) > len(neg_data) (total available), force n_neg = len(neg_data)')
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
        video_dirname = os.path.basename(video_dir)
        frame_files = [os.path.join(video_dir, f'{video_dirname}_{i:{self.img_naming_format[video_dirname]}}.{self.img_ext}') for i in range(start, end+1)] # note that the end frame is included
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
    import transforms.volume_transforms as vt
    import transforms.video_transforms as transform
    transform = transform.Compose([transform.Resize((20,20)), vt.ClipToTensor()])

    import time
    s_t = time.time()
    root_dir = '/data5/sunlicai/Dataset/MEGC/2021/SAMM'
    dataset = SAMMDataset(root_dir, partition='val', split=29, transform=transform, n_frames=16, stride=7)
    e_t = time.time()
    print(f'Time used: {e_t - s_t:.1f}s.')
    for sample in dataset.data:
        print(sample)

    video, label = dataset[0]
    print(video, label)

    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    count = {}
    for batch in data_loader:
        videos, labels = batch
        for label in labels:
            count.setdefault(label.item(), 0)
            count[label.item()] += 1
            print(label)
    print(count)