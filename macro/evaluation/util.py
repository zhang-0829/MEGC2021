# *_*coding:utf-8 *_*
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from model.model import CNNRNN


def predict(model, val_loader, device, pos_th=0.5):
    preds, intervals = [], []
    with torch.no_grad():
        for batch in val_loader:
            clips, clip_ids = batch
            clips = clips.to(device)
            logit = model(clips)
            pred = torch.sigmoid(logit)
            pred = (pred > pos_th).long().detach().cpu().numpy()
            preds.append(pred)
            intervals.extend(clip_ids.detach().cpu().numpy())

        preds = np.concatenate(preds).tolist()
        intervals = [(clip_id[0], clip_id[-1]) for clip_id in intervals] # (start, end)
        pred_dict = dict(zip(intervals, preds))

        return pred_dict


def load_model(checkpoint_path, n_gpu, device):
    # build model
    model = CNNRNN(backbone='EfficientFace')  # note that need to match those of pre-trained model
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # load checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = {name.replace('module.', ''):param for name, param in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    # prepare model for testing
    model = model.to(device)
    model.eval()

    return model


class VideoDataset(Dataset):
    def __init__(self, video_dir, win_len, stride, hop_len, transform=None, img_loader=default_loader, img_ext='jpg'):
        self.video_dir = video_dir
        self.win_len = win_len
        self.stride = stride
        self.hop_len = hop_len
        self.transform = transform
        self.img_loader = img_loader
        self.img_ext = img_ext
        self.img_files = sorted(glob.glob(os.path.join(self.video_dir, f'*.{self.img_ext}')),
                                key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # sliding window
        self.clips = []
        start = 0
        while True:
            end = min(start + (self.win_len - 1) * self.stride + 1, len(self.img_files))
            clip = list(range(start, end, self.stride))
            # padding the last frame if needed
            clip = clip + [clip[-1]] * (self.win_len - len(clip))
            self.clips.append(clip)
            if end == len(self.img_files):
                break
            start += self.hop_len * self.stride

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        clip = self.clips[item]
        frames = []
        frame_ids = []
        for idx in clip:
            img_file = self.img_files[idx]
            frame = self.img_loader(img_file)
            frames.append(frame)
            frame_id = int(img_file.split('_')[-1].split('.')[0])
            frame_ids.append(frame_id)

        frames = self.transform(frames) if self.transform is not None else frames
        frame_ids = torch.tensor(frame_ids)

        return frames, frame_ids


def is_true_positive(sp_interval, gt_intervals, threshold=0.5):
    hit, max_iou, gt_interval_best_matched = False, 0, (0, 0)
    ious = []

    pred_set = set(range(sp_interval[0], sp_interval[1] + 1)) # note that the end is included
    for gt_interval in gt_intervals:
        gt_set = set(range(gt_interval[0], gt_interval[1] + 1))
        intersection = pred_set.intersection(gt_set)
        union = pred_set.union(gt_set)
        iou = len(intersection) / len(union)
        ious.append(iou)

    if len(ious) > 0: # in case of gt_intervals is []
        max_iou = max(ious)
        gt_interval_best_matched = gt_intervals[ious.index(max_iou)]
        if max_iou >= threshold:
            hit = True

    return hit, max_iou, gt_interval_best_matched


def count(sub_name, video_dirname, label_dict, pred_dict, log_file, merge_th=1):
    n_true_pos, n_pos = 0, 0
    gt_intervals = label_dict[sub_name][video_dirname] if sub_name in label_dict and video_dirname in label_dict[sub_name] else []
    print(f'Ground truth intervals: \n{gt_intervals}')
    n_true = len(gt_intervals)
    # get spotted intervals
    sp_intervals = []
    for interval, pred in pred_dict.items():
        if pred == 1: # positive
            sp_intervals.append(interval)
    print(f'Spotted intervals: \n{sp_intervals}')
    # merge spotted intervals
    sp_intervals_merged = []
    last_sp_interval = None
    for idx, sp_interval in enumerate(sp_intervals):
        if idx == 0:
            last_sp_interval = sp_interval
            continue
        if (sp_interval[0] - last_sp_interval[1]) <= merge_th: # merge intervals
            last_sp_interval = (last_sp_interval[0], sp_interval[1])
        else: # update
            sp_intervals_merged.append(last_sp_interval)
            last_sp_interval = sp_interval
    if last_sp_interval is not None and last_sp_interval not in sp_intervals_merged:
        sp_intervals_merged.append(last_sp_interval)
    print(f'Spotted intervals (merged): \n{sp_intervals_merged}')
    # count
    n_pos = len(sp_intervals_merged)
    sp_gt_matched_dict = {}
    for sp_interval in sp_intervals_merged:
        # judge whether spotted interval is a true positive according to the iou with ground truth interval and the threshold
        hit, max_iou, gt_interval_best_matched = is_true_positive(sp_interval, gt_intervals)
        hit_flag = '✔' if hit else '✕'
        print(f"Spotted interval: [{sp_interval[0]:5d}, {sp_interval[1]:5d}], "
              f"best matched ground truth interval: [{gt_interval_best_matched[0]:5d}, {gt_interval_best_matched[1]:5d}], "
              f"max iou: {max_iou:.2f}, {hit_flag}")
        if hit: # true positive
            n_true_pos += 1
            sp_gt_matched_dict[sp_interval] = gt_interval_best_matched
            gt_intervals.remove(gt_interval_best_matched)

    # write to log using the format in the official provided sample.log
    intervals = [(interval, 'sp') for interval in sp_intervals_merged] + [(interval, 'gt') for interval in gt_intervals]
    intervals.sort(key=lambda x: x[0][0]) # sorted by start time of the interval
    video_name = video_dirname[:7] if sub_name[0] == 's' else video_dirname # 's' for CASME
    with open(log_file, 'a+') as f:
        for interval, type in intervals:
            if type == 'sp': # spotted interval
                if interval in sp_gt_matched_dict:
                    gt_interval = sp_gt_matched_dict[interval]
                    f.write(f"{video_name} {gt_interval[0]:5d} {gt_interval[1]:5d} {interval[0]:5d} {interval[1]:5d} TP\n")
                else:
                    f.write(f"{video_name}     -     - {interval[0]:5d} {interval[1]:5d} FP\n")
            else: # ground truth interval
                f.write(f"{video_name} {interval[0]:5d} {interval[1]:5d}     -     - FN\n")

    return n_true_pos, n_pos, n_true


def cal_metric(n_true_pos, n_pos, n_true):
    precision = n_true_pos / n_pos if n_pos > 0 else 0
    recall = n_true_pos / n_true if n_true > 0 else 0
    f1_score = 2 * recall * precision / (recall + precision)

    return precision, recall, f1_score