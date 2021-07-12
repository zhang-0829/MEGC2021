# *_*coding:utf-8 *_*
import os
import sys
sys.path.insert(0, '../')
print(sys.path)
import time
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.data_util import get_transform
from evaluation.util import VideoDataset, load_model, predict, count, cal_metric


def get_ground_truth(label_file):
    df = pd.read_excel(label_file, sheet_name='FACS_Movement_Only', skiprows=9, engine='openpyxl')

    label_dict = {}
    subjects = []
    for idx, row in df.iterrows():
        sub_name, exp_name, exp_type = f"{row['Subject']:03d}", row['Filename'], row['Type']
        exp_type = exp_type[:5] # for micro-expression name
        onset, apex, offset = round(row['Onset']), round(row['Apex']), round(row['Offset'])  # note that float type for these items in excel

        # get real start and end
        start = onset if onset > 0 else apex  # use apex instead if offset is 0
        end = offset
        # handle offset overflow problem
        """
        020_6_5002-5476.jpg does not exist for '020_6_5' ([3532,5476])
        036_7_8202-8298.jpg does not exist for '036_7_4' ([5873,8298])
        """
        if exp_name == '020_6_5':
            end = 5001
        elif exp_name == '036_7_4':
            end = 8201
        interval = (start, end)

        # get video dirname
        video_dirname = '_'.join(exp_name.split('_')[:2])

        label_dict.setdefault(exp_type, {})
        label_dict[exp_type].setdefault(sub_name, {})
        label_dict[exp_type][sub_name].setdefault(video_dirname, [])
        label_dict[exp_type][sub_name][video_dirname].append(interval)
        if sub_name not in subjects:
            subjects.append(sub_name)

    return label_dict, subjects


def get_video_dirname(img_dir):
    sub2video = {}
    for video_dir in os.scandir(img_dir):
        video_dirname = video_dir.name
        sub_name = video_dirname.split('_')[0]
        sub2video.setdefault(sub_name, [])
        sub2video[sub_name].append(video_dirname)
    return sub2video


def main(params):
    start_time = time.time()

    # get ground truth
    label_file = os.path.join(params.data_dir, 'SAMM_LongVideos_V2_Release.xlsx')
    img_dir = os.path.join(params.data_dir, 'SAMM_longvideos')
    label_dict, subjects = get_ground_truth(label_file) # subjects are global, total 30 for SAMM
    macro_label_dict = label_dict['Macro']
    sub2video = get_video_dirname(img_dir) # videos are global for each subject

    # get transform
    transform = get_transform(partition='val')
    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log file
    log_file = f'./SAMM_Submission_pos_th_{params.pos_th}_merge_th_{params.merge_th}_hop_len_{params.hop_len}.log'

    n_true_pos, n_pos, n_true = 0, 0, 0
    for sub_idx, sub_name in enumerate(subjects, 1): # for each subject (global subjects)
        if params.exclude and sub_name not in macro_label_dict:
            continue
        # if sub_name != '012':
        #     continue
        print('=' * 30 + f"Processing subject '{sub_name}' ({sub_idx}/{len(subjects)})" + '=' * 30)
        # load model
        checkpoint_path = os.path.join(params.model_dir, f'{sub_name}/model_best.pth')
        print('Loading checkpoint: {} ...'.format(checkpoint_path))
        model = load_model(checkpoint_path, len(params.gpus), device)
        # get video
        for video_dirname in sorted(sub2video[sub_name], key=lambda x: int(x[-1])): # for each video (global videos)
            # the videos with no macro-expressions are excluded
            if params.exclude and video_dirname not in macro_label_dict[sub_name]:
                continue
            # if video_dirname != '012_3':
            #     continue
            video_dir = os.path.join(img_dir, video_dirname) # note that there is no sub_name in video_dir
            print(f"==> Processing '{video_dirname}' of '{sub_name}'")
            # build data loader
            dataset = VideoDataset(video_dir, params.win_len, params.stride, params.hop_len, transform=transform)
            val_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, num_workers=4)
            # get model prediction
            pred_dict = predict(model, val_loader, device, pos_th=params.pos_th)
            # count numbers of true positives, positives, true for each video
            n_true_pos_v, n_pos_v, n_true_v = count(sub_name, video_dirname, macro_label_dict, pred_dict, log_file, merge_th=params.merge_th)
            print(f"{sub_name}/{video_dirname}: n_true_pos = {n_true_pos_v}, n_pos = {n_pos_v}, n_true = {n_true_v}\n")
            # aggregate
            n_true_pos += n_true_pos_v
            n_pos += n_pos_v
            n_true += n_true_v

    # calculate metrics
    precision, recall, f1_score = cal_metric(n_true_pos, n_pos, n_true)
    print(f'Total counts: n_true_pos = {n_true_pos}, n_pos = {n_pos}, n_true = {n_true}')
    print(f'Final results: precision = {precision:.4f}, recall = {recall:.4f}, f1_score = {f1_score:.4f}')
    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate SAMM')
    parser.add_argument('--data_dir', type=str, default='/data5/sunlicai/Dataset/MEGC/2021/SAMM',
                        help='root dir of dataset')
    parser.add_argument('--model_dir', type=str, default="/data5/sunlicai/Code/MEGC_2021/saved/SAMM/Test/0701_094851",
                        help='root dir of trained model')
    parser.add_argument('--win_len', type=int, default=16, help='window len') # note that this is not the real win len (1 + (win_len-1) * stride)
    parser.add_argument('--hop_len', type=int, default=7, help='hop len') # note that this is not the real hop len (hop_len * stride)
    parser.add_argument('--stride', type=int, default=7, help='stride')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--pos_th', type=float, default=0.5, help='positive threshold (0.5-1)')
    parser.add_argument('--merge_th', type=int, default=1, help='threshold (>=1) for merging adjacent spotted intervals')
    parser.add_argument('--gpus', type=str, default='2', help='gpus used')
    parser.add_argument('--exclude', type=bool, default=True, help='whether evaluate non-expression videos')

    params = parser.parse_args()
    main(params)