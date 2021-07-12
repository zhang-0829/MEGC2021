# *_*coding:utf-8 *_*
import os
import sys
sys.path.insert(0, '../')
print(sys.path)
import time
import argparse
import openpyxl
import torch
from torch.utils.data import DataLoader
from utils.data_util import get_transform
from evaluation.util import VideoDataset, load_model, predict, count, cal_metric


def get_ground_truth(label_file, img_dir):
    def get_naming_rule(ws_naming1, ws_naming2):
        idx2sub = {}
        for row in ws_naming1.rows:
            vals = []
            for ele in row:
                vals.append(ele.value)
            idx2sub[vals[2]] = vals[1]  # ex: 1 --> s15
        exp2id = {}
        for row in ws_naming2.rows:
            vals = []
            for ele in row:
                vals.append(ele.value)
            exp2id[vals[1]] = vals[0]  # ex: anger1 --> 0401
        return idx2sub, exp2id

    # load label
    wb = openpyxl.load_workbook(label_file)
    ws_label, ws_naming1, ws_naming2  = wb.worksheets
    idx2sub, exp2id = get_naming_rule(ws_naming1, ws_naming2)
    subjects = list(idx2sub.values()) # global subjects

    # get video dirs
    video_dirnames = {}
    for sub_dir in os.scandir(img_dir):
        if sub_dir.is_dir():
            sub_name = sub_dir.name
            video_dirnames.setdefault(sub_name, [])
            for video_dir in os.scandir(sub_dir):
                if video_dir.is_dir():
                    video_dirname = video_dir.name
                    video_dirnames[sub_name].append(video_dirname)

    label_dict = {}
    for row in ws_label.rows:
        vals = []
        for ele in row:
            vals.append(ele.value)
        sub_idx, exp_name, onset, apex, offset, au, polarity, exp_type, exp = vals

        # get ground truth interval for expressions
        start = onset
        end = offset if offset != 0 else apex
        interval = (start, end)

        # get video dirname
        sub_name = idx2sub[sub_idx]
        exp_id = exp2id[exp_name.split('_')[0]]
        video_dirname = ''
        for ele in video_dirnames[sub_name]:
            if exp_id in ele:
                video_dirname = ele
                break
        assert video_dirname != '', f"Could not find video dirname for '{exp_name}' of '{sub_name}' "

        label_dict.setdefault(exp_type, {})
        label_dict[exp_type].setdefault(sub_name, {})
        label_dict[exp_type][sub_name].setdefault(video_dirname, [])
        label_dict[exp_type][sub_name][video_dirname].append(interval) # note that some subjects do not have macro/micro-expressions

    return label_dict, subjects


def get_checkpoint_path(file):
    with open(file, 'r') as f:
        checkpoint_paths = [line.strip() for line in f.readlines()]
        return checkpoint_paths


def main(params):
    start_time = time.time()

    # get ground truth
    label_file = os.path.join(params.data_dir, 'CAS(ME)^2code_final.xlsx')
    img_dir = os.path.join(params.data_dir, 'longVideoFaceCropped')
    label_dict, subjects = get_ground_truth(label_file, img_dir) # subjects are global, total 22 for CASME
    macro_label_dict = label_dict['macro-expression']

    # get transform
    transform = get_transform(partition='val')
    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log file
    log_file = f'./CASME_Submission_pos_th_{params.pos_th}_merge_th_{params.merge_th}.log'

    n_true_pos, n_pos, n_true = 0, 0, 0
    for sub_idx, sub_name in enumerate(subjects, 1): # for each subject (global subjects)
        if params.exclude and sub_name not in macro_label_dict:
            continue
        sub_dir = os.path.join(img_dir, sub_name)
        print('=' * 30 + f"Processing subject '{sub_name}' ({sub_idx}/{len(subjects)})" + '=' * 30)
        # load model
        # checkpoint_path = os.path.join(params.model_dir, f'split{sub_idx}/model_best.pth')
        checkpoint_path = os.path.join(params.model_dir, f'{sub_name}/model_best.pth')
        print('Loading checkpoint: {} ...'.format(checkpoint_path))
        model = load_model(checkpoint_path, len(params.gpus), device)
        for video_dirname in sorted(os.listdir(sub_dir), key=lambda x: int(x[3:7])): # for each video (global videos)
            # the videos with no macro-expressions are excluded
            if params.exclude and video_dirname not in macro_label_dict[sub_name]:
                continue
            video_dir = os.path.join(sub_dir, video_dirname)
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
    assert n_true == 300, 'Invalid number for true intervals!'
    precision, recall, f1_score = cal_metric(n_true_pos, n_pos, n_true)
    print(f'Total counts: n_true_pos = {n_true_pos}, n_pos = {n_pos}, n_true = {n_true}')
    print(f'Final results: precision = {precision:.4f}, recall = {recall:.4f}, f1_score = {f1_score:.4f}')
    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate CASME')
    parser.add_argument('--data_dir', type=str, default="/data5/sunlicai/Dataset/MEGC/2021/CAS(ME)2_longVideoFaceCropped",
                        help='root dir of dataset')
    # parser.add_argument('--model_dir', type=str, default="/data5/sunlicai/Code/MEGC_2021/saved/CASME/Test/0625_000436",
    #                     help='root dir of trained model')
    parser.add_argument('--model_dir', type=str, default="/data5/sunlicai/Code/MEGC_2021/saved/CASME/Test/0701_094851",
                        help='root dir of trained model') # new model train in 07/01
    parser.add_argument('--win_len', type=int, default=16, help='window len')
    parser.add_argument('--hop_len', type=int, default=8, help='hop len')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--pos_th', type=float, default=0.7, help='positive threshold (0.5-1)')
    parser.add_argument('--merge_th', type=int, default=1, help='threshold (>=1) for merging adjacent spotted intervals')
    parser.add_argument('--gpus', type=str, default='2', help='gpus used')
    parser.add_argument('--exclude', type=bool, default=True, help='whether evaluate non-expression videos')

    params = parser.parse_args()
    main(params)