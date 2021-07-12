# *_*coding:utf-8 *_*
import os
import openpyxl
from transforms import tensor_transforms, video_transforms, volume_transforms


def get_splits(config):
    dataset = config['data_loader']['args']['dataset_name']
    assert  dataset in ['CASME', 'SAMM'], 'Invalid dataset!'
    data_dir = config['data_loader']['args']['root_dir']
    img_dirname = 'longVideoFaceCropped' if dataset == 'CASME' else 'SAMM_longvideos'

    TOTAL_SPLITS = set(range(1, 23)) if dataset == 'CASME' else set(range(1, 31)) # 22 splits (subjects) for CASME, 30 splits for SAMM

    if "splits" in config['data_loader']:
        splits = set(config['data_loader']['splits'])
    else:
        img_dir = os.path.join(data_dir, img_dirname)
        sub_dirs = os.listdir(img_dir)
        if dataset == 'SAMM':
            sub_dirs = set([sub_dir.split('_')[0] for sub_dir in sub_dirs])
        n_splits = len(sub_dirs)
        splits = set(range(1, n_splits + 1))
    assert splits.issubset(TOTAL_SPLITS), f'Error: invalid splits ({splits}) for {dataset}Dataset.'

    return splits


def get_dataset_info(config):
    dataset = config['data_loader']['args']['dataset_name']
    data_dir = config['data_loader']['args']['data_dir']

    if dataset == 'CASME':
        label_file = os.path.join(data_dir, 'CAS(ME)^2code_final.xlsx')
        n_splits, meta_info = process_CASME_label_file(label_file)
    elif dataset == 'SAMM':
        label_file = os.path.join(data_dir, '')
        n_splits, meta_info = process_SAMM_label_file(label_file)
    else:
        raise ValueError('Invalid dataset!')

    meta_info['name'] = dataset

    return n_splits, meta_info


def process_CASME_label_file(label_file):
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

    wb = openpyxl.load_workbook(label_file)
    ws_label, ws_naming1, ws_naming2  = wb.worksheets
    idx2sub, exp2id = get_naming_rule(ws_naming1, ws_naming2)
    n_splits = len(idx2sub)
    assert n_splits == 22, f'Error: invalid number of subjects for CASME ({n_splits} vs. {22}).'

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

    meta_info = {'label_dict': label_dict, 'idx2sub': idx2sub, 'exp2id': exp2id}

    return n_splits, meta_info


def process_SAMM_label_file(label_file):
    n_splits = None
    meta_info = None



    return n_splits, meta_info


def get_transform(
    partition='train',
    type='sc',
    resize=(256, 256),
    crop=(224, 224),
    color=(0.4, 0.4, 0.4, 0.2),
    min_area=0.5
):
    if partition == 'train':
        if type == 'rsc': # RandomResizedCrop
            transforms = [
                video_transforms.RandomResizedCrop(crop, scale=(min_area, 1.)),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ColorJitter(*color),
            ]
        else: # 'sc': Resize + RandomCrop
            transforms = [
                video_transforms.Resize(resize),
                video_transforms.RandomCrop(crop),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ColorJitter(*color),
            ]
    else:
        if type == 'rsc':
            transforms = [
                video_transforms.Resize(int(crop[0] / 0.875)),
                video_transforms.CenterCrop(crop),
            ]
        else: # 'sc'
            transforms = [
                video_transforms.Resize(resize),
                video_transforms.CenterCrop(crop),
            ]

    # to tensor
    transforms += [volume_transforms.ClipToTensor()]
    # normalize
    transforms += [tensor_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    # compose
    transform = video_transforms.Compose(transforms)

    return transform