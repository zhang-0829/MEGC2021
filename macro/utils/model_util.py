# *_*coding:utf-8 *_*
import torch


def get_trainable_params(model, config):
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                params.append({'params': param, 'lr': config['optimizer']['args']['lr'] / 10})
            else:
                params.append({'params': param})
    return params


def load_checkpoint(pretrained_path):
    checkpoint = torch.load(pretrained_path)
    state_dict = checkpoint['state_dict']
    new_state_dict = {name.replace('module.', ''): param for name, param in state_dict.items()}
    return new_state_dict
