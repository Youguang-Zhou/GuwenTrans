import logging
import os
import subprocess
from logging import FileHandler, StreamHandler
from pathlib import Path

import torch

root_path = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).rstrip().decode('utf-8'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def move_to_device(sample):
    if torch.is_tensor(sample):
        return sample.to(device)
    elif isinstance(sample, list):
        return [move_to_device(x) for x in sample]
    elif isinstance(sample, dict):
        return {key: move_to_device(value) for key, value in sample.items()}
    else:
        return sample


def save_checkpoint(save_dir, model, optimizer, epoch, valid_loss):
    os.makedirs(save_dir, exist_ok=True)
    last_epoch = getattr(save_checkpoint, 'last_epoch', -1)
    prev_best = getattr(save_checkpoint, 'best_loss', float('inf'))
    save_checkpoint.last_epoch = max(last_epoch, epoch)
    save_checkpoint.best_loss = min(prev_best, valid_loss)

    state_dict = {
        'epoch': epoch,
        'val_loss': valid_loss,
        'best_loss': save_checkpoint.best_loss,
        'last_epoch': save_checkpoint.last_epoch,
        'model': model.state_dict(),
        'model_args': model.args,
        'model_kwargs': model.kwargs,
        'optimizer': optimizer.state_dict(),
    }

    if valid_loss < prev_best:
        torch.save(state_dict, f'{save_dir}/checkpoint_best.pt')
    if last_epoch < epoch:
        torch.save(state_dict, f'{save_dir}/checkpoint_last.pt')


def load_checkpoint(save_dir, model, optimizer):
    checkpoint_path = f'{save_dir}/checkpoint_last.pt'
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        save_checkpoint.best_loss = state_dict['best_loss']
        save_checkpoint.last_epoch = state_dict['last_epoch']
        print(f'Loaded checkpoint {checkpoint_path}')
        return state_dict


def init_logging(log_file):
    logging.basicConfig(level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='[%(asctime)s] %(message)s',
                        handlers=[StreamHandler(), FileHandler(log_file)])
