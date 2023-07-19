import logging
import os
from logging import FileHandler, StreamHandler

import torch


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def init_logging(start_string, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='[%(asctime)s] %(message)s',
        handlers=[StreamHandler(), FileHandler(os.path.join(save_dir, 'log.txt'))],
    )
    logging.info(start_string)


def save_checkpoint(save_dir, tag, model, optimizer, lr_scheduler, epoch, best_valid_loss):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'model_config': model.config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'curr_epoch': epoch,
        'best_valid_loss': best_valid_loss,
    }
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint.{tag}.pt'))
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.last.pt'))


def load_checkpoint(save_dir, tag):
    checkpoint_last_path = os.path.join(save_dir, f'checkpoint.{tag}.pt')
    if os.path.isfile(checkpoint_last_path):
        return torch.load(checkpoint_last_path)
    else:
        return None
