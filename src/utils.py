import logging
import os
from logging import FileHandler, StreamHandler

import torch
import yaml


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def load_config_file(config_file):
    with open(config_file) as f:
        return yaml.safe_load(f)


def init_logger(init_string, save_dir, master_process=True):
    if not master_process:
        _logger = logging.getLogger(__name__)
        _logger.propagate = False
        return _logger
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='[%(asctime)s] %(message)s',
        handlers=[StreamHandler(), FileHandler(os.path.join(save_dir, 'log.txt'))],
    )
    logger = logging.getLogger()
    logger.info(init_string)
    return logger


def save_checkpoint(save_dir, tag, ckpt_to_save):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(ckpt_to_save, os.path.join(save_dir, f'checkpoint.{tag}.pt'))


def load_checkpoint(save_dir, tag):
    checkpoint_path = os.path.join(save_dir, f'checkpoint.{tag}.pt')
    if os.path.isfile(checkpoint_path):
        return torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        return None
