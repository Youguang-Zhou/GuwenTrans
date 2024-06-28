'''
To run on a single GPU:
python 2_train.py

To run with DDP on 8 GPUs on 1 node:
torchrun --standalone --nproc_per_node=8 2_train.py
'''

import argparse
import os
import pickle
import shutil
import time
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from sentencepiece import SentencePieceProcessor
from torch import no_grad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.Dataset import Dataset
from src.StatsLogger import StatsLogger
from src.Transformer import ModelConfig, Transformer
from src.utils import get_device, init_logger, load_checkpoint, load_config_file, save_checkpoint


def get_args():
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--prep-dir', default='preprocessed')
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--config-file', default='config.yml')
    return parser.parse_args()


def main(args):
    # parse args
    prep_dir = args.prep_dir
    save_dir = args.save_dir
    config_file = args.config_file

    # read configs
    config = load_config_file(config_file)
    seed = config['seed']
    save_interval = config['save_interval']
    total_batch_size = config['total_batch_size']
    batch_size = config['batch_size']
    max_epoch = config['max_epoch']
    optimizer_args = config['optimizer_args']
    scheduler_args = config['scheduler_args']

    # seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # init ddp
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        dist.init_process_group(backend='nccl')
        ddp_rank = dist.get_rank()
        ddp_world_size = dist.get_world_size()
        device_id = ddp_rank % torch.cuda.device_count()
        master_process = ddp_rank == 0
        DEVICE = f'cuda:{device_id}'
    else:
        ddp_rank = 0
        ddp_world_size = 1
        device_id = 0
        master_process = True
        DEVICE = get_device()

    # init logger
    logger = init_logger('🌟 Training', save_dir, master_process)
    logger.info(f'Load configuration file {config_file}:\n\n{yaml.dump(config, sort_keys=False)}')

    # init tensorboard
    if master_process:
        tensorboard = SummaryWriter(log_dir=f'{save_dir}/runs')

    # load tokenizer and copy to results folder
    tokenizer_path = f'{prep_dir}/tokenizer.model'
    tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
    if master_process:
        shutil.copy(tokenizer_path, save_dir)

    # build model
    model_config = ModelConfig(**config['model_args'], vocab_size=tokenizer.GetPieceSize(), pad_id=tokenizer.pad_id())
    model = Transformer(model_config)
    model.to(DEVICE)
    logger.info(f'Built a model with {sum([p.numel() for p in model.parameters()]):,} parameters (device: {DEVICE})')

    # build criterion
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())

    # build optimizer
    optimizer = AdamW(model.parameters(), **optimizer_args, fused=torch.cuda.is_available())

    # build scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, **scheduler_args)

    # build stats logger
    stats_logger = StatsLogger(logger)

    # load dataset
    def load_dataset(split: Literal['train', 'valid']):
        with open(f'{prep_dir}/{split}_ids.src', 'rb') as f_src, open(f'{prep_dir}/{split}_ids.tgt', 'rb') as f_tgt:
            return Dataset(
                src_token_ids=pickle.load(f_src),
                tgt_token_ids=pickle.load(f_tgt),
                bos_id=tokenizer.bos_id(),
                eos_id=tokenizer.eos_id(),
                pad_id=tokenizer.pad_id(),
            )

    train_dataset = load_dataset(split='train')
    valid_dataset = load_dataset(split='valid')

    # build data loader
    if ddp:
        train_loader = DataLoader(
            train_dataset,
            batch_size,
            sampler=DistributedSampler(train_dataset),
            collate_fn=train_dataset.collater,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size,
            sampler=DistributedSampler(valid_dataset),
            collate_fn=valid_dataset.collater,
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=train_dataset.collater)
        valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True, collate_fn=valid_dataset.collater)
    num_iters = len(train_loader)

    # gradient accumulation
    assert total_batch_size % (batch_size * model_config.max_sequence_length * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (batch_size * model_config.max_sequence_length * ddp_world_size)
    logger.info(f'Gradient accumulation step: {grad_accum_steps}')

    # tracking epoch and valid loss
    curr_epoch = 1
    best_valid_loss = float('inf')

    # load last checkpoint if one exists
    checkpoint = load_checkpoint(save_dir, tag='last')
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        curr_epoch = checkpoint['curr_epoch'] + 1
        best_valid_loss = checkpoint['best_valid_loss']
        if curr_epoch == max_epoch + 1:
            logger.info(f'MAX_EPOCH ({max_epoch}) reached.')
            exit()

    if ddp:
        model = DDP(model, device_ids=[device_id])
    raw_model = model.module if ddp else model

    # start training!
    for epoch in range(curr_epoch, max_epoch + 1):
        # timer
        t0 = time.time()

        # switch train mode
        model.train()

        # make shuffling work properly across multiple epochs
        if ddp:
            train_loader.sampler.set_epoch(epoch)

        # zero the stats for progress bar
        stats_logger.zero_stats(curr_epoch=epoch)

        # set up progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)

        for i, sample in enumerate(progress_bar):
            # [batch_size, seq_len]
            src_inputs = sample['src_inputs'].to(DEVICE)
            tgt_inputs = sample['tgt_inputs'].to(DEVICE)
            tgt_outputs = sample['tgt_outputs'].to(DEVICE)

            # count number of pad tokens in batch
            src_num_pad = torch.sum(src_inputs == tokenizer.pad_id())
            tgt_num_pad = torch.sum(tgt_inputs == tokenizer.pad_id())

            # forward passing
            out = model(src_inputs, tgt_inputs)

            # cross entropy loss and backward propagation (with normalization)
            loss = criterion(out.view(-1, out.size(-1)), tgt_outputs.view(-1))
            loss = loss / grad_accum_steps
            loss.backward()

            # gradient accumulation
            if (i + 1) % grad_accum_steps == 0:
                # clip gradient
                grad_norm = clip_grad_norm_(model.parameters(), 1.0)

                # update optimizer
                optimizer.step()
                optimizer.zero_grad()

                # update scheduler
                scheduler.step(epoch + i / num_iters)

                # update statistics
                stats_logger.step(grad_norm=grad_norm)

            # update statistics
            train_loss = loss * grad_accum_steps
            pad_percent = (src_num_pad + tgt_num_pad) / (batch_size * model_config.max_sequence_length * 2)
            if ddp:
                dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(pad_percent, op=dist.ReduceOp.AVG)
            stats_logger.step(train_loss=train_loss.item(), lr=scheduler.get_last_lr()[0], pad_percent=pad_percent)

            # update progress bar
            progress_bar.set_postfix(dict(stats_logger))

        # wait for the GPU to finish work
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # print out training statistics
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = batch_size * model_config.max_sequence_length * num_iters * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        stats_logger.step(
            train_loss=stats_logger.train_loss_sum / num_iters,
            lr=scheduler.get_last_lr()[0],
            grad_norm=stats_logger.grad_norm_sum / num_iters * grad_accum_steps,
            duration=dt,
            tokens_per_sec=tokens_per_sec,
        )
        stats_logger.log_train()

        # print out validation statistics
        valid_loss = validate(valid_loader, model, criterion, DEVICE)
        if ddp:
            dist.all_reduce(valid_loss, op=dist.ReduceOp.AVG)
        stats_logger.step(valid_loss=valid_loss.item())
        stats_logger.log_valid(best_valid_loss=best_valid_loss)

        # save checkpoint
        if master_process:
            ckpt_to_save = {
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'curr_epoch': epoch,
                'best_valid_loss': best_valid_loss,
            }
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                ckpt_to_save['best_valid_loss'] = best_valid_loss
                save_checkpoint(save_dir, 'best', ckpt_to_save)
            if save_interval == 0:
                save_checkpoint(save_dir, 'last', ckpt_to_save)
            elif epoch == 1 or epoch % save_interval == 0 or epoch == max_epoch:
                save_checkpoint(save_dir, f'epoch-{epoch}', ckpt_to_save)

        # update tensorboard
        if master_process:
            tensorboard.add_scalars(save_dir, {'train_loss': stats_logger.train_loss, 'valid_loss': valid_loss}, epoch)

    # finish
    if master_process:
        tensorboard.close()
    if ddp:
        dist.destroy_process_group()
    logger.info('Training finished!')


@no_grad()
def validate(valid_loader, model, criterion, device):
    model.eval()
    valid_loss = 0
    for sample in tqdm(valid_loader, desc=f'Validating...', leave=False):
        src_inputs = sample['src_inputs'].to(device)
        tgt_inputs = sample['tgt_inputs'].to(device)
        tgt_outputs = sample['tgt_outputs'].to(device)
        out = model(src_inputs, tgt_inputs)
        loss = criterion(out.view(-1, out.size(-1)), tgt_outputs.view(-1))
        valid_loss += loss
    return valid_loss / len(valid_loader)


if __name__ == '__main__':
    main(args=get_args())
