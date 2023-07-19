import argparse
import logging
import pickle

import torch
import yaml
from sentencepiece import SentencePieceProcessor
from torch import no_grad
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import src.utils as utils
from src.Dataset import Dataset
from src.ModelConfig import ModelConfig
from src.StatsLogger import StatsLogger
from src.Transformer import Transformer
from src.utils import get_device

DEVICE = get_device()


def get_args():
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--config-file', default='config.yml')
    parser.add_argument('--tokenizer-file', default='preprocessed/bpe_tokenizer.model')
    return parser.parse_args()


def main(args):
    # parse args
    save_dir = args.save_dir
    config_file = args.config_file
    tokenizer_file = args.tokenizer_file

    # read configs
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        seed = config['seed']
        save_interval = config['save_interval']
        batch_size = config['batch_size']
        max_epoch = config['max_epoch']
        lr = config['lr']
        enable_lr_decay = config['enable_lr_decay']

    # seed
    torch.manual_seed(seed)

    # init logging
    utils.init_logging('🌟 Training', save_dir)
    logging.info(f'Load configuration file {config_file}:\n\n{yaml.dump(config, sort_keys=False)}')

    # init tensorboard
    tensorboard = SummaryWriter(log_dir=f'{save_dir}/runs')

    # load tokenizer
    tokenizer = SentencePieceProcessor(tokenizer_file)

    # load dataset
    def load_dataset(split):
        with open(f'preprocessed/{split}_ids.src', 'rb') as f_src, open(f'preprocessed/{split}_ids.tgt', 'rb') as f_tgt:
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
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=train_dataset.collater)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True, collate_fn=valid_dataset.collater)

    # build model
    model_config = ModelConfig(
        **config['model_params'],
        bos_token_id=tokenizer.bos_id(),
        eos_token_id=tokenizer.eos_id(),
        pad_token_id=tokenizer.pad_id(),
    )
    model = Transformer(model_config)
    model.to(DEVICE)
    logging.info(f'Built a model with {model.num_parameters()} parameters (device: {DEVICE})')

    # build optimizer
    optimizer = AdamW(model.parameters(), lr)

    # build learning rate scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0.1 * lr)

    # build stats logger for logging use
    stats_logger = StatsLogger()

    # load last checkpoint if one exists
    state_dict = utils.load_checkpoint(save_dir, tag='last')
    if state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler_state_dict'])
        curr_epoch = state_dict['curr_epoch'] + 1
        best_valid_loss = state_dict['best_valid_loss']
        if curr_epoch == max_epoch:
            logging.info(f'MAX_EPOCH ({max_epoch}) reached.')
            exit()
    else:
        curr_epoch = 1
        best_valid_loss = float('inf')

    # tracking best validation loss
    best_valid_loss = float('inf')

    # start training!
    for epoch in range(curr_epoch, max_epoch + 1):
        # switch train mode
        model.train()

        # zero the stats for progress bar
        stats_logger.zero_stats()

        # set up progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)

        for sample in progress_bar:
            # src_inputs:  <--- [batch_size, seq_len]
            src_inputs = sample['src_inputs'].to(DEVICE)
            # tgt_inputs:  <--- [batch_size, seq_len]
            tgt_inputs = sample['tgt_inputs'].to(DEVICE)
            # tgt_outputs: <--- [batch_size, seq_len]
            tgt_outputs = sample['tgt_outputs'].to(DEVICE)

            # zero the gradient buffers
            optimizer.zero_grad()

            # forward passing
            out = model(src_inputs, tgt_inputs, tgt_outputs)

            # loss
            loss = out.loss
            loss.backward()

            # update optimizer
            optimizer.step()

            # update statistics
            stats_logger.step(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

            # update progress bar
            progress_bar.set_postfix(dict(stats_logger))

        # update learning rate scheduler
        if enable_lr_decay:
            lr_scheduler.step()

        # calculate average training loss for this epoch
        train_loss = stats_logger.train_loss
        logging.info(f'Epoch {epoch:{len(str(max_epoch))}d} | {str(stats_logger)}')

        # calculate average validation loss
        valid_loss = validate(model, valid_loader)
        log_str = f'Epoch {epoch:{len(str(max_epoch))}d} | valid_loss {valid_loss:.4f}'
        if valid_loss < best_valid_loss:
            log_str += ' <------------------------------ best validation loss'
        logging.info(log_str)

        # save checkpoint
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            utils.save_checkpoint(save_dir, 'best', model, optimizer, lr_scheduler, epoch, best_valid_loss)
        if save_interval != 0 and epoch % save_interval == 0:
            utils.save_checkpoint(save_dir, f'epoch-{epoch}', model, optimizer, lr_scheduler, epoch, best_valid_loss)

        # update tensorboard
        tensorboard.add_scalars(save_dir, {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch)

    # finish
    tensorboard.close()
    logging.info('Training finished!')


@no_grad()
def validate(model, valid_loader):
    model.eval()
    valid_losses = torch.zeros(len(valid_loader))
    for i, sample in enumerate(tqdm(valid_loader, desc=f'Validating...', leave=False)):
        src_inputs = sample['src_inputs'].to(DEVICE)
        tgt_inputs = sample['tgt_inputs'].to(DEVICE)
        tgt_outputs = sample['tgt_outputs'].to(DEVICE)
        out = model(src_inputs, tgt_inputs, tgt_outputs)
        valid_losses[i] = out.loss.item()
    return valid_losses.mean()


if __name__ == '__main__':
    main(args=get_args())
