import argparse
import logging
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from tqdm import tqdm

import src.utils as utils
from src.dataset import Seq2SeqDataset
from src.dictionary import Dictionary
from src.transformer import Transformer

train_loss_arr = []
valid_loss_arr = []

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)
    seed = config['seed']
    src_lang = config['src_lang']
    tgt_lang = config['tgt_lang']
    hyper_params = config['hyper_params']
    model_params = config['model_params']

    lr = hyper_params['lr']
    batch_size = hyper_params['batch_size']
    max_tokens = hyper_params['max_tokens']
    max_epoch = hyper_params['max_epoch']
    clip_norm = hyper_params['clip_norm']
    patience = hyper_params['patience']


def get_args():
    parser = argparse.ArgumentParser('GuwenTrans')
    parser.add_argument('--data', default='data_prepared')
    parser.add_argument('--train-on-tiny', action='store_true')
    parser.add_argument('--log-file', default='log.out', help='path to save logs')
    parser.add_argument('--save-dir', default='results', help='path to save checkpoints')
    return parser.parse_args()


##########################
# MAIN TRAINING ENTRANCE #
##########################
def main(args):
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # arguments
    data = args.data
    log_file = args.log_file
    save_dir = args.save_dir
    train_on_tiny = args.train_on_tiny

    utils.init_logging(log_file)
    logging.info(f'Current device: {utils.device}')
    logging.info(f'Hyper Parameters:\n{yaml.dump(hyper_params)}')
    logging.info(f'Model Parameters:\n{yaml.dump(model_params)}')

    # load dictionaries
    src_dict = Dictionary.load(f'{data}/dict.{src_lang}')
    tgt_dict = Dictionary.load(f'{data}/dict.{tgt_lang}')
    logging.info(f'Loaded a source dictionary ({src_lang}) with {len(src_dict)} words')
    logging.info(f'Loaded a target dictionary ({tgt_lang}) with {len(tgt_dict)} words')

    # load datasets
    def load_data(split):
        return Seq2SeqDataset(f'{data}/{split}.{src_lang}', f'{data}/{split}.{tgt_lang}', src_dict, tgt_dict)
    train_dataset = load_data(split='train') if not train_on_tiny else load_data(split='train_tiny')
    valid_dataset = load_data(split='valid')

    # build model and optimization criterion
    assert src_dict.pad_id == tgt_dict.pad_id
    model = Transformer(len(src_dict), len(tgt_dict), src_dict.pad_id, **model_params)
    model.to(utils.device)
    criterion = nn.CrossEntropyLoss(ignore_index=src_dict.pad_id, reduction='sum')
    logging.info(f'Built a model with {sum(p.numel() for p in model.parameters())} parameters')

    # instantiate optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # load last checkpoint if one exists
    state_dict = utils.load_checkpoint(save_dir, model, optimizer)
    last_epoch = state_dict['last_epoch'] if state_dict is not None else -1

    # track validation performance for early stopping
    bad_epochs = 0
    best_validate = float('inf')

    for epoch in range(last_epoch+1, max_epoch):
        train_loader = DataLoader(train_dataset,
                                  num_workers=1,
                                  collate_fn=train_dataset.collater,
                                  batch_sampler=BatchSampler(SequentialSampler(train_dataset), batch_size, drop_last=False))
        stats = OrderedDict()
        stats['train_loss'] = 0
        stats['batch_size'] = 0
        stats['lr'] = 0
        stats['num_tokens'] = 0
        # display progress
        progress_bar = tqdm(train_loader, desc=f'| Epoch {epoch:03d}', leave=False)

        model.train()
        for i, sample in enumerate(progress_bar):
            if len(sample) == 0:
                continue

            sample = utils.move_to_device(sample)

            src_tokens = sample['src_tokens']
            src_lengths = sample['src_lengths']
            tgt_inputs = sample['tgt_inputs']
            tgt_tokens = sample['tgt_tokens']  # ground truth
            num_tokens = sample['num_tokens']

            output = model(src_tokens, tgt_inputs)
            loss = criterion(output.view(-1, output.size(-1)), tgt_tokens.view(-1)) / len(src_lengths)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            # update statistics for progress bar
            stats['lr'] += optimizer.param_groups[0]['lr']
            stats['train_loss'] += loss.item() * len(src_lengths) / num_tokens
            stats['num_tokens'] += num_tokens / len(src_tokens)
            stats['batch_size'] += batch_size
            progress_bar.set_postfix({k: f'{v/(i+1):.4g}' for k, v in stats.items()})

        train_loss_arr.append(stats['train_loss'] / len(progress_bar))
        logging.info(f"Epoch {epoch:03d}: {' | '.join(k + f' {v / len(progress_bar):.4g}' for k, v in stats.items())}")

        # calculate validation loss
        valid_perplexity = validate(model, criterion, valid_dataset, epoch)

        # save checkpoints
        utils.save_checkpoint(save_dir, model, optimizer, epoch, valid_perplexity)

        # check whether to terminate training
        if valid_perplexity < best_validate:
            best_validate = valid_perplexity
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            logging.info(f'No validation set improvements observed for {patience:d} epochs. Early stop!')
            break


def validate(model, criterion, valid_dataset, epoch):
    stats = OrderedDict()
    stats['valid_loss'] = 0
    stats['batch_size'] = batch_size
    stats['lr'] = lr
    stats['num_tokens'] = 0
    valid_loader = DataLoader(valid_dataset, num_workers=1, collate_fn=valid_dataset.collater,
                              batch_sampler=BatchSampler(SequentialSampler(valid_dataset), batch_size, drop_last=False))
    model.eval()
    for sample in valid_loader:
        if len(sample) == 0:
            continue
        with torch.no_grad():
            sample = utils.move_to_device(sample)
            src_tokens = sample['src_tokens']
            tgt_inputs = sample['tgt_inputs']
            tgt_tokens = sample['tgt_tokens']
            output = model(src_tokens, tgt_inputs)
            loss = criterion(output.view(-1, output.size(-1)), tgt_tokens.view(-1))
        stats['valid_loss'] += loss.item()
        stats['num_tokens'] += sample['num_tokens']
    valid_loss = stats['valid_loss'] / stats['num_tokens']
    valid_loss_arr.append(valid_loss)
    perplexity = np.exp(valid_loss)
    stats['valid_loss'] = valid_loss
    stats['num_tokens'] = stats['num_tokens'] / stats['batch_size'] / len(valid_loader)
    stats['valid_perplexity'] = perplexity
    logging.info(f"Epoch {epoch:03d}: {' | '.join(k + f' {v:.4g}' for k, v in stats.items())}")
    return perplexity


if __name__ == '__main__':
    args = get_args()
    main(args)
    with open('results/loss.pkl', 'wb') as f:
        pickle.dump({
            'train_loss_arr': train_loss_arr,
            'valid_loss_arr': valid_loss_arr,
        }, f)
