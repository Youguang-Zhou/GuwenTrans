import argparse
import logging

import numpy as np
import torch
import yaml
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from tqdm import tqdm

import src.utils as utils
from src.dataset import Seq2SeqDataset
from src.transformer import Transformer

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)
    seed = config['seed']
    src_lang = config['src_lang']
    tgt_lang = config['tgt_lang']
    batch_size = config['hyper_params']['batch_size']
    max_len = config['hyper_params']['max_len']


def get_args():
    parser = argparse.ArgumentParser('GuwenTrans')
    parser.add_argument('--data', default='data_prepared')
    parser.add_argument('--checkpoint-path', default='checkpoint_best.pt')
    parser.add_argument('--translate-output', default='model_translations.txt')
    return parser.parse_args()


def main(args):
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # arguments
    data = args.data
    checkpoint_path = args.checkpoint_path
    translate_output = args.translate_output

    # load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # build model and criterion
    model_args = state_dict['model_args']
    model_kwargs = state_dict['model_kwargs']
    model = Transformer(**model_args, **model_kwargs)
    model.to(utils.device)
    model.eval()
    model.load_state_dict(state_dict['model'])
    logging.info(f'Loaded a model from checkpoint {checkpoint_path}')

    # load dataset
    src_dict = model_args['src_dict']
    tgt_dict = model_args['tgt_dict']
    test_dataset = Seq2SeqDataset(f'{data}/test.{src_lang}', f'{data}/test.{tgt_lang}', src_dict, tgt_dict)
    test_loader = DataLoader(test_dataset,
                             num_workers=8,
                             collate_fn=test_dataset.collater,
                             batch_sampler=BatchSampler(SequentialSampler(test_dataset), batch_size, drop_last=False))

    # iterate over the test set
    output_sents = []
    for sample in tqdm(test_loader, desc='| Generation', leave=False):
        src_tokens = sample['src_tokens'].to(utils.device)
        go_slice = torch.ones(src_tokens.shape[0], 1).fill_(tgt_dict.sos_id).type_as(src_tokens)
        prev_words = go_slice  # [batch_size, 1]
        next_words = None

        for _ in range(max_len):
            with torch.no_grad():
                # compute the decoder output by repeatedly feeding it the decoded sentence prefix
                # [batch_size, 1, vocab_size]
                output = model(src_tokens, prev_words)
            # suppress <UNK>s
            _, next_candidates = torch.topk(output, 2, dim=-1)
            best_candidates = next_candidates[:, :, 0]
            backoff_candidates = next_candidates[:, :, 1]
            next_words = torch.where(best_candidates == tgt_dict.unk_id, backoff_candidates, best_candidates)
            prev_words = torch.cat([go_slice, next_words], dim=1)

        # segment into sentences
        decoded_batch = next_words.cpu().numpy()
        output_tokens = [decoded_batch[row, :] for row in range(decoded_batch.shape[0])]

        # remove padding
        temp = list()
        for tokens in output_tokens:
            first_eos = np.where(tokens == tgt_dict.eos_id)[0]
            if len(first_eos) > 0:
                temp.append(tokens[:first_eos[0]])
            else:
                temp.append(tokens)
        output_tokens = temp

        # convert tokenIds into string
        output_sents.extend([tgt_dict.tokenIds_to_string(tokens) for tokens in output_tokens])

    with open(translate_output, 'w') as f:
        for sent in output_sents:
            f.write(f'{sent}\n')

    logging.info(f'Output {len(output_sents)} translations to {translate_output}')


if __name__ == '__main__':
    args = get_args()
    main(args)
