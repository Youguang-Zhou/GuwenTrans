import argparse
import logging
import os
import pickle
from pathlib import Path

import yaml
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

import src.utils as utils


def get_args():
    parser = argparse.ArgumentParser('Preprocessing')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--save-dir', default='preprocessed')
    parser.add_argument('--config-file', default='config.yml')
    return parser.parse_args()


def main(args):
    # parse args
    data_dir = args.data_dir
    save_dir = args.save_dir
    config_file = args.config_file

    # read configs
    with open(config_file) as f:
        config = yaml.safe_load(f)
        vocab_size = config['model_params']['vocab_size']
        max_sequence_length = config['model_params']['max_sequence_length']

    # init logging
    utils.init_logging('🌟 Preprocessing', save_dir)

    # create folder
    os.makedirs(save_dir, exist_ok=True)

    # read data
    with open(f'{data_dir}/src.txt') as f:
        src_data = f.read().splitlines()
    with open(f'{data_dir}/tgt.txt') as f:
        tgt_data = f.read().splitlines()

    assert len(src_data) == len(tgt_data)

    # filter those sentences that beyond max_sequence_length
    _temp_src = []
    _temp_tgt = []
    for i in range(len(src_data)):
        if len(src_data[i]) <= max_sequence_length and len(tgt_data[i]) <= max_sequence_length:
            _temp_src.append(src_data[i])
            _temp_tgt.append(tgt_data[i])
    src_data = _temp_src
    tgt_data = _temp_tgt

    assert len(src_data) == len(tgt_data)

    # merge all txt data together to train tokenizer
    with open(f'{save_dir}/all.txt', 'w') as f:
        f.writelines([s + '\n' for s in src_data])
        f.writelines([t + '\n' for t in tgt_data])

    # define name and path of tokenizer
    tokenizer_name = 'bpe_tokenizer'
    tokenizer_path = Path(save_dir) / f'{tokenizer_name}.model'

    # train tokenizer
    if os.path.isfile(tokenizer_path):
        logging.info(f'【{tokenizer_path}】already exists.')
    else:
        SentencePieceTrainer.train(
            f'--input={save_dir}/all.txt                 \
            --model_prefix={save_dir}/{tokenizer_name}   \
            --vocab_size={vocab_size}                    \
            --model_type=bpe                             \
            --pad_id=3'
        )
        os.remove(Path(save_dir) / 'all.txt')

    # tokenization
    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

    # encode with same tokenizer
    src_token_ids = tokenizer.Encode(src_data)
    tgt_token_ids = tokenizer.Encode(tgt_data)

    # split dataset
    n = len(src_token_ids)

    src_train_ids = src_token_ids[: int(n * 0.9)]
    tgt_train_ids = tgt_token_ids[: int(n * 0.9)]

    with open(Path(save_dir) / 'train_ids.src', 'wb') as f:
        pickle.dump(src_train_ids, f)
    with open(Path(save_dir) / 'train_ids.tgt', 'wb') as f:
        pickle.dump(tgt_train_ids, f)

    src_valid_ids = src_token_ids[int(n * 0.9) :]
    tgt_valid_ids = tgt_token_ids[int(n * 0.9) :]

    with open(Path(save_dir) / 'valid_ids.src', 'wb') as f:
        pickle.dump(src_valid_ids, f)
    with open(Path(save_dir) / 'valid_ids.tgt', 'wb') as f:
        pickle.dump(tgt_valid_ids, f)

    logging.info(f'total sentence pairs: {n}')
    logging.info(f'vocab size: {tokenizer.GetPieceSize()}')
    logging.info(f'train data has {len(src_train_ids)} sentences')
    logging.info(f'\tmax_src_len: {max([len(sent) for sent in src_train_ids])}')
    logging.info(f'\tmax_tgt_len: {max([len(sent) for sent in tgt_train_ids])}')
    logging.info(f'valid data has {len(src_valid_ids)} sentences')
    logging.info(f'\tmax_src_len: {max([len(sent) for sent in src_valid_ids])}')
    logging.info(f'\tmax_tgt_len: {max([len(sent) for sent in tgt_valid_ids])}')
    logging.info('Note: sentencepiece added token ▁ at the beginning of sentence')


if __name__ == '__main__':
    main(args=get_args())
