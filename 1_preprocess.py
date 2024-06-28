import argparse
import os
import pickle
from pathlib import Path

from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from tqdm import tqdm

from src.utils import init_logger, load_config_file


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
    config = load_config_file(config_file)
    vocab_size = config['vocab_size']
    max_sequence_length = config['model_args']['max_sequence_length']

    # init logger
    logger = init_logger('🌟 Preprocessing', save_dir)

    # read data
    src_file = os.path.join(data_dir, 'src.txt')
    tgt_file = os.path.join(data_dir, 'tgt.txt')
    with open(src_file) as f:
        src_data = f.read().splitlines()
    with open(tgt_file) as f:
        tgt_data = f.read().splitlines()

    assert len(src_data) == len(tgt_data)

    # define name and path of tokenizer
    tokenizer_name = 'tokenizer'
    tokenizer_path = os.path.join(save_dir, f'{tokenizer_name}.model')

    # check if tokenizer exists
    if os.path.isfile(tokenizer_path):
        logger.info(f'【{tokenizer_path}】already exists.')
    else:
        SentencePieceTrainer.train(
            input=','.join([src_file, tgt_file]),
            model_prefix=f'{save_dir}/{tokenizer_name}',
            vocab_size=vocab_size,
            pad_id=3,
            bos_piece='<bos>',
            eos_piece='<eos>',
        )

    # tokenization
    tokenizer = SentencePieceProcessor(model_file=tokenizer_path)

    # preprocess sentences based on max_sequence_length
    src_token_ids = []
    tgt_token_ids = []
    total_origin_num_tokens = 0
    for i in tqdm(range(len(src_data)), desc='Tokenizing', leave=False):
        src_sent = src_data[i]
        tgt_sent = tgt_data[i]
        src_ids = tokenizer.Encode(src_sent)
        tgt_ids = tokenizer.Encode(tgt_sent)
        total_origin_num_tokens += len(src_ids) + len(tgt_ids)
        if len(src_ids) < max_sequence_length and len(tgt_ids) < max_sequence_length:
            # 原文和译文都小于 max_sequence_length
            src_token_ids.append(src_ids)
            tgt_token_ids.append(tgt_ids)
        else:
            # 通常来说，原文比译文短，如果比译文长那就是大概率对齐出错了
            if len(src_sent) > len(tgt_sent):
                continue
            # 按句号分割子句子
            src_sub_sents = [s + '。' for s in src_sent.split('。') if s != '']
            tgt_sub_sents = [t + '。' for t in tgt_sent.split('。') if t != '']
            # 填充 src 和 tgt 至最大长度 max_sequence_length
            idx = 0
            num_chunks = min(len(src_ids), len(tgt_ids)) // max_sequence_length
            for _ in range(num_chunks):
                src = ''
                tgt = ''
                src_ids = []
                tgt_ids = []
                while idx < len(src_sub_sents) and idx < len(tgt_sub_sents):
                    src += src_sub_sents[idx]
                    tgt += tgt_sub_sents[idx]
                    next_src_ids = tokenizer.Encode(src)
                    next_tgt_ids = tokenizer.Encode(tgt)
                    idx += 1
                    # 必须同时满足小于 max_sequence_length
                    if len(next_src_ids) < max_sequence_length and len(next_tgt_ids) < max_sequence_length:
                        src_ids = next_src_ids
                        tgt_ids = next_tgt_ids
                    else:
                        break
                if len(src_ids) == 0 or len(tgt_ids) == 0:
                    continue
                src_token_ids.append(src_ids)
                tgt_token_ids.append(tgt_ids)

    assert len(src_token_ids) == len(tgt_token_ids)

    # split dataset
    n = len(src_token_ids)
    total_num_src_tokens = sum([len(s) for s in src_token_ids])
    total_num_tgt_tokens = sum([len(t) for t in tgt_token_ids])
    token_utilization = (total_num_src_tokens + total_num_tgt_tokens) / total_origin_num_tokens

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

    logger.info(f'Vocab size: {tokenizer.GetPieceSize():,}')
    logger.info(f'Total sentence pairs: {n:,}')
    logger.info(f'Total src_tokens: {total_num_src_tokens:,}')
    logger.info(f'Total tgt_tokens: {total_num_tgt_tokens:,}')
    logger.info(f'Token utilization: {token_utilization*100:.2f}%')
    logger.info(f'Train data has {len(src_train_ids):,} sentences')
    logger.info(f'\tmax_src_len: {max([len(sent) for sent in src_train_ids])}')
    logger.info(f'\tmax_tgt_len: {max([len(sent) for sent in tgt_train_ids])}')
    logger.info(f'Valid data has {len(src_valid_ids):,} sentences')
    logger.info(f'\tmax_src_len: {max([len(sent) for sent in src_valid_ids])}')
    logger.info(f'\tmax_tgt_len: {max([len(sent) for sent in tgt_valid_ids])}')


if __name__ == '__main__':
    main(args=get_args())
