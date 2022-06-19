import argparse
import logging
import os
import pickle
import re
import shutil
from glob import glob

import yaml
from sklearn.model_selection import train_test_split

import src.utils as utils
from src.dictionary import Dictionary

with open(f'config.yml', 'r') as f:
    config = yaml.safe_load(f)
    seed = config['seed']
    src_lang = config['src_lang']
    tgt_lang = config['tgt_lang']
    preprocess_params = config['preprocess_params']


def get_args():
    parser = argparse.ArgumentParser('Data Preprocessing')
    parser.add_argument('--data-raw', default='data_raw')
    parser.add_argument('--data-prep', default='data_prepared')
    parser.add_argument('--log-file', default='log.out', help='path to save logs')
    return parser.parse_args()


def main(args):
    data_raw = args.data_raw
    data_prep = args.data_prep
    log_file = args.log_file

    utils.init_logging(log_file)

    # 1. 合并所有数据
    # 下载数据后手动重命名文件夹名称为data。
    # 合并data目录下source里的所有内容，命名为all.old;
    # 合并data目录下target里的所有内容，命名为all.new。
    fname_src = f'all.{src_lang}'  # 文言文
    fname_tgt = f'all.{tgt_lang}'  # 白话文
    # 合并文言文
    with open(fname_src, 'wb') as out:
        for fname in sorted(glob('data/source/*')):
            if fname == fname_src:
                continue
            with open(fname, 'rb') as f:
                shutil.copyfileobj(f, out)
    # 合并白话文
    with open(fname_tgt, 'wb') as out:
        for fname in sorted(glob('data/target/*')):
            if fname == fname_tgt:
                continue
            with open(fname, 'rb') as f:
                shutil.copyfileobj(f, out)

    # 2. 分割数据集
    with open(fname_src, 'r') as f:
        X = f.read().split('\n')
    with open(fname_tgt, 'r') as f:
        y = f.read().split('\n')
    assert len(X) == len(y)
    logging.info(f'Total sentences: {len(y)}')

    split = 0.005
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split, random_state=seed)
    _, X_train_tiny, _, y_train_tiny = train_test_split(X_train, y_train, test_size=split, random_state=seed)
    logging.info(
        f'train: {len(X_train)}\t({len(X_train)/len(X):.3f}) <---> train_tiny: {len(X_train_tiny)}\t({len(X_train_tiny)/len(X):.3f})')
    logging.info(f'val:   {len(X_val)  }\t({len(X_val)  /len(X):.3f})')
    logging.info(f'test:  {len(X_test) }\t({len(X_test) /len(X):.3f})')
    assert len(X_train)+len(X_val)+len(X_test) == len(X)

    i = 0
    logging.info('Example:')
    logging.info(f'source: {X_train[i]}')
    logging.info(f'target: {y_train[i]}')

    # 3. 保存到本地
    os.makedirs(data_raw, exist_ok=True)
    for fname, arr in [(f'train.{src_lang}', X_train),
                       (f'train.{tgt_lang}', y_train),
                       (f'train_tiny.{src_lang}', X_train_tiny),
                       (f'train_tiny.{tgt_lang}', y_train_tiny),
                       (f'valid.{src_lang}', X_val),
                       (f'valid.{tgt_lang}', y_val),
                       (f'test.{src_lang}', X_test),
                       (f'test.{tgt_lang}', y_test)]:

        with open(f'{data_raw}/{fname}', 'w') as f:
            f.write('\n'.join(arr))
    shutil.move(fname_src, f'{data_raw}/{fname_src}')
    shutil.move(fname_tgt, f'{data_raw}/{fname_tgt}')

    # 4. 进一步处理
    os.makedirs(data_prep, exist_ok=True)
    src_dict = Dictionary.build_from_file(f'{data_raw}/train.{src_lang}', **preprocess_params)
    tgt_dict = Dictionary.build_from_file(f'{data_raw}/train.{tgt_lang}', **preprocess_params)
    src_dict.save(f'{data_prep}/dict.{src_lang}')
    tgt_dict.save(f'{data_prep}/dict.{tgt_lang}')
    logging.info(f'Built a source dictionary ({src_lang}) with {len(src_dict)} words')
    logging.info(f'Built a target dictionary ({tgt_lang}) with {len(tgt_dict)} words')

    for lang, dictionary in [(src_lang, src_dict),
                             (tgt_lang, tgt_dict)]:
        for fname in [f'train.{lang}',
                      f'train_tiny.{lang}',
                      f'valid.{lang}',
                      f'test.{lang}']:
            make_dataset(f'{data_raw}/{fname}', f'{data_prep}/{fname}', dictionary)


def make_dataset(input_fname, output_fname, dictionary):
    num_sent, num_unks, num_token = 0, 0, 0
    tokens_list = []
    with open(input_fname, 'r') as f:
        for line in f:
            tokens, num_unk = dictionary.string_to_tokenIds(line, return_unk_count=True)
            num_sent += 1
            num_unks += num_unk
            num_token += len(tokens)
            tokens_list.append(tokens)

    with open(output_fname, 'wb') as f:
        pickle.dump(tokens_list, f)
        logging.info(
            f'Built a dataset for {input_fname}: {num_sent} sentences, {num_token} tokens, {100*num_unks/num_token:.3f}% replaced by {dictionary.unk}')


if __name__ == '__main__':
    args = get_args()
    main(args)
