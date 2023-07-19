import argparse
import logging

import torch
import yaml
from sentencepiece import SentencePieceProcessor
from torch import inference_mode

import src.utils as utils
from src.Transformer import Transformer
from src.utils import get_device

DEVICE = get_device()


def get_args():
    parser = argparse.ArgumentParser('Generation')
    parser.add_argument('--inputs', default='我见青山多妩媚，料青山见我应如是。')
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--config-file', default='config.yml')
    parser.add_argument('--tokenizer-file', default='preprocessed/bpe_tokenizer.model')
    return parser.parse_args()


@inference_mode()
def main(args):
    # parse args
    inputs = args.inputs
    save_dir = args.save_dir
    config_file = args.config_file
    tokenizer_file = args.tokenizer_file

    # read configs
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        seed = config['seed']
        max_new_tokens = config['max_new_tokens']
        temperature = config['temperature']
        top_k = config['top_k']

    # seed
    torch.manual_seed(seed)

    # init logging
    utils.init_logging(f'🌟 Generation (device: {DEVICE})', save_dir)

    # load checkpoint
    state_dict = utils.load_checkpoint(save_dir, tag='best')

    # build tokenizer
    tokenizer = SentencePieceProcessor(tokenizer_file)

    # build model
    model = Transformer(state_dict['model_config'])
    model.load_state_dict(state_dict['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # start generate!
    src_inputs = torch.tensor([tokenizer.Encode(inputs)], device=DEVICE)

    output_ids = model.generate(
        src_inputs=src_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    output_sents = ''.join(tokenizer.Decode(output_ids.tolist()))
    logging.info(f'\nInput:\n{inputs}\n\nOutput:\n{output_sents}')


if __name__ == '__main__':
    main(args=get_args())
