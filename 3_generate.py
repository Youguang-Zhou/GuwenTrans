import argparse

import torch
import yaml
from sentencepiece import SentencePieceProcessor
from torch import inference_mode

from src.Transformer import ModelConfig, Transformer
from src.utils import get_device, init_logger, load_checkpoint, load_config_file

DEVICE = get_device()


def get_args():
    parser = argparse.ArgumentParser('Generation')
    parser.add_argument('--input-text', default='我见青山多妩媚，青山见我应如是。')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--config-file', default='config.yml')
    return parser.parse_args()


@inference_mode()
def main(args):
    # parse args
    input_text = args.input_text
    results_dir = args.results_dir
    config_file = args.config_file

    # read configs
    config = load_config_file(config_file)
    seed = config['seed']
    generation_args = config['generation_args']

    # seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # init logger
    logger = init_logger(f'🌟 Generation (device: {DEVICE})', results_dir)
    logger.info(f'Load generation configuration:\n\n{yaml.dump(generation_args, sort_keys=False)}')

    # load tokenizer
    tokenizer = SentencePieceProcessor(model_file=f'{results_dir}/tokenizer.model')

    # load checkpoint
    checkpoint = load_checkpoint(results_dir, tag='best')

    # build model
    model_config = ModelConfig(**config['model_args'], vocab_size=tokenizer.GetPieceSize(), pad_id=tokenizer.pad_id())
    model = Transformer(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # [batch_size=1, seq_len]
    input_ids = torch.tensor(tokenizer.Encode(input_text)).unsqueeze(0).to(DEVICE)

    # start generate!
    output_ids = model.generate(input_ids, **generation_args, bos_id=tokenizer.bos_id(), eos_id=tokenizer.eos_id())

    # print results
    input_tokens = tokenizer.Decode(input_ids.tolist())[0].replace(',', '，')
    output_tokens = tokenizer.Decode(output_ids.tolist())[0].replace(',', '，')
    logger.info(f'Input:\n{input_tokens}')
    logger.info(f'Output:\n{output_tokens}')


if __name__ == '__main__':
    main(args=get_args())
