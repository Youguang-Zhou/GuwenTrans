import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_dict, tgt_dict):
        self.src_dict, self.tgt_dict = src_dict, tgt_dict
        with open(src_file, 'rb') as f:
            self.src_dataset = pickle.load(f)
            self.src_sizes = np.array([len(sent) for sent in self.src_dataset])
        with open(tgt_file, 'rb') as f:
            self.tgt_dataset = pickle.load(f)
            self.tgt_sizes = np.array([len(sent) for sent in self.tgt_dataset])

    def __getitem__(self, idx):
        return {
            'source': self.src_dataset[idx].long(),
            'target': self.tgt_dataset[idx].long(),
        }

    def __len__(self):
        return len(self.src_dataset)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        def merge(tokens_list, only_sos=False, only_eos=False):
            assert self.src_dict.pad_id == self.tgt_dict.pad_id
            pad_id = self.src_dict.pad_id
            max_length = max(tokens.size(0) for tokens in tokens_list)
            if only_sos:
                max_length -= 1
            if only_eos:
                max_length -= 1
            result = tokens_list[0].new(len(tokens_list), max_length).fill_(pad_id)
            for i, v in enumerate(tokens_list):
                if only_sos:
                    result[i, :len(v)-1] = v[:-1]
                elif only_eos:
                    result[i, :len(v)-1] = v[1:]
                else:
                    result[i, :len(v)].copy_(v)
            return result

        src_tokens = merge([s['source'] for s in samples])
        tgt_inputs = merge([s['target'] for s in samples], only_sos=True)  # for training use
        tgt_tokens = merge([s['target'] for s in samples], only_eos=True)  # for criterion use
        src_lengths = torch.LongTensor([len(s['source']) for s in samples])

        return {
            'src_tokens': src_tokens,
            'tgt_inputs': tgt_inputs,
            'tgt_tokens': tgt_tokens,
            'src_lengths': src_lengths,
            'num_tokens': sum(len(s['target']) for s in samples),
        }
