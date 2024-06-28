from typing import List

import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(
        self,
        src_token_ids: List[List[int]],
        tgt_token_ids: List[List[int]],
        bos_id: int,
        eos_id: int,
        pad_id: int,
    ):
        '''
        src_token_ids: list of source sentences where each sentence is represented by a list of token ids
        tgt_token_ids: list of target sentences where each sentence is represented by a list of token ids
        bos_id:        token id of begin of sentence
        eos_id:        token id of end of sentence
        pad_id:        token id of padding token
        '''
        self.src_token_ids = src_token_ids
        self.tgt_token_ids = tgt_token_ids
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        assert len(src_token_ids) == len(tgt_token_ids)

    def __getitem__(self, idx):
        return {
            'src': torch.tensor(self.src_token_ids[idx]),
            'tgt': torch.tensor([self.bos_id, *self.tgt_token_ids[idx], self.eos_id]),
        }

    def __len__(self):
        return len(self.src_token_ids)

    def collater(self, samples):
        '''
        1. keep only <s>  for tgt_inpnuts
        2. keep only </s> for tgt_outputs
        3. add padding token <pad> for the sentences less than max length in a batch
        '''

        def merge(batched_token_ids, only_bos=False, only_eos=False):
            # get max sentence length in this batch
            max_length = max(token_ids.size(0) for token_ids in batched_token_ids)
            if only_bos:
                max_length -= 1
            if only_eos:
                max_length -= 1
            # initialize return batch
            result = batched_token_ids[0].new(len(batched_token_ids), max_length).fill_(self.pad_id)
            for i, v in enumerate(batched_token_ids):
                if only_bos:
                    # only keep <s>
                    result[i, : len(v) - 1] = v[:-1]
                elif only_eos:
                    # only keep </s>
                    result[i, : len(v) - 1] = v[1:]
                else:
                    # unchange
                    result[i, : len(v)].copy_(v)
            return result

        src_inputs = merge([s['src'] for s in samples])
        tgt_inputs = merge([s['tgt'] for s in samples], only_bos=True)  # for training use
        tgt_outputs = merge([s['tgt'] for s in samples], only_eos=True)  # for criterion use

        return {
            'src_inputs': src_inputs,
            'tgt_inputs': tgt_inputs,
            'tgt_outputs': tgt_outputs,
        }
