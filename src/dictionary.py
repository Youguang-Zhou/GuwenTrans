import re
from collections import Counter

import torch


class Dictionary:
    def __init__(self, sos='<s>', eos='</s>', pad='<pad>', unk='<unk>'):
        self.sos, self.eos, self.pad, self.unk = sos, eos, pad, unk
        self.tokens = []
        self.token2id = {}
        self.token2count = {}
        self.sos_id = self._add_token(sos, n=0)
        self.eos_id = self._add_token(eos, n=0)
        self.pad_id = self._add_token(pad, n=0)
        self.unk_id = self._add_token(unk, n=0)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx] if idx < len(self.tokens) else self.unk

    @classmethod
    def build_from_file(cls, fname, threshold_unk=None, threshold_tokens=None):
        '''
            fname: 数据文件路径
            threshold_unk:    token出现次数少于这个数字的设置为<unk>
            threshold_tokens: 需要保留的token数量
        '''
        dictionary = cls()
        # read tokens from file
        tokens = []
        with open(fname, 'r') as f:
            for line in f:
                for token in cls._tokenize(line):
                    tokens.append(token)
        # apply thresholds
        num_retain_tokens = len(tokens) if threshold_tokens == None else threshold_tokens
        unk_cut_line = 0 if threshold_unk == None else threshold_unk
        for token, count in Counter(tokens).most_common(num_retain_tokens):
            if count >= unk_cut_line:
                dictionary._add_token(token, count)
        return dictionary

    def string_to_tokenIds(self, string, return_unk_count=False):
        '''默认添加<s>和</s>'''
        # apply tokenizer
        tokens = self._tokenize(string)
        # initialize token ids
        token_ids = torch.IntTensor(len(tokens)+2)
        # the first token is <s>
        token_ids[0] = self.sos_id
        num_unk = 0
        for i, token in enumerate(tokens):
            id = self.get_token_id(token)
            token_ids[i+1] = id
            if id == self.unk_id:
                num_unk += 1
        # the last token is </s>
        token_ids[-1] = self.eos_id
        if return_unk_count:
            return token_ids, num_unk
        else:
            return token_ids

    def tokenIds_to_string(self, token_ids):
        '''默认不添加<s>和</s>'''
        return ''.join(self[id] for id in token_ids if (id != self.eos_id) and (id != self.sos_id))

    def get_token_id(self, token):
        return self.token2id.get(token, self.unk_id)

    def save(self, fname):
        with open(fname, 'w') as f:
            for token, count in Counter(self.token2count).most_common():
                if token != self.sos and token != self.eos and token != self.pad and token != self.unk:
                    f.write(f'{token} {count}\n')

    @classmethod
    def load(cls, fname):
        with open(fname, 'r') as f:
            dictionary = cls()
            for line in f.readlines():
                token, count = line.split(' ')
                dictionary._add_token(token, int(count))
            return dictionary

    def _add_token(self, token, n=1):
        if token in self.token2id:
            id = self.token2id[token]
            self.token2count[token] += n
            return id
        else:
            id = len(self)
            self.tokens.append(token)
            self.token2id[token] = id
            self.token2count[token] = n
            return id

    @staticmethod
    def _tokenize(string):
        return list(re.sub('\s+', '', string))
