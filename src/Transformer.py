import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor, inference_mode
from tqdm import tqdm


@dataclass
class ModelConfig:
    # model args
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    max_sequence_length: int
    # tokenizer args
    vocab_size: int
    pad_id: int


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, max_sequence_length: int):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # [max_sequence_length, 1]
        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # [1, max_sequence_length, d_model]
        pe = torch.zeros(1, max_sequence_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor):
        '''
        x: [batch_size, seq_len, d_model]
        '''
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.src_embed_layer = nn.Embedding(config.vocab_size, config.d_model, config.pad_id)
        self.tgt_embed_layer = nn.Embedding(config.vocab_size, config.d_model, config.pad_id)
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout, config.max_sequence_length)

        self.transformer_layer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_encoder_layers=config.n_layers,
            num_decoder_layers=config.n_layers,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation=nn.GELU(),
            batch_first=True,
            norm_first=True,
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, src_inputs: Tensor, tgt_inputs: Tensor):
        '''
        src_inputs: [batch_size, seq_len]
        tgt_inputs: [batch_size, seq_len]
        '''
        # [batch_size, seq_len] ---> [batch_size, seq_len, d_model]
        src_embeddings = self.src_embed_layer(src_inputs)
        tgt_embeddings = self.tgt_embed_layer(tgt_inputs)
        # positional embedding
        src_states = self.pos_encoder(src_embeddings)
        tgt_states = self.pos_encoder(tgt_embeddings)
        # padding mask
        src_key_padding_mask = src_inputs.eq(self.config.pad_id)
        tgt_key_padding_mask = tgt_inputs.eq(self.config.pad_id)
        # causal mask
        causal_mask = self.generate_causal_mask(tgt_inputs.size(1)).to(tgt_inputs.device)
        # forward passing
        decoder_out = self.transformer_layer(
            src_states,
            tgt_states,
            tgt_mask=causal_mask,  # mask attention
            src_key_padding_mask=src_key_padding_mask,  # mask padding
            tgt_key_padding_mask=tgt_key_padding_mask,  # mask padding
            memory_key_padding_mask=src_key_padding_mask,  # mask padding
        )
        # [batch_size, seq_len, d_model]
        decoder_out = self.norm(decoder_out)
        # [batch_size, seq_len, vocab_size]
        output = self.lm_head(decoder_out)
        return output

    @staticmethod
    def generate_causal_mask(size: int):
        return torch.triu(torch.ones((size, size), dtype=torch.bool), diagonal=1)

    @inference_mode()
    def generate(self, src_inputs: Tensor, temperature: float, top_p: float, bos_id: int, eos_id: int):
        # [batch_size=1, seq_len=1:max_sequence_length]
        output_ids = torch.tensor([bos_id]).unsqueeze(0).to(src_inputs.device)

        for _ in tqdm(range(self.config.max_sequence_length), leave=False):
            logits = self(src_inputs, output_ids)

            if temperature > 0:
                # sample tokens
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)

                # https://github.com/meta-llama/llama3/blob/main/llama/generation.py
                def sample_top_p(probs, p):
                    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                    probs_sum = torch.cumsum(probs_sort, dim=-1)
                    mask = probs_sum - probs_sort > p
                    probs_sort[mask] = 0.0
                    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                    next_token = torch.multinomial(probs_sort, num_samples=1)
                    next_token = torch.gather(probs_idx, -1, next_token)
                    return next_token

                next_token = sample_top_p(probs, top_p)
            else:
                # greedy decode
                next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # prepare for next iteration
            output_ids = torch.cat([output_ids, next_token], dim=1)

            # check if it is end of sentence token
            if next_token == eos_id:
                break

        return output_ids
