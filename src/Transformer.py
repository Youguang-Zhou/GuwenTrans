import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer as utils
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from .ModelConfig import ModelConfig


class Transformer(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.config = config
        self.dropout = nn.Dropout(config.dropout)

        self.src_embed_layer = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.tgt_embed_layer = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.pos_embed_layer = nn.Embedding(config.max_sequence_length, config.hidden_size)

        self.transformer_layer = nn.Transformer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            num_encoder_layers=config.num_hidden_layers,
            num_decoder_layers=config.num_hidden_layers,
            dim_feedforward=4 * config.hidden_size,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.register_buffer('attn_mask', utils.generate_square_subsequent_mask(config.max_sequence_length).bool())

    def forward(self, src_inputs, tgt_inputs, tgt_outputs=None, **_):
        '''
        src_inputs:  <--- [batch_size, seq_len]
        tgt_inputs:  <--- [batch_size, seq_len]
        tgt_outputs: <--- [batch_size, seq_len]
        '''
        # embeddings: <--- [batch_size, seq_len, d_model]
        src_embeddings = self.src_embed_layer(src_inputs)
        tgt_embeddings = self.tgt_embed_layer(tgt_inputs)
        # positional embedding
        src_pos_embeddings = self.pos_embed_layer(torch.arange(0, src_inputs.size(1), device=src_inputs.device))
        tgt_pos_embeddings = self.pos_embed_layer(torch.arange(0, tgt_inputs.size(1), device=tgt_inputs.device))
        # added together
        src_states = self.dropout(src_embeddings + src_pos_embeddings)
        tgt_states = self.dropout(tgt_embeddings + tgt_pos_embeddings)
        # padding mask
        src_key_padding_mask = src_inputs.eq(self.config.pad_token_id)
        tgt_key_padding_mask = tgt_inputs.eq(self.config.pad_token_id)
        # attention mask
        sent_len = tgt_inputs.size(1)
        attn_mask = self.attn_mask[:sent_len, :sent_len]
        # forward passing
        decoder_out = self.transformer_layer(
            src_states,
            tgt_states,
            tgt_mask=attn_mask,  # mask attention
            src_key_padding_mask=src_key_padding_mask,  # mask padding
            tgt_key_padding_mask=tgt_key_padding_mask,  # mask padding
            memory_key_padding_mask=src_key_padding_mask,  # mask padding
        )
        # logits: <--- [batch_size, seq_len, vocab_size]
        logits = self.lm_head(decoder_out)

        loss = None
        if tgt_outputs is not None:
            loss = F.cross_entropy(
                input=logits.view(-1, logits.size(-1)),
                target=tgt_outputs.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return Seq2SeqLMOutput(loss, logits)

    def prepare_inputs_for_generation(self, input_ids, src_inputs, **_):
        return {
            'src_inputs': src_inputs,
            'tgt_inputs': input_ids,
            'tgt_outputs': None,
        }
