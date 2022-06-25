import math

import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 pad_id,
                 embed_dim=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 ffn_embed_dim=2048,
                 dropout=0.1,
                 max_positions=1000):
        super().__init__()

        self.args = {'src_vocab_size': src_vocab_size,
                     'tgt_vocab_size': tgt_vocab_size,
                     'pad_id': pad_id
                     }
        self.kwargs = {'embed_dim': embed_dim,
                       'nhead': nhead,
                       'num_encoder_layers': num_encoder_layers,
                       'num_decoder_layers': num_decoder_layers,
                       'ffn_embed_dim': ffn_embed_dim,
                       'dropout': dropout,
                       'max_positions': max_positions
                       }

        self.pad_id = pad_id
        self.embed_scale = math.sqrt(embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_positions+self.pad_id+1, dropout)

        self.src_embed_layer = nn.Embedding(src_vocab_size, embed_dim, self.pad_id)
        self.tgt_embed_layer = nn.Embedding(tgt_vocab_size, embed_dim, self.pad_id)

        self.transformer_layer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ffn_embed_dim,
            dropout=dropout,
            batch_first=True
        )

        self.project_layer = nn.Linear(embed_dim, tgt_vocab_size)

        self.register_buffer('self_attn_mask', nn.Transformer.generate_square_subsequent_mask(max_positions))

    def forward(self, src_tokens, tgt_inputs):
        '''
            src_tokens:  <--- [batch_size, sent_len(src_time_steps)]
            tgt_inputs:  <--- [batch_size, sent_len(tgt_time_steps)]
        '''
        # [batch_size, sent_len, embed_dim]
        src_embeddings = self.embed_scale * self.src_embed_layer(src_tokens)
        tgt_embeddings = self.embed_scale * self.tgt_embed_layer(tgt_inputs)
        # positional encoding
        src_states = self.pos_encoder(src_embeddings)
        tgt_states = self.pos_encoder(tgt_embeddings)
        # padding mask
        src_key_padding_mask = src_tokens.eq(self.pad_id) if src_tokens.eq(self.pad_id).any() else None
        tgt_key_padding_mask = tgt_inputs.eq(self.pad_id) if tgt_inputs.eq(self.pad_id).any() else None
        # attention mask
        sent_len = tgt_inputs.size(1)
        self_attn_mask = self.self_attn_mask[:sent_len, :sent_len]
        # forward passing
        out = self.transformer_layer(src_states, tgt_states,
                                     tgt_mask=self_attn_mask,  # mask attention
                                     src_key_padding_mask=src_key_padding_mask,     # mask padding
                                     tgt_key_padding_mask=tgt_key_padding_mask,     # mask padding
                                     memory_key_padding_mask=src_key_padding_mask)  # mask padding
        # final layer
        out = self.project_layer(out)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_positions, dropout):
        super().__init__()

        position = torch.arange(max_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000) / embed_dim))
        pe = torch.zeros(1, max_positions, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
            x: Tensor, shape [batch_size, sent_len, embed_dim]
        '''
        x = x + self.pe[:, x.size(1)]
        return self.dropout(x)
