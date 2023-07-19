from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    model_type = 'transformer'

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_attention_heads,
        num_hidden_layers,
        dropout,
        max_sequence_length,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
