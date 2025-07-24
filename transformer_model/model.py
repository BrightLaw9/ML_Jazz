
# Music transformer as described in the research paper by Huang et al., 2018
import torch
from math import sqrt
from torch import nn

class TransformerModel(nn.Module): 
    def __init__(self, d_model, num_layers, num_heads, d_ff, max_rel_dist, max_abs_position, vocab_size, dropout, bias, layernorm_eps):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_rel_dist = max_rel_dist,
        self.max_position = max_abs_position
        self.vocab_size = vocab_size

        self.input_embedding = nn.Embedding(vocab_size, d_model)
       # self.positional_encoding = abs_positional_encoding(max_abs_position, d_model)
        self.input_dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',  # default
            layer_norm_eps=layernorm_eps,
            batch_first=True,  # Optional depending on your data shape
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        )

        self.final = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        """
        Forward pass through the Music Transformer. Embeds x according to Vaswani et. al, 2017, adds absolute
        positional encoding if present, performs dropout, passes through the stack of decoder layers, and
        projects into the vocabulary space. DOES NOT SOFTMAX OR SAMPLE OUTPUT; OUTPUTS LOGITS.

        Args:
            x (torch.Tensor): input batch of sequences of shape (batch_size, seq_len)
            mask (optional): mask for input batch indicating positions in x to mask with 1's. Default: None

        Returns:
            input batch after above steps of forward pass through MusicTransformer
        """

        # embed x according to Vaswani et. al, 2017
        x = self.input_embedding(x)
        x *= sqrt(self.d_model)

        # add absolute positional encoding if max_position > 0, and assuming max_position >> seq_len_x
        if self.max_position > 0:
            x += self.positional_encoding[:, :x.shape[-2], :]

        # input dropout
        x = self.input_dropout(x)

        # pass through decoder
        x = self.decoder(x, memory=None, tgt_mask=mask)

        # final projection to vocabulary space
        return self.final(x)