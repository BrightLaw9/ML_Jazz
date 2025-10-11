# Music transformer as described in the research paper by Huang et al., 2018
from math import sqrt
from torch.nn import nn

class TransformerBlock(nn.Module): 
    def __init__(self, d_model=512, nhead=8, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), # take the d_model and expand it to dim_ff for more expressivity in capturing attention patterns
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )

        self.ln_attention = nn.LayerNorm(d_model)
        self.ln_ff = nn.LayerNorm(d_model)
    
    def forward(self, x, attn_mask=None):
        x2, _ = self.attn(x, x, x, attn_mask=attn_mask) # 3 x for query, key, value weighting
        x = self.ln_attention(x + x2) #changes from attn
        x2 = self.ff(x)
        x = self.ln_ff(x + x2) # Adding changes from ff
        return x

class JazzTransformer:
    def __init__(self, vocab_size, d_model=512, n_layers=8, nhead=8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.input_ln = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(n_layers)])
        self.out_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, attn_mask=None):
        x = self.token_emb(tokens)
        x = x.transpose(0, 1)
        x = self.input_ln(x)
        for layer in self.layers: 
            x = layer(x, attn_mask)
        x = x.transpose(0,1)
        return self.out_head(x)

# class TransformerModel(nn.Module): 
#     def __init__(self, d_model, num_layers, num_heads, d_ff, max_rel_dist, max_abs_position, vocab_size, dropout, bias, layernorm_eps):
#         super(TransformerModel, self).__init__()
#         self.d_model = d_model
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.d_ff = d_ff
#         self.max_rel_dist = max_rel_dist,
#         self.max_position = max_abs_position
#         self.vocab_size = vocab_size

#         self.input_embedding = nn.Embedding(vocab_size, d_model)
#        # self.positional_encoding = abs_positional_encoding(max_abs_position, d_model)
#         self.input_dropout = nn.Dropout(dropout)

        
#         self.final = nn.Linear(d_model, vocab_size)

    # def forward(self, x, mask=None):
    #     """
    #     Forward pass through the Music Transformer. Embeds x according to Vaswani et. al, 2017, adds absolute
    #     positional encoding if present, performs dropout, passes through the stack of decoder layers, and
    #     projects into the vocabulary space. DOES NOT SOFTMAX OR SAMPLE OUTPUT; OUTPUTS LOGITS.

    #     Args:
    #         x (torch.Tensor): input batch of sequences of shape (batch_size, seq_len)
    #         mask (optional): mask for input batch indicating positions in x to mask with 1's. Default: None

    #     Returns:
    #         input batch after above steps of forward pass through MusicTransformer
    #     """

    #     # embed x according to Vaswani et. al, 2017
    #     x = self.input_embedding(x)
    #     x *= sqrt(self.d_model)

    #     # add absolute positional encoding if max_position > 0, and assuming max_position >> seq_len_x
    #     # if self.max_position > 0:
    #     #     x += self.positional_encoding[:, :x.shape[-2], :]

    #     # input dropout
    #     x = self.input_dropout(x)

    #     # pass through decoder
    #     x = self.decoder(x, memory=None, tgt_mask=mask)

    #     # final projection to vocabulary space
    #     return self.final(x)