# Music transformer as described in the research paper by Huang et al., 2018
from math import sqrt
import torch.nn as nn
import torch
from .relative_self_attn import RelativeSelfAttention
from .constants import D_MODEL, NUM_LAYERS, NHEAD, DIM_FF, NOTE_ON_OFFSET, DUR_OFFSET, VELOCITY_OFFSET, TIME_SHIFT_OFFSET

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, max_len, dropout=0.1):
        super().__init__()
        self.attn = RelativeSelfAttention(d_model, nhead, max_len)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        # Pre layer norm? - might add more stability
        # x2 = self.attn(self.ln_attention(x))
        # x = x + x2
        # x2 = self.ff(self.ln_ff(x))
        # x = x + x2

        # Post layer norm
        x = self.ln1(x + self.attn(x, attn_mask))
        x = self.ln2(x + self.ff(x))
        return x

class JazzTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=D_MODEL, 
                 n_layers=NUM_LAYERS, nhead=NHEAD, dim_ff=DIM_FF):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.input_ln = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dim_ff, max_len)
            for _ in range(n_layers)]
        )
        self.out_head = nn.Linear(d_model, vocab_size)

        self.vocab_size = vocab_size

    @staticmethod
    def build_logit_mask(tokens, vocab_size):
        B, L = tokens.shape
        mask = torch.zeros(B, L, vocab_size, device=tokens.device)

        for b in range(B):
            
            for t in range(L):
                # update state using ground truth
                tok = tokens[b, t].item()
                if NOTE_ON_OFFSET <= tok < DUR_OFFSET:
                    mask[b, t, DUR_OFFSET:TIME_SHIFT_OFFSET] = 1
                elif DUR_OFFSET <= tok < TIME_SHIFT_OFFSET:
                    mask[b, t, VELOCITY_OFFSET:] = 1
                    mask[b, t, TIME_SHIFT_OFFSET:VELOCITY_OFFSET] = 1
                elif TIME_SHIFT_OFFSET <= tok < VELOCITY_OFFSET:
                    mask[b, t, VELOCITY_OFFSET:] = 1
                    mask[b, t, TIME_SHIFT_OFFSET:VELOCITY_OFFSET] = 1
                elif VELOCITY_OFFSET <= tok:
                    mask[b, t, NOTE_ON_OFFSET:DUR_OFFSET] = 1

        return mask

    def forward(self, tokens, attn_mask=None):
        x = self.token_emb(tokens)
        # x = x.transpose(0, 1)
        x = self.input_ln(x)
        for layer in self.layers: 
            x = layer(x, attn_mask)
        # x = x.transpose(0,1)
        logits = self.out_head(x)

        logit_mask = JazzTransformer.build_logit_mask(tokens, self.vocab_size)
        #print(f"Logit mask", logit_mask)

        logits = logits.masked_fill(logit_mask == 0, float('-inf'))
        # print("Logits: ", logits)
        return logits

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