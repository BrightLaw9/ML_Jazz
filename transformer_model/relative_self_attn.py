import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_len = max_len

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

        # Relative position embeddings
        self.rel_emb = nn.Parameter(
            torch.randn(2 * max_len - 1, self.d_head)
        )

    def _skew(self, x):
        # x: (B, H, L, 2L-1)
        B, H, L, _ = x.size()

        # pad on the last dimension
        x = F.pad(x, (1, 0))          # (B, H, L, 2L)

        # reshape so that relative positions shift correctly
        x = x.view(B, H, L, -1)       # (B, H, L, 2L)

        # drop the first column
        x = x[..., :L]                # (B, H, L, L)

        return x

    def forward(self, x, mask=None):
        B, L, _ = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Content-based attention
        content_scores = torch.matmul(q, k.transpose(-2, -1))

        # Relative position scores
        # print("Rel embeddings", self.rel_emb.T.unsqueeze(0).unsqueeze(0).size())
        rel_logits = torch.matmul(
            q,                          # (B, H, L, d_head)
            self.rel_emb.T.unsqueeze(0).unsqueeze(0)  # (1,1,d_head,2L-1)
        )
        # print("Rel logits: ", rel_logits.size())
        rel_logits = self._skew(rel_logits)

        # print("Content scores: ", content_scores.size())
        # print("Rel logits: ", rel_logits.size())

        scores = (content_scores + rel_logits) / (self.d_head ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out(out)
