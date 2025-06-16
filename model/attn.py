import torch
import torch.nn as nn
from math import sqrt

class FullAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None,  attention_dropout=0.1):
        super(FullAttention, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        # self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x):
        B, L, D= x.shape
        H = self.n_heads

        query = self.query_projection(x).view(B, L, H, -1)
        key = self.key_projection(x).view(B, L, H, -1)
        value = self.value_projection(x).view(B, L, H, -1)
        E= value.shape[-1]

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", query, key)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshe->blhe", A, value)

        out = out.reshape(B, L, -1)

        return out

