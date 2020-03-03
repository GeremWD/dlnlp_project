
from torch import nn
import torch.nn.functional as F
from position_embedding import *
from copy import deepcopy

class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, attention, feedforward_dim, dropout):
        super().__init__()
        self.attention = attention
        self.first_norm = nn.LayerNorm(embedding_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(dropout))

        self.second_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask):
        residual = x
        x = self.attention(x, mask)
        x = x + residual
        x = self.first_norm(x)
        residual = x
        x = self.feedforward(x)
        x = residual + x
        x = self.second_norm(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, n_head, feedforward_dim, dropout, output_dim):
        super().__init__()
        self_attn = RelativeMultiHeadAttn(embedding_dim, n_head, dropout)

        self.layers = nn.ModuleList([TransformerLayer(embedding_dim, deepcopy(self_attn), feedforward_dim, dropout)
                       for _ in range(num_layers)])
        self.lin = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.lin(x)