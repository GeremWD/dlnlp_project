
from torch import nn
import torch.nn.functional as F
from position_embedding import *
from copy import deepcopy

class TransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, feedforward_dim, dropout):
        """
        :param int d_model: 一般512之类的
        :param self_attn: self attention模块，输入为x:batch_size x max_len x d_model, mask:batch_size x max_len, 输出为
            batch_size x max_len x d_model
        :param int feedforward_dim: FFN中间层的dimension的大小
        :param float dropout: 一共三个位置的dropout的大小
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = self_attn

        self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(feedforward_dim, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len x hidden_size
        :param mask: batch_size x max_len, 为0的地方为pad
        :return: batch_size x max_len x hidden_size
        """
        residual = x
        x = self.self_attn(x, mask)
        x = x + residual
        x = self.norm1(x)
        residual = x
        x = self.ffn(x)
        x = residual + x
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, feedforward_dim, dropout, output_dim, fc_dropout=0.3):
        super().__init__()
        dropout_attn = dropout
        self.d_model = d_model

        self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout_attn)

        self.layers = nn.ModuleList([TransformerLayer(d_model, deepcopy(self_attn), feedforward_dim, dropout)
                       for _ in range(num_layers)])
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.out_fc = nn.Linear(d_model, output_dim)
        

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len x d_model
        :param mask: batch_size x max_len. 有value的地方为1
        :return:
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.out_fc(x)