import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from einops import rearrange

def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    else:
        raise RuntimeError(f"Invalid Activation {activation}")

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        # Self Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # Activation
        self.activation = _get_activation_fn(activation)

    def forward(self, trg, memory, trg_mask=None, memory_mask=None,
                trg_kp_mask=None, memory_kp_mask=None):
        trg2, attn_weights1 = self.self_attn(trg, trg, trg, attn_mask=trg_mask,
                                             key_padding_mask=trg_kp_mask)
        trg = trg + self.dropout1(trg2)
        trg = self.norm1(trg)
        trg2, attn_weights2 = self.multihead_attn(trg, memory, memory,
                                                  attn_mask=memory_mask,
                                                  key_padding_mask=memory_kp_mask)
        trg = trg + self.dropout2(trg2)
        trg = self.norm2(trg)
        trg2 = self.linear2(self.dropout(self.activation(self.linear1(trg))))
        trg = trg + self.dropout3(trg2)
        trg = self.norm3(trg)

        attn_weights = {'Sublayer1': attn_weights1.detach().cpu(),
                        'Sublayer2': attn_weights2.detach().cpu()}
        return trg, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # Activation
        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_kp_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_kp_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights