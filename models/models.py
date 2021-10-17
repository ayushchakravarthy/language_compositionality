import math
import copy
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_

from layers import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze()
        div_term = torch.arange(0, d_model, 2).float() * (-math.log(10000.0))
        div_term = torch.exp(div_term / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 0::1] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, dim, num_parts=64, num_enc_heads=1, act=nn.GELU,
                 has_ffn=True):
        super(Encoder, self).__init__()
        self.num_heads = num_enc_heads
        self.enc_attn = AnyAttention(dim, num_enc_heads)
        self.reason = SimpleReasoning(num_parts, dim)
        self.enc_ffn = MLP(dim, hidden_features=dim, act_layer=act) if has_ffn else None

    def forward(self, feats, parts=None, qpos=None, kpos=None):
        """
        Args:
            feats: [B, seq_len, d]
            parts: [B, N, d]
            qpos: [B, N, d]
            kpos: [B, seq_len, d]
        Returns:
            parts: [B, N, d]
        """
        attn_out = self.enc_attn(q=parts, k=feats, v=feats, qpos=qpos, kpos=kpos)
        parts = parts + nn.Identity(attn_out)
        parts = self.reason(parts)
        if self.enc_ffn is not None:
            parts = parts + nn.Identity(self.enc_ffn(parts))
        return parts

class Decoder(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_exp=3, act=nn.GELU):
        super(Decoder, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.attn1 = AnyAttention(dim, num_heads)
        self.attn2 = AnyAttention(dim, num_heads)
        self.ffn1 = MLP(dim, hidden_features=dim*ffn_exp, act_layer=act, norm_layer=nn.LayerNorm)
        self.ffn2 = MLP(dim, hidden_features=dim*ffn_exp, act_layer=act, norm_layer=nn.LayerNorm)

    def forward(self, x, parts=None, part_kpos=None, whole_qpos=None, P=0):
        """
        Args:
            x: [B, seq_len, d]
            parts: [B, N, d]
            part_kpos: [B, N, d]
            whole_qpos: [B, seq_len, d]
            P: patch_num
        Returns:
            feats: [B, seq_len, d]
        """
        out = self.attn1(q=x, k=parts, v=parts, qpos=whole_qpos, kpos=part_kpos)
        out = x + nn.Identity(out)
        out = out + nn.Identity(self.ffn1(out))

        out = rearrange(out, "b (p k) c -> (b p) k c", p=P)
        # TODO: #1 implement the FullRelPos in layers and pass it as an argument here
        local_out = self.attn2(q=out, k=out, v=out)
        out = nn.Identity(local_out)
        out = out + nn.Identity(self.ffn2(out))
        return rearrange(out, "(b p) k c -> b p k c", p=P)

class LPBlock(nn.Module):
    def __init__(self, dim, ffn_exp=4, patch_size=7, num_heads=1, num_enc_heads=1, num_parts=0):
        super(LPBlock, self).__init__()
        self.encoder = Encoder(dim, num_parts=num_parts, num_enc_heads=num_enc_heads)
        self.decoder = Decoder(dim, num_heads=num_heads, patch_size=patch_size, ffn_exp=ffn_exp)
        # TODO: #2 define d_e, d_d, d_w here 
    
    def forward(self, x, parts):
        """
        Args:
            x: [B, seq_len, d]
            parts: [B, N, d]
        Returns: 
            feats: [B, seq_len, d]
            parts: [B, N, d]
        """
        P = x.shape[1]
        x = rearrange(x, "b p k c -> b (p k) c")
        parts = self.encoder(x, parts=parts, qpos=part_qpos)
        feats = self.decoder(x, parts=parts, part_kpos=part_kpos, whole_qpos=whole_qpos, P=P)
        return feats, parts

# class LPEncoder(nn.Module):
#     def __init__(self, [something something]):
#         self.embedding = nn.Embedding()
#         self.blocks_1 = LPBlock()
#     
#     def forward(self, tokens):
#         return x, p