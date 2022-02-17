import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import PositionalEncoding, RelativeEmbedding
from .tf_layers import TransformerEncoderLayer, TransformerDecoderLayer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.norm = norm

        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, trg, memory, trg_mask=None, memory_mask=None,
                trg_kp_mask=None, memory_kp_mask=None):
        attn_weights = []
        output = trg
        for mod in self.layers:
            output, attn_wts = mod(output, memory, trg_mask=trg_mask,
                                       memory_mask=memory_mask,
                                       trg_kp_mask=trg_kp_mask,
                                       memory_kp_mask=memory_kp_mask)
            attn_weights.append(attn_wts)
        if self.norm is not None:
            output = self.norm(output)
        
        return output, attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self,encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.norm = norm

        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src, mask=None, src_kp_mask=None):
        attn_weights = []
        output = src
        for mod in self.layers:
            output, attn_wts = mod(output, src_mask=mask,
                                   src_kp_mask=src_kp_mask)
            attn_weights.append(attn_wts.detach().cpu())

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, nhead,
                 n_layers,  dim_feedforward,
                 dropout, pad_idx, device):
        super(Transformer,self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = n_layers
        self.num_decoder_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.activation = 'relu'
        self.device = device

        # Input
        self.src_embedding = nn.Embedding(src_vocab_size,d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size,d_model)
        self.positional_encoding = PositionalEncoding(d_model,dropout)
        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, self.activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, self.num_encoder_layers,
                                          encoder_norm)
        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, self.activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, self.num_decoder_layers,
                                          decoder_norm)
        # Output
        self.linear = nn.Linear(d_model,trg_vocab_size)

        self.softmax = nn.LogSoftmax(dim=-1)
        

        # Initialize
        self._reset_parameters()
    

    def forward(self,src,trg):
        # src: [B, src_seq_len]
        # trg: [B, trg_seq_len]
        # Masks
        src_mask = None
        trg_mask,src_kp_mask,trg_kp_mask = self._get_masks(src,trg)

        # Input
        # src: [B, src_seq_len, d_model]
        src = self.src_embedding(src)
        src = self.positional_encoding(src)

        # trg: [B, trg_seq_len, d_model]
        trg = self.trg_embedding(trg)
        trg = self.positional_encoding(trg)

        # Encoder
        memory, enc_attn_wts = self.encoder(src, mask=src_mask,
                                            src_kp_mask=src_kp_mask)
        # Decoder
        memory_mask = None
        memory_kp_mask = None
        out, dec_attn_wts = self.decoder(trg, memory, trg_mask=trg_mask,
                                         memory_mask=memory_mask,
                                         trg_kp_mask=trg_kp_mask,
                                         memory_kp_mask=memory_kp_mask)
        # Output
        out = self.linear(out)

        # Softmax
        out = self.softmax(out)
        
        # Attention weights
        attn_wts = {'Encoder':enc_attn_wts,
                    'Decoder':dec_attn_wts}
        return out, attn_wts

    def _get_masks(self,src,trg):
        sz = trg.shape[1]
        trg_mask = self._generate_square_subsequent_mask(sz)
        trg_mask = trg_mask.to(self.device)
        src_kp_mask = (src == self.pad_idx).to(self.device)
        trg_kp_mask = (trg == self.pad_idx).to(self.device)
        return trg_mask,src_kp_mask,trg_kp_mask

    def _generate_square_subsequent_mask(self,sz):
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)