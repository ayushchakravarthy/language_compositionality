import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange

from .lp_layers import LPLayerEncoder, LPLayerDecoder
from .tf_layers import TransformerEncoderLayer, TransformerDecoderLayer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2).float() * (-math.log(10000.0))
        div_term = torch.exp(div_term / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.shape = [0]
        self.register_buffer('pe', pe)

    def _get_pe(self, x):
        return self.pe[:x.size(0), :]

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LPBlock(nn.Module):
    def __init__(self, dim, ffn_exp=4, num_heads=1, num_parts=0, dropout=0.1):
        super(LPBlock, self).__init__()
        self.encoder = LPLayerEncoder(dim, num_parts=num_parts, num_heads=num_heads)
        self.decoder = LPLayerDecoder(dim, num_heads=num_heads, ffn_exp=ffn_exp)
        
        self.whole_qpos = PositionalEncoding(dim, dropout)

    
    def forward(self, x, parts, part_qpos, part_kpos, mask=None):
        """
        Args:
            x: [B, seq_len, d]
            parts: [B, N, d]
            mask: [B, seq_len]
        Returns: 
            feats: [B, seq_len, d]
            parts: [B, N, d]
            block_attn_maps: {encoder_attn_maps, decoder_attn_maps}
        """
        parts, enc_attn_maps = self.encoder(x, parts=parts, qpos=part_qpos, mask=mask)
        feats, dec_attn_maps = self.decoder(x, parts=parts, part_kpos=part_kpos, whole_qpos=self.whole_qpos, mask=mask)

        block_attn_maps = {
            'encoder': enc_attn_maps,
            'decoder': dec_attn_maps
        }
        return feats, parts, part_qpos, mask, block_attn_maps

class Stage(nn.Module):
    def __init__(self, dim, num_blocks, num_heads, num_parts, ffn_exp, dropout, last_enc=False):
        super(Stage, self).__init__()
        self.part_qpos = nn.Parameter(torch.Tensor(1, num_parts, 1, dim // num_heads))
        self.part_kpos = nn.Parameter(torch.Tensor(1, num_parts, 1, dim // num_heads))
        self.last_enc = last_enc

        blocks = [
            LPBlock(
                dim=dim,
                ffn_exp=ffn_exp,
                num_heads=num_heads,
                num_parts=num_parts,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)
        if self.last_enc:
            self.last_encoder = LPLayerEncoder(
                dim,
                num_parts,
                num_heads,
                has_ffn=False
            )
        self._init_weights()
    
    def _init_weights(self):
        init.kaiming_uniform_(self.part_qpos, a=math.sqrt(5))
        init.trunc_normal_(self.part_qpos, std=.02)
        init.kaiming_uniform_(self.part_kpos, a=math.sqrt(5))
        init.trunc_normal_(self.part_kpos, std=.02)
    
    def forward(self, x, parts=None, mask=None):
        """
        Args:
            x: [B, seq_len, d]
            parts: [B, N, d]
            mask: [B, seq_len]
        Returns:
            out: [B, seq_len, d] or [B, N, d]
            parts: [B, N, d]
            stage_attn_maps: list of block attn maps
        """
        part_qpos, part_kpos = self.part_qpos, self.part_kpos
        part_qpos = part_qpos.expand(x.shape[0], -1, -1, -1)
        part_kpos = part_kpos.expand(x.shape[0], -1, -1, -1)

        stage_attn_maps = []

        for blk in self.blocks:
            x, parts, part_qpos, mask, attn_map = blk(
                x, 
                parts=parts,
                part_qpos=part_qpos,
                part_kpos=part_kpos,
                mask=mask
            )

            stage_attn_maps.append(attn_map)

        if self.last_enc:
            dec_mask = None if mask is None else rearrange(mask, 'b s -> b 1 1 s')
            out, attn_map = self.last_encoder(x, parts=parts, qpos=part_qpos, mask=dec_mask)
            stage_attn_maps.append(attn_map)
        else:
            out = x
        return out, parts, mask, stage_attn_maps
        

class LPEncoder(nn.Module):
    def __init__(self, dim, num_layers, num_heads, num_parts, ffn_exp, dropout, act=nn.GELU):
        super(LPEncoder, self).__init__()

        self.depth = len(num_layers)
        self.act = act()
        self.rpn_tokens = nn.Parameter(torch.Tensor(1, num_parts[0], dim))

        for i, n_l in enumerate(num_layers):
            setattr(
                self,
                "layer_{}".format(i),
                Stage(
                    dim=dim,
                    num_blocks=n_l,
                    num_heads=num_heads[i],
                    num_parts=num_parts[i],
                    ffn_exp=ffn_exp,
                    dropout=dropout,
                    last_enc=(i==len(num_layers)-1)
                )
            )
        
        self._init_weights()

    def _init_weights(self):
        init.kaiming_uniform_(self.rpn_tokens, a=math.sqrt(5))
        init.trunc_normal_(self.rpn_tokens, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if not torch.sum(m.weight.data == 0).item() == m.num_features:  # zero gamma
                    m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, seq_len, d]
            mask: [B, seq_len]
        Returns:
            out: [B, N, d]
            total_attn_maps: list of all LPEncoder attn maps
        """
        out = x
        rpn_tokens = self.rpn_tokens.expand(out.shape[0], -1, -1)

        total_attn_maps = []

        for i in range(self.depth):
            layer = getattr(self, "layer_{}".format(i))
            out, rpn_tokens, mask, attn_map = layer(out, rpn_tokens, mask)
            total_attn_maps.append(attn_map)
        
        out = self.act(out)

        return out, total_attn_maps

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



class LanguageParser(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, nhead,
                 num_decoder_layers, dim_feedforward, dropout, pad_idx, device):
        super(LanguageParser, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.activation = 'relu'
        self.device = device

        # Input
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Encoder
        self.encoder = lp_base(d_model)

        # Decoder 
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, self.activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers,
                                          decoder_norm)

        # Output
        self.linear = nn.Linear(d_model, trg_vocab_size)

        # Softmax
        self.softmax = nn.LogSoftmax(dim=-1)

        # Initialize parameters
        self._reset_parameters()

    def _get_masks(self, src, trg):
        sz = trg.shape[1]
        trg_mask = self._generate_square_subsequent_mask(sz)
        trg_mask = trg_mask.to(self.device)
        src_kp_mask = (src == self.pad_idx).to(self.device)
        trg_kp_mask = (trg == self.pad_idx).to(self.device)
        return trg_mask, src_kp_mask, trg_kp_mask
    
    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def forward(self, src, trg):
        """
        Args:
            src: [B, src_seq_len]
            trg: [B, trg_seq_len]
        Returns:
            out: [B, trg_seq_len, d]
            attn_maps: list of all attn maps
        """
        src_mask = None
        # trg_mask: [trg_seq_len, trg_seq_len]
        # src_kp_mask: [B, src_seq_len]
        # trg_kp_mask: [B, trg_seq_len]
        trg_mask, src_kp_mask, trg_kp_mask = self._get_masks(src, trg)

        # Input
        # src: [B, src_seq_len, d]
        # trg: [B, trg_seq_len, d]
        src = self.src_embedding(src)
        src = self.positional_encoding(src)
        trg = self.trg_embedding(trg)
        trg = self.positional_encoding(trg)

        # Encoder
        memory, enc_attn_wts = self.encoder(src, src_kp_mask)

        memory_mask = None
        memory_kp_mask = None

        # Decoder
        out, dec_attn_wts = self.decoder(trg, memory, trg_mask=trg_mask,
                                         memory_mask=memory_mask,
                                         trg_kp_mask=trg_kp_mask,
                                         memory_kp_mask=memory_kp_mask) 
        
        # Output
        out = self.linear(out)

        # Softmax
        out = self.softmax(out)

        # Attention Weights
        # Decide on what goes in the enc_attn_wts
        attn_wts = {'Encoder': enc_attn_wts,
                    'Decoder': dec_attn_wts}
        
        return out, attn_wts

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, nhead,
                 num_encoder_layers, num_decoder_layers, dim_feedforward,
                 dropout, pad_idx, device):
        super(Transformer,self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
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
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)
        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, self.activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers,
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



# TODO: figure out how to change num_parts without breaking
def lp_base(dim):
    model_cfg = dict(dim=dim, num_layers=[2, 2, 4, 4], num_heads=[8, 8, 8, 8],
                     num_parts=[64, 64, 64, 64], ffn_exp=3, dropout=0.1)
    return LPEncoder(**model_cfg)
