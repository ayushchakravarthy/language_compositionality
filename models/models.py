import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange

from .layers import *


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    else:
        raise RuntimeError(f"Invalid Activation {activation}")

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

class Encoder(nn.Module):
    def __init__(self, dim, num_parts=64, num_heads=1, act=nn.GELU,
                 has_ffn=True):
        super(Encoder, self).__init__()
        self.enc_attn = AnyAttention(dim, num_heads)
        self.reason = SimpleReasoning(num_parts, dim)
        self.enc_ffn = MLP(dim, hidden_features=dim, act_layer=act) if has_ffn else None

    def forward(self, feats, parts=None, qpos=None, kpos=None, mask=None):
        """
        Args:
            feats: [B, seq_len, d]
            parts: [B, N, d]
            qpos: [B, N, 1, d]
            kpos: [B, seq_len, d]
            mask: [B, seq_len]
        Returns:
            parts: [B, N, d]
            attn_map: [B, N, num_heads, seq_len]
        """
        mask = None if mask is None else rearrange(mask, 'b s -> b 1 1 s')
        attn_out, attn_map = self.enc_attn(q=parts, k=feats, v=feats, qpos=qpos, kpos=kpos, mask=mask, is_class=False)
        parts = parts + attn_out
        parts = self.reason(parts)
        if self.enc_ffn is not None:
            parts = parts + self.enc_ffn(parts)
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

    def forward(self, x, parts=None, part_kpos=None, whole_qpos=None, mask=None):
        """
        Args:
            x: [B, seq_len, d]
            parts: [B, N, d]
            part_kpos: [B, N, 1, d] 
            whole_qpos: PositionalEncoding instance
            mask: [B, seq_len]
            P: patch_num
        Returns:
            feats: [B, seq_len, d]
            attn_map: [[B, seq_len, num_heads, N], [B, seq_len, num_heads, N]]
        """
        dec_mask = None if mask is None else rearrange(mask, 'b s -> b s 1 1')
        out, attn_map1 = self.attn1(q=x, k=parts, v=parts, qpos=whole_qpos, kpos=part_kpos, mask=dec_mask, is_class=True)
        out = x + out
        out = out + self.ffn1(out)

        # self attention
        local_out, attn_map2 = self.attn2(q=out, k=out, v=out, mask=dec_mask)
        out = local_out
        out = out + self.ffn2(out)
        return out

class LPBlock(nn.Module):
    def __init__(self, dim, ffn_exp=4, num_heads=1, num_parts=0, dropout=0.1):
        super(LPBlock, self).__init__()
        self.encoder = Encoder(dim, num_parts=num_parts, num_heads=num_heads)
        self.decoder = Decoder(dim, num_heads=num_heads, ffn_exp=ffn_exp)
        
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
        # P = x.shape[1]
        # x = rearrange(x, "b p k c -> b (p k) c")
        block_attn_maps = []
        parts = self.encoder(x, parts=parts, qpos=part_qpos, mask=mask)
        feats = self.decoder(x, parts=parts, part_kpos=part_kpos, whole_qpos=self.whole_qpos, mask=mask)
        return feats, parts, part_qpos, mask

class Stage(nn.Module):
    def __init__(self, dim, num_blocks, num_heads, num_parts, ffn_exp, dropout, last_enc=False):
        super(Stage, self).__init__()
        self.part_qpos = nn.Parameter(torch.Tensor(1, num_parts, 1, dim // num_heads))
        self.part_kpos = nn.Parameter(torch.Tensor(1, num_parts, 1, dim // num_heads))
        self.last_enc = last_enc

        blocks = [
            LPBlock(
                dim,
                ffn_exp,
                num_heads,
                num_parts,
                dropout
            )
            for i in range(num_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)
        if self.last_enc:
            self.last_enc = Encoder(
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
        TODO: Write docstring
        """
        part_qpos, part_kpos = self.part_qpos, self.part_kpos
        part_qpos = part_qpos.expand(x.shape[0], -1, -1, -1)
        part_kpos = part_kpos.expand(x.shape[0], -1, -1, -1)

        for blk in self.blocks:
            x, parts, part_qpos, mask = blk(
                x, 
                parts=parts,
                part_qpos=part_qpos,
                part_kpos=part_kpos,
                mask=mask
            )
        dec_mask = None if mask is None else rearrange(mask, 'b s -> b 1 1 s')
        if self.last_enc is not None:
            out = self.last_enc(x, parts=parts, qpos=part_qpos, mask=None)
        else:
            out = x
        return out, parts, mask

        

"""
This class should have the computation from the blocks and output parts and wholes from multiple blocks
"""
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
        args:
            x: [seq_len, B, d]
            mask: [B, seq_len]
        returns:
            TODO: something something parts from the last_encoder learned parts
        """
        out = x
        rpn_tokens = self.rpn_tokens.expand(x.shape[0], -1, -1)
        # reshape mask?

        for i in range(self.depth):
            layer = getattr(self, "layer_{}".format(i))
            out, rpn_tokens, mask = layer(out, rpn_tokens, mask)
        
        out = self.act(out)

        # ideally this should be N, B, d
        print(out.shape)
        exit()
        # does this go into the transformer decoder directly?
        return out

class LPDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(LPDecoder, self).__init__()
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

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        # Self Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

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

        # Encoder stuff should go here
        # Define encoder and probably also define the rpn_tokens which should go in as parts
        self.encoder = lp_base()

        # Decoder 
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, self.activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = LPDecoder(decoder_layer, num_decoder_layers,
                                 decoder_norm)

        # Output
        self.linear = nn.Linear(d_model, trg_vocab_size)

        # Initialize parameters
        self._reset_parameters()

    def _get_masks(self, src, trg):
        sz = trg.shape[0]
        trg_mask = self._generate_square_subsequent_mask(sz)
        trg_mask = trg_mask.to(self.device)
        src_kp_mask = (src == self.pad_idx).transpose(0, 1).to(self.device)
        trg_kp_mask = (trg == self.pad_idx).transpose(0, 1).to(self.device)
        return trg_mask, src_kp_mask, trg_kp_mask
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def forward(self, src, trg):
        """
        args:
            src: [src_seq_len, B]
            trg: [trg_seq_len, B]

        """
        src_mask = None
        # trg_mask: [trg_seq_len, trg_seq_len]
        # src_kp_mask: [B, src_seq_len]
        # trg_kp_mask: [B, trg_seq_len]
        trg_mask, src_kp_mask, trg_kp_mask = self._get_masks(src, trg)

        # Input
        # src: [src_seq_len, B, d]
        # trg: [trg_seq_len, B, d]
        src = self.src_embedding(src)
        src = self.positional_encoding(src)
        trg = self.trg_embedding(trg)
        trg = self.positional_encoding(trg)

        # Encoder stuff should go here!
        feats, parts, enc_attn_wts = self.encoder(src, src_kp_mask)
        memory = parts
        # Decide on what goes in the memory in decoder
        memory_mask = None
        memory_kp_mask = None
        out, dec_attn_wts = self.decoder(trg, memory, trg_mask=trg_mask,
                                         memory_mask=memory_mask,
                                         trg_kp_mask=trg_kp_mask,
                                         memory_kp_mask=memory_kp_mask) 
        
        # Output
        out = self.linear(out)

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
        self.decoder = LPDecoder(decoder_layer, num_decoder_layers,
                                          decoder_norm)
        # Output
        self.linear = nn.Linear(d_model,trg_vocab_size)

        self.softmax = nn.LogSoftmax(dim=-1)
        

        # Initialize
        self._reset_parameters()
    
    def predict(self, src):
        src = self.src_embedding(src)
        src = self.positional_encoding(src)


    def forward(self,src,trg):
        # src: [src_len, B]
        # trg: [trg_len, B]
        # Masks
        src_mask = None
        trg_mask,src_kp_mask,trg_kp_mask = self._get_masks(src,trg)

        # Input
        # src: [src_len, B, d_model]
        src = self.src_embedding(src)
        src = self.positional_encoding(src)

        # trg: [trg_len, B, d_model]
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
        sz = trg.shape[0]
        trg_mask = self._generate_square_subsequent_mask(sz)
        trg_mask = trg_mask.to(self.device)
        src_kp_mask = (src == self.pad_idx).transpose(0,1).to(self.device)
        trg_kp_mask = (trg == self.pad_idx).transpose(0,1).to(self.device)
        return trg_mask,src_kp_mask,trg_kp_mask

    def _generate_square_subsequent_mask(self,sz):
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)


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

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class TransformerDefault(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead, 
                 src_vocab_size, trg_vocab_size, pad_idx, device, dim_feedforward = 512, dropout = 0.1):
        super(TransformerDefault, self).__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device=device
        )
        self.linear = nn.Linear(emb_size, trg_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.trg_tok_emb = TokenEmbedding(trg_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.device = device
        self.pad_idx = pad_idx

        self._reset_parameters()
    
    def forward(self, src, trg):
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.create_mask(src, trg)

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        trg_emb = self.positional_encoding(self.trg_tok_emb(trg))

        outs = self.transformer(src_emb, trg_emb, src_mask, trg_mask, None, src_padding_mask, trg_padding_mask, src_padding_mask)

        return self.linear(outs)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def create_mask(self, src, trg):
        src_seq_len = src.shape[0]
        trg_seq_len = trg.shape[0]

        trg_mask = self.generate_square_subsequent_mask(trg_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)

        src_padding_mask = (src == self.pad_idx).transpose(0, 1).to(self.device)
        trg_padding_mask = (trg == self.pad_idx).transpose(0, 1).to(self.device)

        return src_mask, trg_mask, src_padding_mask, trg_padding_mask
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)


def lp_base():
    model_cfg = dict(dim=64, num_layers=[1, 1, 8, 1], num_heads=[2, 2, 4, 4],
                     num_parts=[64, 64, 128, 128], ffn_exp=3, dropout=0.3)
    return LPEncoder(**model_cfg)
