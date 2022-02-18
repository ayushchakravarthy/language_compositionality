# Experimental architecture testing the idea of maintaining a separation
# between roles and fillers throughout the entire architecture (i.e. no binding)

import os
import math
import pdb
import string

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import RelativeEmbedding

def build_tp_sep_transformer(params, pad_idx, vocab_size):
    d_vocab = vocab_size
    d_pos = 200 # max input size    
    d_f = params.dim_feedforward    
    n_L = params.n_layers
    n_I = params.nhead  
    d_x = params.d_model  # token embedding dimension
    d_p = params.d_model  # position embedding dimension    

    d_v = d_x // n_I  # value dimension
    d_r = d_x // n_I  # role dimension  
    d_k = d_x // n_I  # key dimension
    d_q = d_x // n_I  # query dimension 

    dropout = params.dropout    
    cat_xm = params.cat_xm # concatenate x and m for output
    sp_kernel = params.sp_kernel # use alternate similarity computation for search part of attention
    threshold = params.threshold
    scheme = params.encoding_scheme


    embedding = EmbeddingMultilinearSinusoidal(d_vocab=d_vocab,
                                               d_x=d_x,
                                               d_r=d_r,
                                               dropout=dropout,
                                               max_length=d_pos,
                                               cat_xm=cat_xm,
                                               scheme=scheme)
    encoder = Encoder(
        d_x,
        d_q,
        d_k,
        d_v,
        d_f,
        n_I,
        n_L,
        sp_kernel,
        threshold,
        scheme,
        dropout
    )
    decoder = Decoder(
        d_x,
        d_q,
        d_k,
        d_v,
        d_f,
        sp_kernel,
        threshold,
        scheme,
        cat_xm,
        n_I,
        n_L,
        dropout
    )
    model = Seq2Seq(embedding=embedding,
                    encoder=encoder,
                    decoder=decoder,
                    pad_idx=pad_idx,
                    d_x=d_x,
                    d_vocab=d_vocab,
                    cat_xm=cat_xm)
    return model

"""
Positional Embedding
Idea: Modify the traditional sinusoidal Positional Encoding to 
work with the separated input space
__init__:
    Args:
        d_vocab
        d_x
        d_r
        dropout
        max_len
forward:
    Args:
        x: [B, src_seq_len]
        m: [B, src_seq_len]
    Output:
        (x emb_x): [B, src_seq_len] [B, src_seq_len]
        emb_m: [B, src_seq_len, d]
"""
class EmbeddingMultilinearSinusoidal(nn.Module):
    def __init__(self, d_vocab, d_x, d_r, dropout, max_length, cat_xm, scheme):
        super(EmbeddingMultilinearSinusoidal, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length
        self.d_x = d_x
        self.cat_xm = cat_xm     
        self.scheme = scheme

        # token encodings
        self.x_embedding = nn.Embedding(d_vocab, d_x)
        self.m_embedding = nn.Embedding(d_vocab, d_x)
        self.scale = torch.sqrt(torch.FloatTensor([d_x]))    

        # sinusoidal encoding
        if self.scheme == 'absolute':
            pe = torch.zeros(max_length, d_x)
            position = torch.arange(0., max_length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0., d_x, 2) *
                                 -(math.log(10000.0) / d_x))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            # pe = [1, seq_len, d_p]     
            # x -> r

        self.linear = nn.Linear(d_x, d_x)
        self.mul_scale = torch.FloatTensor([1. / math.sqrt(math.sqrt(2) - 1)])       
        self.reset_parameters()      

    def forward(self, x, m):
        # x = [batch_size, src_seq_len]
        # m = [batch_size, src_seq_len]
        emb_x = self.x_embedding(x) * self.scale.to(x.device)
        emb_m = self.m_embedding(m) * self.scale.to(m.device)    
        # sinusoidal
        if self.scheme == 'absolute':
            pos_sin_emb = torch.autograd.Variable(self.pe[:, :x.size(1)],
                                                  requires_grad=False)
            x = emb_x + pos_sin_emb # 0.5 * pos_sin_emb + 0.5 * pos_id_emb
        else:
            x = emb_x
        # x = [batch_size, src_seq_len, d_x]     
        r = self.linear(x) + 1 # ~N(1,1)
        # r = [batch_size, src_seq_len, d_r]     
        x = x * r
        x = self.dropout(x)
        # x = [batch_size, src_seq_len, d_x]
        return (x, emb_x), emb_m     

    def transpose_forward(self, trg):
        # trg = [batch_size, trg_seq_len, d_v, d_r]      
        if self.cat_xm:
            l_m = torch.matmul(trg[0], torch.transpose(self.m_embedding.weight, 0, 1))
            l_x = torch.matmul(trg[1], torch.transpose(self.x_embedding.weight, 0, 1))
            logits = (l_m, l_x)
        else:
            logits = torch.matmul(trg, torch.transpose(self.m_embedding.weight, 0, 1))
        # logits = [batch_size, trg_seq_len, d_vocab]
        return logits    

    def reset_parameters(self):
        nn.init.normal_(self.x_embedding.weight,
                        mean=0,
                        std=1./math.sqrt(self.d_x))
        nn.init.normal_(self.m_embedding.weight,
                        mean=0,
                        std=1./math.sqrt(self.d_x))
        nn.init.normal_(self.linear.weight,
                        mean=0,
                        std=1./math.sqrt(self.d_x))

"""
Transformer Encoder
Idea: standard transformer encoder just with the modified attention for filler and role
__init__:
    Args:
        d_x
        d_q
        d_k
        d_v
        d_f
        n_I
        n_L
        dropout
forward:
    Args:
        src_x: [B, src_seq_len, d_x]
        src_m: [B, src_seq_len, d_x]
        src_mask: [B, 1, attn_sz]
    Output:
        src_x: [B, src_seq_len, d_x]
        src_m: [B, src_seq_len, d_x]
"""
class Encoder(nn.Module):
    def __init__(self, d_x, d_q, d_k, d_v, d_f, n_I, n_L, sp_kernel, threshold, scheme, dropout):
        super().__init__()

        layers = [EncoderLayer(
            d_x,
            d_q,
            d_k,
            d_v,
            d_f,
            n_I,
            sp_kernel,
            threshold,
            scheme,
            dropout
        )]
        for _ in range(n_L - 1):
          layers.append(EncoderLayer(
            d_x,
            d_q,
            d_k,
            d_v,
            d_f,
            n_I,
            sp_kernel,
            threshold,
            scheme,
            dropout
          ))
        self.layers = nn.ModuleList(layers)

    def forward(self, src_x, src_m, src_mask):
        # src_x = [batch_size, src_seq_len, p.d_x]
        # src_m = [batch_size, src_seq_len, p.d_x]
        # src_mask = [batch_size, 1, attn_size]
        encoder_attn_maps = []
        for layer in self.layers:
            src_x, src_m, attn = layer(src_x, src_m, src_mask)
            encoder_attn_maps.append(attn)

        return src_x, src_m, encoder_attn_maps


"""
Transformer Encoder Layer
Idea: an implementation for one transformer decoder layer modified for the linguistic separation idea
__init__:
    Args:
        d_x
        d_q
        d_k
        d_v
        n_I
        d_f
        dropout
forward:
    Args:
        src_x: [B, src_seq_len, d_x]
        src_m: [B, src_seq_len, d_x]
        src_mask: [B, src_seq_len]
    Output:
        src_x: [B, src_seq_len, d_x]
        src_m: [B, src_seq_len, d_x]
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_x, d_q, d_k, d_v, d_f, n_I, sp_kernel, threshold, scheme, dropout):
        super().__init__()

        # sublayer 1
        self.x_layernorm1 = nn.LayerNorm(d_x)
        self.m_layernorm1 = nn.LayerNorm(d_x)
        self.MHA = SelfAttention(d_x, d_q, d_k, d_v, n_I, sp_kernel, threshold, dropout, scheme)
        self.x_dropout1 = nn.Dropout(dropout)
        self.m_dropout1 = nn.Dropout(dropout)
        # sublayer 2
        self.x_layernorm2 = nn.LayerNorm(d_x)
        self.m_layernorm2 = nn.LayerNorm(d_x)
        self.densefilter = PositionwiseFeedforward(d_x=d_x,
                                                   d_f=d_f,
                                                   dropout=dropout)
        self.x_dropout2 = nn.Dropout(dropout)
        self.m_dropout2 = nn.Dropout(dropout)
        # output
        self.x_layernorm3 = nn.LayerNorm(d_x)
        self.m_layernorm3 = nn.LayerNorm(d_x)

    def forward(self, src_x, src_m, src_mask):
        # src_x = [batch_size, src_seq_size, p.d_x]
        # src_m = [batch_size, src_seq_size, p.d_x]
        # src_mask = [batch_size, src_seq_size]

        # sublayer 1
        x = self.x_layernorm1(src_x)
        m = self.m_layernorm1(src_m)
        x, m, attn = self.MHA(x, x, m, src_mask)
        x = self.x_dropout1(x)
        m = self.m_dropout1(m)
        src_x = self.x_layernorm2(src_x + x)
        src_m = self.m_layernorm2(src_m + m)

        # sublayer 2
        x, m = self.densefilter(x, m)
        x = self.x_dropout2(x)
        m = self.m_dropout2(m)
        src_x = self.x_layernorm3(src_x + x)
        src_m = self.m_layernorm3(src_m + m)

        return src_x, src_m, attn

"""
Transformer Decoder
Idea: an implementation for transformer decoder modified for the linguistic separation idea
__init__:
    Args:
        d_x
        d_q
        d_k
        d_v
        d_f
        cat_xm
        n_I
        n_L
        dropout
    #TODO: maybe d_r
forward:
    Args:
        trg_x, trg_m: [B, trg_seq_len, d_x]
        src_x, src_m: [B, src_seq_len, d_x]
        trg_mask: [B, src_seq_len]
        src_mask: [B, trg_seq_len]
    Output:
        trg: [B, trg_seq_len, d_x]
"""
class Decoder(nn.Module):
    def __init__(
        self,
        d_x,
        d_q,
        d_k,
        d_v,
        d_f,
        sp_kernel,
        threshold,
        scheme,
        cat_xm,
        n_I,
        n_L,
        dropout
    ):
        super().__init__()
        self.cat_xm = cat_xm

        self.layers = nn.ModuleList([DecoderLayer(
            d_x,
            d_q,
            d_k,
            d_v,
            d_f,
            sp_kernel,
            threshold,
            scheme,
            n_I,
            dropout
        ) for _ in range(n_L)])
        # if self.cat_xm:
        #   self.out = nn.Linear(2*d_x, d_x)

    def forward(self, trg_x, trg_m, src_x, src_m, trg_mask, src_mask):
        # trg_x, trg_m = [batch_size, trg_seq_size, d_x]
        # src_x, src_m = [batch_size, src_seq_size, d_x]
        # trg_mask = [batch_size, trg_seq_size]
        # src_mask = [batch_size, src_seq_size]
        decoder_attn_maps = []


        for layer in self.layers:
            trg_x, trg_m, attn_self, attn_enc = layer(trg_x, trg_m, src_x, src_m, trg_mask, src_mask)
            attns = {
                'Sublayer1': attn_self,
                'Sublayer2': attn_enc
            }
            decoder_attn_maps.append(attns)

        if self.cat_xm:
            trg = (trg_m, trg_x)
        else:
            trg = trg_m

        return trg, decoder_attn_maps


"""
Transformer Decoder Layer
Idea: an implementation for one transformer decoder layer modified for the linguistic separation idea
__init__:
    Args:
        d_x
        d_q
        d_k
        d_v
        d_f
        n_I
        dropout
forward:
    Args:
        src_x: [B, src_seq_len, d_x]
        src_m: [B, src_seq_len, d_x]
        src_mask: [B, src_seq_len]
    Output:
        trg_x, trg_m: [B, trg_seq_len, d_x]
        src_x, src_m: [B, src_seq_len, d_x]
        trg_mask: [B, src_seq_len]
        src_mask: [B, trg_seq_len]
"""
class DecoderLayer(nn.Module):
    def __init__(self, d_x, d_q, d_k, d_v, d_f, sp_kernel, threshold, scheme, n_I, dropout):
        super().__init__()

        # sublayer 1
        self.x_layernorm1 = nn.LayerNorm(d_x)
        self.m_layernorm1 = nn.LayerNorm(d_x)
        self.selfAttn = SelfAttention(d_x, d_q, d_k, d_v, n_I, sp_kernel, threshold, dropout, scheme)
        self.x_dropout1 = nn.Dropout(dropout)
        self.m_dropout1 = nn.Dropout(dropout)
        # sublayer 2
        self.x_layernorm2 = nn.LayerNorm(d_x)
        self.m_layernorm2 = nn.LayerNorm(d_x)
        self.encAttn = SelfAttention(d_x, d_q, d_k, d_v, n_I, sp_kernel, threshold, dropout, ed=True)
        self.x_dropout2 = nn.Dropout(dropout)
        self.m_dropout2 = nn.Dropout(dropout)
        # sublayer 3
        self.x_layernorm3 = nn.LayerNorm(d_x)
        self.m_layernorm3 = nn.LayerNorm(d_x)
        self.densefilter = PositionwiseFeedforward(d_x=d_x,
                                                   d_f=d_f,
                                                   dropout=dropout)
        self.x_dropout3 = nn.Dropout(dropout)
        self.m_dropout3 = nn.Dropout(dropout)

        self.x_layernorm4 = nn.LayerNorm(d_x)
        self.m_layernorm4 = nn.LayerNorm(d_x)


    def forward(self, trg_x, trg_m, src_x, src_m, trg_mask, src_mask):
        # trg_x, trg_m = [batch_size, trg_seq_size, p.d_x]
        # src_x, src_m = [batch_size, src_seq_size, p.d_x]
        # trg_mask = [batch_size, trg_seq_size]
        # src_mask = [batch_size, src_seq_size]

        # self attention
        x = self.x_layernorm1(trg_x)
        m = self.m_layernorm1(trg_m)
        x, m, attn_self = self.selfAttn(x, x, m, trg_mask)
        x = self.x_dropout1(x)
        m = self.m_dropout1(m)
        trg_x = self.x_layernorm2(trg_x + x)
        trg_m = self.m_layernorm2(trg_m + m)

        # encoder attention
        x, m, attn_enc = self.encAttn(x, src_x, src_m, src_mask)
        x = self.x_dropout2(x)
        m = self.m_dropout2(m)
        trg_x = self.x_layernorm3(trg_x + x)
        trg_m = self.m_layernorm3(trg_m + m)

        # dense filter
        x, m = self.densefilter(x, m)
        x = self.x_dropout3(x)
        m = self.m_dropout3(m)
        trg_x = self.x_layernorm4(trg_x + x)
        trg_m = self.m_layernorm4(trg_m + m)

        return trg_x, trg_m, attn_self, attn_enc
    

"""
Self Attention Layer
Idea: modified dual stream attention (refer slides)
__init__:
    Args:
        d_x
        d_q
        d_k
        d_v
        n_I
        dropout
forward:
    Args:
        query, key, value: [B, seq_len, d_x]
        src_mask: [B, 1, 1, pad_seq]
        trg_mask: [B, 1, pad_seq, past_seq] (?)
    Out:
        x, m: [B, seq_len, d_x]
"""
class SelfAttention(nn.Module):
    def __init__(self, d_x, d_q, d_k, d_v, n_I, sp_kernel, threshold, dropout, scheme='absolute', ed=False):
        super().__init__()
        self.d_x = d_x
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.n_I = n_I
        self.scheme = scheme
        # TODO: make this a bit more clear?
        self.sp_kernel = sp_kernel and ed

        if scheme == 'relative':
            self.pos_embed = RelativeEmbedding(200, d_q)

        if sp_kernel:
            self.tau = threshold
            self.threshold = torch.nn.Threshold(self.tau, 0.0)

        self.dot_scale = torch.FloatTensor([math.sqrt(d_k)])

        self.W_q = nn.Linear(self.d_x, d_q * n_I)
        self.W_k = nn.Linear(self.d_x, d_k * n_I)
        self.W_v = nn.Linear(self.d_x, d_v * n_I)

        self.W_xo = nn.Linear(d_v * n_I, d_x)
        self.W_mo = nn.Linear(d_k * n_I, d_x)

        self.dropout = nn.Dropout(dropout)
        self.mul_scale = torch.FloatTensor([1./math.sqrt(math.sqrt(2) - 1)])



    def forward(self, query, key, value, mask=None):
        # query = key = value = [batch_size, seq_len, p.d_x]
        # src_mask = [batch_size, 1, 1, pad_seq]
        # trg_mask = [batch_size, 1, pad_seq, past_seq]

        bsz = query.shape[0]

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        # Q, K, V = [batch_size, seq_len, n_heads * d_*]

        Q = Q.view(bsz, -1, self.n_I, self.d_q).permute(0,2,1,3)
        K = K.view(bsz, -1, self.n_I, self.d_k).permute(0,2,1,3)
        V = V.view(bsz, -1, self.n_I, self.d_v).permute(0,2,1,3)
        # Q, K, V = [batch_size, n_heads, seq_size, d_*]

        if self.scheme == 'relative':
            dot = torch.einsum('bhid, bhjd -> bhij', Q, K)
            rel_pos = self.pos_embed(Q)
            dot += rel_pos
        else:
            dot = torch.einsum('bhid, bhjd -> bhij', Q, K)

        dot /= self.dot_scale.to(key.device)

        # if self.sp_kernel:
        #     dot = self.threshold(dot) / self.dot_scale.to(key.device)
        # else:
        #     dot = dot / self.dot_scale.to(key.device)

        # energy   = [batch_size, n_heads, query_pos     , key_pos]
        # src_mask = [batch_size, 1      , 1             , attn]
        # trg_mask = [batch_size, 1      , query_specific, attn]

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e10)

        attn = F.softmax(dot, dim=-1)

        if self.sp_kernel:
            attn = self.threshold(attn)
            # TODO: figure whether this is needed
            s_a = torch.sum(attn, dim=-1, keepdim=True)
            if not torch.any(s_a == 0):
                attn /= s_a

        attention = self.dropout(F.softmax(dot, dim=-1))
        # attention = [batch_size, n_heads, seq_size, seq_size]

        k_bar = torch.einsum('bhjd,bhij->bhid', K, attention)
        v_bar = torch.einsum('bhjd,bhij->bhid', V, attention)
        # k_bar, v_bar = [batch_size, n_heads, seq_size, d_*]

        k_bar = k_bar.permute(0,2,1,3).contiguous()
        v_bar = v_bar.permute(0,2,1,3).contiguous()
        # k_bar, v_bar = [batch_size, seq_size, n_heads, d_*]

        k_bar = k_bar.view(bsz, -1, self.n_I * self.d_v)
        v_bar = v_bar.view(bsz, -1, self.n_I * self.d_v)
        # k_bar, v_bar = [batch_size, src_seq_size, n_heads * d_v]

        x = self.W_xo(k_bar)
        m = self.W_mo(v_bar)
        # x, m = [batch_size, seq_size, d_x]

        return x, m, attention

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

        nn.init.normal_(self.W_r.weight,
                        mean=0,
                        std=1./math.sqrt(self.d_r))

"""
PositionwiseFeedforward
__init__:
    Args:
        d_x
        d_f
        dropout
forward:
    Args:
        x, m: [B, seq_len, d_x]
    Output:
        x, m: [B, seq_len, d_x]
"""
class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_x, d_f, dropout):
        super().__init__()

        self.d_x = d_x
        self.d_f = d_f

        self.x_linear1 = nn.Linear(d_x, d_f)
        self.x_linear2 = nn.Linear(d_f, d_x)
        self.x_dropout = nn.Dropout(dropout)

        self.m_linear1 = nn.Linear(d_x, d_f)
        self.m_linear2 = nn.Linear(d_f, d_x)
        self.m_dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def forward(self, x, m):
        # x, m = [batch_size, seq_size, d_x]

        x = self.x_linear1(x)
        x = self.x_dropout(F.relu(x))
        x = self.x_linear2(x)

        m = self.m_linear1(m)
        m = self.m_dropout(F.relu(m))
        m = self.m_linear2(m)

        # x, m = [batch_size, seq_size, d_x]
        return x, m

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_linear1.weight)
        nn.init.xavier_uniform_(self.x_linear2.weight)
        nn.init.xavier_uniform_(self.m_linear1.weight)
        nn.init.xavier_uniform_(self.m_linear2.weight)


"""
Seq2Seq
Idea: Sequence-to-Sequence Transformer Model
__init__:
    Args:
        embedding
        encoder
        decoder
        pad_idx
        d_x
        d_vocab
make_masks:
    Args:
        src: [B, src_seq_len]
        trg: [B, trg_seq_len]
    Outputs:
        src_mask: [B, 1, 1, pad_seq]
        trg_mask: [B, 1, pad_seq, past_seq]
forward:
    Args:
        src, src_ann: [B, src_seq_len]
        trg, trg_ann: [B, trg_seq_len]
    Outputs:
        logits: [B, trg_seq_len, d]

"""
class Seq2Seq(nn.Module):
    def __init__(self, embedding, encoder, decoder, pad_idx,
                 d_x, d_vocab, cat_xm):
        super().__init__()

        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.cat_xm = cat_xm
        self.softmax = nn.LogSoftmax(dim=-1)


    def make_masks(self, src, trg):
        # src = [batch_size, src_seq_size]
        # trg = [batch_size, trg_seq_size]

        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        # trg_mask = [batch_size, 1, trg_seq_size, 1]
        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))

        trg_mask = trg_pad_mask & trg_sub_mask.bool()

        # src_mask = [batch_size, 1, 1, pad_seq]
        # trg_mask = [batch_size, 1, pad_seq, past_seq]
        return src_mask, trg_mask

    def forward(self, src, trg, src_ann=None, trg_ann=None):
        # src, src_ann = [batch_size, src_seq_size]
        # trg, trg_ann = [batch_size, trg_seq_size]

        src_mask, trg_mask = self.make_masks(src, trg)
        # src_mask = [batch_size, 1, 1, pad_seq]
        # trg_mask = [batch_size, 1, pad_seq, past_seq]

        # Use annotations for src_x, trg_x
        if src_ann is not None:
            src_x = src_ann
            src_m = src
        else:
            src_x = src
            src_m = src
        if trg_ann is not None:
            trg_x = trg_ann
            trg_m = trg
        else:
            trg_x = trg
            trg_m = trg

        (src_x, src_emb_x), src_m = self.embedding(src_x, src_m)
        (trg_x, trg_emb_x), trg_m = self.embedding(trg_x, trg_m)
        # src_x, src_m = [batch_size, src_seq_size, p.d_x]
        # trg_x, trg_m = [batch_size, trg_seq_size, p.d_x]


        src_x, src_m, encoder_attn_maps = self.encoder(src_x, src_m, src_mask)
        # src_x, src_m = [batch_size, src_seq_size, p.d_x]

        trg, decoder_attn_maps = self.decoder(trg_x, trg_m, src_x, src_m, trg_mask, src_mask)
        # trg = [batch_size, trg_seq_size, d_x]

        logits = self.embedding.transpose_forward(trg)
        # logits = [batch_size, trg_seq_size, d_vocab]

        if self.cat_xm:
            logits = (self.softmax(logits[0]), self.softmax(logits[1]))
        else:
            logits = self.softmax(logits)

        attn_maps = {
            'Encoder': encoder_attn_maps,
            'Decoder': decoder_attn_maps
        }

        return logits, attn_maps