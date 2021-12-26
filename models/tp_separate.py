# Experimental architecture testing the idea of maintaining a separation
# between roles and fillers throughout the entire architecture (i.e. no binding)

import os
import math
import pdb
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function


def build_tp_sep_transformer(params, pad_idx):
  d_vocab = params.d_vocab
  d_pos = 200 # max input size

  d_f = params.filter

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
  use_xv = params.use_xv # use separate value vectors for x (rather than just keys)
  use_adversary = params.use_adversary # use "lexical adversary"
  adv_lambda = params.adv_lambda # scale of gradients from adversary
  adv_theta = params.adv_theta # minimum loss to backpropagate to adversary
  adv_lr = params.adv_lr # learning rate for adversary

  embedding = EmbeddingMultilinearSinusoidal(d_vocab=d_vocab,
                                             d_x=d_x,
                                             d_r=d_r,
                                             dropout=dropout,
                                             max_length=200)
  encoder = Encoder(
      d_x,
      d_q,
      d_k,
      d_v,
      d_f,
      n_I,
      n_L,
      use_xv,
      dropout
  )
  decoder = Decoder(
      d_x,
      d_q,
      d_k,
      d_v,
      d_f,
      use_xv,
      cat_xm,
      n_I,
      n_L,
      dropout
  )
  model = Seq2Seq(embedding=embedding,
                  encoder=encoder,
                  decoder=decoder,
                  pad_idx=pad_idx,
                  use_adversary=use_adversary,
                  d_x=d_x,
                  d_vocab=d_vocab,
                  adv_lambda=adv_lambda,
                  adv_theta=adv_theta)

  return model

# Scale adversary gradient
class GradientReverse(Function):
  scale = 1.0
  @staticmethod
  def forward(ctx, x):
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    return GradientReverse.scale * grad_output.neg()

def grad_reverse(x, scale=1.0):
  GradientReverse.scale = scale
  return GradientReverse.apply(x)

"""
Lexical Adversary
Idea: penalize embeddings for having information about token identity
__init__
    Args:
        d_x
        d_vocab
        adv_lambda
forward:
    Args:
        emb_x: [B, seq_len, d_x]
    Output:
        logits: [B, seq_len, d_vocab]

"""
class Adversary(nn.Module):
  def __init__(self, d_x, d_vocab, adv_lambda):
    super(Adversary, self).__init__()
    self.d_x = d_x
    self.d_vocab = d_vocab
    self.lam = adv_lambda

    self.linear = nn.Linear(self.d_x, self.d_vocab, bias=False)

  def forward(self, emb_x):
    emb_x = grad_reverse(emb_x, self.lam) # [batch_size, seq_len, d_x]
    logits = self.linear(emb_x) # [batch_size, seq_len, d_vocab]
    return logits


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
  def __init__(self, d_vocab, d_x, d_r, dropout, max_length):
    super(EmbeddingMultilinearSinusoidal, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.max_length = max_length
    self.d_x = d_x

    # token encodings
    self.x_embedding = nn.Embedding(d_vocab, d_x)
    self.m_embedding = nn.Embedding(d_vocab, d_x)
    self.scale = torch.sqrt(torch.FloatTensor([d_x]))

    # sinusoidal encoding
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
    pos_sin_emb = torch.autograd.Variable(self.pe[:, :x.size(1)],
                                          requires_grad=False)
    x = emb_x + pos_sin_emb # 0.5 * pos_sin_emb + 0.5 * pos_id_emb
    # x = [batch_size, src_seq_len, d_x]

    r = self.linear(x) + 1 # ~N(1,1)
    # r = [batch_size, src_seq_len, d_r]

    x = x * r
    x = self.dropout(x)
    # x = [batch_size, src_seq_len, d_x]
    return (x, emb_x), emb_m

  def transpose_forward(self, trg):
    # trg = [batch_size, trg_seq_len, d_v, d_r]

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
        use_xv
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
  def __init__(self, d_x, d_q, d_k, d_v, d_f, n_I, n_L, use_xv, dropout):
    super().__init__()

    layers = [EncoderLayer(
        d_x,
        d_q,
        d_k,
        d_v,
        d_f,
        n_I,
        use_xv,
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
        use_xv,
        dropout
      ))
    self.layers = nn.ModuleList(layers)

  def forward(self, src_x, src_m, src_mask):
    # src_x = [batch_size, src_seq_len, p.d_x]
    # src_m = [batch_size, src_seq_len, p.d_x]
    # src_mask = [batch_size, 1, attn_size]
    for layer in self.layers:
      src_x, src_m = layer(src_x, src_m, src_mask)

    return src_x, src_m


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
        use_xv
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
  def __init__(self, d_x, d_q, d_k, d_v, d_f, n_I, use_xv, dropout):
    super().__init__()

    # sublayer 1
    self.x_layernorm1 = nn.LayerNorm(d_x)
    self.m_layernorm1 = nn.LayerNorm(d_x)
    self.MHA = SelfAttention(d_x, d_q, d_k, d_v, n_I, use_xv, dropout)
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
    x, m = self.MHA(x, x, m, src_mask)
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

    return src_x, src_m

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
        use_xv
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
      use_xv,
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
        use_xv,
        n_I,
        dropout
    ) for _ in range(n_L)])
    if self.cat_xm:
      self.out = nn.Linear(2*d_x, d_x)

  def forward(self, trg_x, trg_m, src_x, src_m, trg_mask, src_mask):
    # trg_x, trg_m = [batch_size, trg_seq_size, d_x]
    # src_x, src_m = [batch_size, src_seq_size, d_x]
    # trg_mask = [batch_size, trg_seq_size]
    # src_mask = [batch_size, src_seq_size]

    for layer in self.layers:
      trg_x, trg_m = layer(trg_x, trg_m, src_x, src_m, trg_mask, src_mask)

    if self.cat_xm:
      trg = torch.cat([trg_x,trg_m], dim=2)
      trg = self.out(trg)
    else:
      trg = trg_m

    return trg


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
        use_xv
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
  def __init__(self, d_x, d_q, d_k, d_v, d_f, use_xv, n_I, dropout):
    super().__init__()
    # sublayer 1
    self.x_layernorm1 = nn.LayerNorm(d_x)
    self.m_layernorm1 = nn.LayerNorm(d_x)
    self.selfAttn = SelfAttention(d_x, d_q, d_k, d_v, d_v, n_I, use_xv, dropout)
    self.x_dropout1 = nn.Dropout(dropout)
    self.m_dropout1 = nn.Dropout(dropout)
    # sublayer 2
    self.x_layernorm2 = nn.LayerNorm(d_x)
    self.m_layernorm2 = nn.LayerNorm(d_x)
    self.encAttn = SelfAttention(d_x, d_q, d_k, d_v, d_v, n_I, use_xv, dropout)
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

    # output
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
    x, m = self.selfAttn(x, x, m, trg_mask)
    x = self.x_dropout1(x)
    m = self.m_dropout1(m)
    trg_x = self.x_layernorm2(trg_x + x)
    trg_m = self.m_layernorm2(trg_m + m)

    # encoder attention
    x, m = self.encAttn(x, src_x, src_m, src_mask)
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

    return trg_x, trg_m

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
        use_xv
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
  def __init__(self, d_x, d_q, d_k, d_v, n_I, use_xv, dropout):
    super().__init__()
    self.d_x = d_x
    self.d_q = d_q
    self.d_k = d_k
    self.d_v = d_v
    self.n_I = n_I
    self.use_xv = use_xv # use separate value vectors for x (rather than keys)

    self.W_q = nn.Linear(self.d_x, d_q * n_I)
    self.W_k = nn.Linear(self.d_x, d_k * n_I)
    self.W_v = nn.Linear(self.d_x, d_v * n_I)
    if self.use_xv:
      self.W_xv = nn.Linear(self.d_x, d_v * n_I)

    self.W_xo = nn.Linear(d_v * n_I, d_x)
    self.W_mo = nn.Linear(d_k * n_I, d_x)

    self.dropout = nn.Dropout(dropout)
    self.dot_scale = torch.FloatTensor([math.sqrt(d_k)])
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

    dot = torch.einsum('bhid,bhjd->bhij', Q, K) / self.dot_scale.to(key.device)
    # energy   = [batch_size, n_heads, query_pos     , key_pos]
    # src_mask = [batch_size, 1      , 1             , attn]
    # trg_mask = [batch_size, 1      , query_specific, attn]

    if mask is not None:
      dot = dot.masked_fill(mask == 0, -1e10)

    attention = self.dropout(F.softmax(dot, dim=-1))
    # attention = [batch_size, n_heads, seq_size, seq_size]

    if self.use_xv:
      xV = self.W_xv(key)
      xV = xV.view(bsz, -1, self.n_I, self.d_v).permute(0,2,1,3)
      K = xV # use xV rather than K for values associated with x
      # K = [batch_size, n_heads, seq_size, d_v]

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

    return x, m

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
        use_adversary
        d_x
        d_vocab
        adv_lambda
        adv_theta
        adv_lr
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
        #TODO: adv_stat
train_adversary:
    #TODO: complete this
test_adversary:
    #TODO: complete this

"""
class Seq2Seq(nn.Module):
  def __init__(self, embedding, encoder, decoder, pad_idx, 
               use_adversary, d_x, d_vocab, adv_lambda, adv_theta,
               adv_lr):
    super().__init__()

    self.embedding = embedding
    self.encoder = encoder
    self.decoder = decoder
    self.pad_idx = pad_idx
    self.use_adversary = use_adversary

    # Adversary (optional)
    if self.use_adversary:
      self.adversary = Adversary(
          d_x,
          d_vocab,
          adv_lambda
      )
      self.adv_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
      self.adv_theta = adv_theta
      self.adv_optimizer = torch.optim.Adam(self.adversary.parameters(), adv_lr)


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

    if self.adversary:
      if src_emb_x.requires_grad: # no adversary training during evaluation
        adv_stat = self.train_adversary(src_emb_x, trg_emb_x, src, trg)
      else:
        adv_stat = self.test_adversary(src_emb_x, trg_emb_x, src, trg)
    else:
      adv_stat = None

    src_x, src_m = self.encoder(src_x, src_m, src_mask)
    # src_x, src_m = [batch_size, src_seq_size, p.d_x]

    trg = self.decoder(trg_x, trg_m, src_x, src_m, trg_mask, src_mask)
    # trg = [batch_size, trg_seq_size, d_x]

    logits = self.embedding.transpose_forward(trg)
    # logits = [batch_size, trg_seq_size, d_vocab]

    return logits, adv_stat

  def train_adversary(self, src_emb_x, trg_emb_x, src, trg):
    # src_emb_x = [batch_size, src_seq_len, d_x]
    # trg_emb_x = [batch_size, trg_seq_len, d_x]
    # src = [batch_size, src_seq_len]
    # trg = [batch_size, trg_seq_len]

    # Forward
    src_logits = self.adversary(src_emb_x) # [batch_size, src_seq_len, d_vocab]
    trg_logits = self.adversary(trg_emb_x) # [batch_size, trg_seq_len, d_vocab]

    # Loss
    src = src.reshape(-1) # [batch_size*src_seq_len]
    trg = trg.reshape(-1) # [batch_size*trg_seq_len]
    src_logits = src_logits.contiguous().view(-1, src_logits.shape[-1])
    trg_logits = trg_logits.contiguous().view(-1, trg_logits.shape[-1])
    src_loss = self.adv_loss(src_logits, src)
    trg_loss = self.adv_loss(trg_logits, trg)
    loss = src_loss + trg_loss

    # Backward
    loss.backward(retain_graph=True) # retain graph because model will have its own loss
    if loss.data.item() > self.adv_theta:
      self.adv_optimizer.step()
    self.adv_optimizer.zero_grad()

    return loss.data.item()

  def test_adversary(self, src_emb_x, trg_emb_x, src, trg):
    # src_emb_x = [batch_size, src_seq_len, d_x]
    # trg_emb_x = [batch_size, trg_seq_len, d_x]
    # src = [batch_size, src_seq_len]
    # trg = [batch_size, trg_seq_len]

    # Predictions
    x = torch.cat([src_emb_x, trg_emb_x], dim=1) # [batch_size, src_seq_len + trg_seq_len, d_x]
    logits = self.adversary(x) # [batch_size, src_seq_len + trg_seq_len, d_vocab]
    y_hat = torch.argmax(logits, dim=-1) # [batch_size, src_seq_len + trg_seq_len]

    # Correct
    y = torch.cat([src, trg], dim=1) # [batch_size, src_seq_len + trg_seq_len]
    y_masked = y[y != self.pad_idx] # remove padding
    y_hat_masked = y_hat[y != self.pad_idx] # remove padding
    matches = torch.eq(y_masked, y_hat_masked)
    matches = matches.view(-1).cpu().numpy().tolist() # [batch_size*(src_seq_len + trg_seq_len)]

    return matches
