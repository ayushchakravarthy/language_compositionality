import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from einops import rearrange


def apply_pos(tensor, pos, num_heads, is_class):
    if pos is None:
        return tensor
    elif is_class:
        # return pos(tensor)
        return tensor + pos._get_pe(tensor)
    elif len(tensor.shape) != len(pos.shape):
        tensor = rearrange(tensor, "b n (g c) -> b n g c", g=num_heads)
        tensor = tensor + pos
        tensor = rearrange(tensor, "b n g c -> b n (g c)")
    else:
        tensor = tensor + pos
    return tensor

class SimpleReasoning(nn.Module):
    def __init__(self, np, dim):
        super(SimpleReasoning, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Conv1d(np, np, kernel_size=1, bias=False)
    
    def forward(self, x):
        tokens = self.norm(x)
        tokens = self.linear(tokens)
        return x + tokens

class AnyAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False):
        super(AnyAttention, self).__init__()
        self.norm_q, self.norm_k, self.norm_v = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = (dim / num_heads) ** (-0.5)
        self.num_heads = num_heads
        self.proj = nn.Linear(dim, dim)

    def get_qkv(self, q, k, v, qpos, kpos, is_class):
        q = apply_pos(q, qpos, self.num_heads, is_class)
        k = apply_pos(k, kpos, self.num_heads, is_class=False)
        v = apply_pos(v, None, 0, is_class=False)
        q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)

        return q, k, v
    
    def forward(self, q=None, k=None, v=None, qpos=None, kpos=None, mask=None, rel_pos=None, is_class=None):
        q, k, v = self.get_qkv(q, k, v, qpos, kpos, is_class)

        # reshape
        q = rearrange(q, "b n (g c) -> b n g c", g=self.num_heads)
        k = rearrange(k, "b n (g c) -> b n g c", g=self.num_heads)
        v = rearrange(v, "b n (g c) -> b n g c", g=self.num_heads)

        # compute attention matrix
        attn = torch.einsum("b q g c, b k g c -> b q g k", q, k)
        if rel_pos is not None:
            attn = rel_pos(q, attn)
        attn *= self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), value=float('-inf'))
        attn = F.softmax(attn, dim=-1)
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), value=0)
        out = torch.einsum("b q g k, b k g c -> b q g c", attn, v.float())
        out = rearrange(out, "b q g c -> b q (g c)")
        out = self.proj(out)
        return out, attn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x