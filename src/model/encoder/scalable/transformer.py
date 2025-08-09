import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

try:
    import xformers.ops as xops
except ImportError:
    xops = None


# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)

class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4., mlp_bias=False, 
                 out_features=None, act_layer=nn.GELU, norm_layer=None):
        super().__init__()
        self.norm_exists = norm_layer is not None
        if self.norm_exists:
            self.norm = norm_layer(in_features, bias=False)
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=mlp_bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=mlp_bias)

    def forward(self, x):
        """
        x: (B, L, D)
        Returns: same shape as input 
        """
        if self.norm_exists:
            x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, head_dim=64, qkv_bias=False, qk_scale=None, qk_norm=True, 
                 norm_layer=None):
        super().__init__()
        assert dim % head_dim == 0, 'dim must be divisible by head_dim'
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_exists = norm_layer is not None
        if self.norm_exists:
            self.norm = norm_layer(dim, bias=False)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=False)


    def forward(self, x, support, V):
        """
        x: (B, L, D)
        support: (B, C, D)
        Returns: same shape as input 
        """
        support = rearrange(support, "b (v l) d -> (b v) l d", v=V)
        x = rearrange(x, 'b (v l) d -> (b v) l d', v=V)
        B_s, N_s, C_s = support.shape
        B_x, N_x, C_x = x.shape
        if self.norm_exists:
            x = self.norm(x)

        q = self.q(x).reshape(B_x, N_x, self.num_heads, C_x // self.num_heads)

        k = self.k(support).reshape(B_s, N_s, self.num_heads, C_s // self.num_heads)
        v = self.v(support).reshape(B_s, N_s, self.num_heads, C_s // self.num_heads)

        q, k = self.q_norm(q), self.k_norm(k)
        x = xops.memory_efficient_attention(q, k, v, op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp))
        x = rearrange(x, "(b v) l nh dh -> b (v l) (nh dh)", v=V)

        x = self.proj(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim=64, qkv_bias=False, qk_scale=None, qk_norm=True, 
                 norm_layer=None):
        super().__init__()
        assert dim % head_dim == 0, 'dim must be divisible by head_dim'
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_exists = norm_layer is not None
        if self.norm_exists:
            self.norm = norm_layer(dim, bias=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()


    def forward(self, x):
        """
        x: (B, L, D)
        Returns: same shape as input 
        """

        if self.norm_exists:
            x = self.norm(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 1, 3, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, N, H, C)

        q, k = self.q_norm(q), self.k_norm(k)
        x = xops.memory_efficient_attention(q, k, v, op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp))
        x = rearrange(x, "b l nh dh -> b l (nh dh)")

        x = self.proj(x)
        return x


class ReadBlock(nn.Module):
    def __init__(
        self, dim, head_dim, mlp_ratio=4., 
        mlp_bias=False, qkv_bias=False, qk_scale=None, 
        qk_norm=True, act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, use_flashatt_v2=True):
        super().__init__()

        self.read_cross = CrossAttention(
            dim, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            qk_norm=qk_norm, norm_layer=norm_layer)
        self.read_ff = Mlp(
            in_features=dim, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, 
            act_layer=act_layer, norm_layer=norm_layer)
        

    def forward(self, x, support_tokens, V):
        """
        x: (B, L, D)
        image_tokens: (B, C, D)
        Returns: same shape as input
        """
        
        x = x + self.read_cross(x, support_tokens, V)
        x = x + self.read_ff(x)

        return x
    

class SelfAttnBlock(nn.Module):
    def __init__(
        self, dim, head_dim, mlp_ratio=4., 
        mlp_bias=False, qkv_bias=False, qk_scale=None, 
        qk_norm=True, act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm):
        super().__init__()

        self.attn = SelfAttention(
            dim, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            qk_norm=qk_norm, norm_layer=norm_layer)
        self.mlp = Mlp(
            in_features=dim, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, 
            act_layer=act_layer, norm_layer=norm_layer)
        
    def forward(self, x):
        """
        x: (B, L, D)
        Returns: same shape as input
        """
        
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x