# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# This file contains code that is adapted from
# https://github.com/black-forest-labs/flux.git
from __future__ import annotations

import math
from dataclasses import dataclass
from torch import Tensor, nn
import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

try:
    from flash_attn import (
        flash_attn_varlen_func
    )
    FLASHATTN_IS_AVAILABLE = True
except ImportError:
    FLASHATTN_IS_AVAILABLE = False
    flash_attn_varlen_func = None

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask: Tensor | None = None, backend = 'pytorch') -> Tensor:
    q, k = apply_rope(q, k, pe)
    if backend == 'pytorch':
        if mask is not None and mask.dtype == torch.bool:
            mask = torch.zeros_like(mask).to(q).masked_fill_(mask.logical_not(), -1e20)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # x = torch.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
        x = rearrange(x, "B H L D -> B L (H D)")
    elif backend == 'flash_attn':
        # q: (B, H, L, D)
        # k: (B, H, S, D) now L = S
        # v: (B, H, S, D)
        b, h, lq, d = q.shape
        _, _, lk, _ = k.shape
        q = rearrange(q, "B H L D -> B L H D")
        k = rearrange(k, "B H S D -> B S H D")
        v = rearrange(v, "B H S D -> B S H D")
        if mask is None:
            q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(q.device, non_blocking=True)
            k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(k.device, non_blocking=True)
        else:
            q_lens = torch.sum(mask[:, 0, :, 0], dim=1).int()
            k_lens = torch.sum(mask[:, 0, 0, :], dim=1).int()
        q = torch.cat([q_v[:q_l] for q_v, q_l in zip(q, q_lens)])
        k = torch.cat([k_v[:k_l] for k_v, k_l in zip(k, k_lens)])
        v = torch.cat([v_v[:v_l] for v_v, v_l in zip(v, k_lens)])
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        max_seqlen_q = q_lens.max()
        max_seqlen_k = k_lens.max()

        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k
        )
        x_list = [x[cu_seqlens_q[i]:cu_seqlens_q[i+1]] for i in range(b)]
        x = pad_sequence(tuple(x_list), batch_first=True)
        x = rearrange(x, "B L H D -> B L (H D)")
    else:
        raise NotImplementedError
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor, mask: Tensor | None = None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe, mask=mask)
        x = self.proj(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, context_dim * 2, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, context: Tensor, pe: Tensor, mask: Tensor | None = None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe, mask=mask)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, backend = 'pytorch'):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.backend = backend

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )




    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, mask: Tensor = None, txt_length = None):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        txt, img = x[:, :txt_length], x[:, txt_length:]

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        if mask is not None:
            mask = repeat(mask, 'B L S->  B H L S', H=self.num_heads)
        attn = attention(q, k, v, pe=pe, mask = mask, backend = self.backend)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        x = torch.cat((txt, img), 1)
        return x


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        backend='pytorch'
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)
        self.backend = backend

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, mask: Tensor = None) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        if mask is not None:
            mask = repeat(mask, 'B L S->  B H L S', H=self.num_heads)
        # compute attention
        attn = attention(q, k, v, pe=pe, mask = mask, backend=self.backend)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class DoubleStreamBlockC(DoubleStreamBlock):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float,
                 qkv_bias: bool = False, backend='pytorch',
                 abondon_cond = False):
        super().__init__(hidden_size, num_heads, mlp_ratio,
                 qkv_bias, backend)
        self.abondon_cond = abondon_cond

    def forward(self, x: Tensor, vec: Tensor,
                pe: Tensor, mask: Tensor = None,
                txt_length=None,
                uncondi_length=None,
                uncondi_pe = None,
                mask_uncond = None):
        # pad_sequence(tuple(x_list), batch_first=True)
        if self.abondon_cond:
            x = [ix[:u_l, :] for ix, u_l in zip(x, uncondi_length)]
            x = pad_sequence(x, batch_first=True)
        if not x.shape[1] == pe.shape[2]:
            pe = uncondi_pe
            mask = mask_uncond
        # print("double stream block", x.shape, pe.shape)
        x = super().forward(x, vec, pe, mask, txt_length)
        return x

class SingleStreamBlockC(SingleStreamBlock):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, hidden_size: int,
                    num_heads: int,
                    mlp_ratio: float = 4.0,
                    qk_scale: float | None = None,
                    backend='pytorch',
                    abondon_cond = False):
        super().__init__(hidden_size, num_heads, mlp_ratio,
                 qk_scale, backend)
        self.abondon_cond = abondon_cond

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, mask: Tensor = None,
                uncondi_length = None, uncondi_pe = None, mask_uncond = None) -> Tensor:
        if self.abondon_cond:
            x = [ix[:u_l, :] for ix, u_l in zip(x, uncondi_length)]
            x = pad_sequence(x, batch_first=True)
        if not x.shape[1] == pe.shape[2]:
            pe = uncondi_pe
            mask = mask_uncond
        # print("single stream block", x.shape, pe.shape)
        x = super().forward(x, vec, pe, mask)
        return x


class DoubleStreamBlockD(DoubleStreamBlock):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float,
                 qkv_bias: bool = False, backend='pytorch'):
        super().__init__(hidden_size, num_heads, mlp_ratio,
                 qkv_bias, backend)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.edit_mod = Modulation(hidden_size, double=True)
        self.edit_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.edit_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.edit_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.edit_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, x: Tensor, vec: Tensor,
                pe: Tensor, mask: Tensor = None,
                txt_length=None,
                edit_length=None):
        if edit_length is not None:
            txt, edit, img = x[:, :txt_length], x[:, txt_length:txt_length + edit_length], x[:, txt_length + edit_length:]
        else:
            txt, img = x[:, :txt_length], x[:, txt_length:]
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        if edit_length is not None:
            edit_mod1, edit_mod2 = self.edit_mod(vec)
            # prepare edit for attention
            edit_modulated = self.edit_norm1(edit)
            edit_modulated = (1 + edit_mod1.scale) * edit_modulated + edit_mod1.shift
            edit_qkv = self.edit_attn.qkv(edit_modulated)
            edit_q, edit_k, edit_v = rearrange(edit_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            edit_q, edit_k = self.edit_attn.norm(edit_q, edit_k, edit_v)
        else:
            edit_q, edit_k, edit_v = None, None, None


        # run actual attention
        q = torch.cat((txt_q,) + ((edit_q,) if edit_q is not None else ()) + (img_q,), dim=2)
        k = torch.cat((txt_k,) + ((edit_k,) if edit_k is not None else ()) + (img_k,), dim=2)
        v = torch.cat((txt_v,) + ((edit_v,) if edit_v is not None else ()) + (img_v,), dim=2)
        if mask is not None:
            mask = repeat(mask, 'B L S->  B H L S', H=self.num_heads)
        attn = attention(q, k, v, pe=pe, mask=mask, backend=self.backend)
        if edit_length is not None:
            txt_attn, edit_attn, img_attn = attn[:, : txt_length], attn[:, txt_length:txt_length + edit_length ], attn[:, txt_length + edit_length:]
        else:
            txt_attn, img_attn = attn[:, : txt_length], attn[:, txt_length:]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        # calculate the img bloks
        if edit_length is not None:
            edit = edit + edit_mod1.gate * self.edit_attn.proj(edit_attn)
            edit = edit + edit_mod2.gate * self.edit_mlp((1 + edit_mod2.scale) * self.edit_norm2(edit) + edit_mod2.shift)
            x = torch.cat((txt, edit, img), 1)
        else:
            x = torch.cat((txt, img), 1)
        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


if __name__ == '__main__':
    pe = EmbedND(dim=64, theta=10000, axes_dim=[16, 56, 56])

    ix_id = torch.zeros(64 // 2, 64 // 2, 3)
    ix_id[..., 1] = ix_id[..., 1] + torch.arange(64 // 2)[:, None]
    ix_id[..., 2] = ix_id[..., 2] + torch.arange(64 // 2)[None, :]
    ix_id = rearrange(ix_id, "h w c -> 1 (h w) c")
    pos = torch.cat([ix_id, ix_id], dim = 1)
    a = pe(pos)

    b = torch.cat([pe(ix_id), pe(ix_id)], dim = 2)

    print(a - b)