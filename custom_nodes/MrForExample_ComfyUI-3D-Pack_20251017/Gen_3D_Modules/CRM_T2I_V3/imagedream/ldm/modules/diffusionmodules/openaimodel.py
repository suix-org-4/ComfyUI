from abc import abstractmethod
import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from CRM_T2I_V3.imagedream.ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    convert_module_to_f16, 
    convert_module_to_f32
)
from CRM_T2I_V3.imagedream.ldm.modules.attention import (
    SpatialTransformer, 
    SpatialTransformer3D, 
    exists
)
from CRM_T2I_V3.imagedream.ldm.modules.diffusionmodules.adaptors import (
    Resampler, 
    ImageProjModel
)

## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, num_frames=1):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer3D):
                x = layer(x, context, num_frames=num_frames)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, 3, padding=padding
            )

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    "Learned 2x upsampling without padding"

    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(
            self.channels, self.out_channels, kernel_size=ks, stride=2
        )

    def forward(self, x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), True
        )  # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)


class MultiViewUNetModel(nn.Module):
    """
    The full multi-view UNet model with attention, timestep embedding and camera embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :param camera_dim: dimensionality of camera input.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        camera_dim=None,
        with_ip=False,  # wether add image prompt images
        ip_dim=0,  # number of extra token, 4 for global 16 for local
        ip_weight=1.0,  # weight for image prompt context
        ip_mode="local_resample", # which mode of adaptor, global or local
    ):
        super().__init__()
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        self.with_ip = with_ip  # wether there is image prompt
        self.ip_dim = ip_dim  # num of extra token, 4 for global 16 for local
        self.ip_weight = ip_weight
        assert ip_mode in ["global", "local_resample"]
        self.ip_mode = ip_mode  # which mode of adaptor

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if camera_dim is not None:
            time_embed_dim = model_channels * 4
            self.camera_embed = nn.Sequential(
                linear(camera_dim, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        if self.with_ip and (context_dim is not None) and ip_dim > 0:
            if self.ip_mode == "local_resample":
                # ip-adapter-plus
                hidden_dim = 1280
                self.image_embed = Resampler(
                    dim=context_dim,
                    depth=4,
                    dim_head=64,
                    heads=12,
                    num_queries=ip_dim,  # num token
                    embedding_dim=hidden_dim,
                    output_dim=context_dim,
                    ff_mult=4,
                )
            elif self.ip_mode == "global":
                self.image_embed = ImageProjModel(
                    cross_attention_dim=context_dim, 
                    clip_extra_context_tokens=ip_dim)         
            else:
                raise ValueError(f"{self.ip_mode} is not supported")

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer3D(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                                with_ip=self.with_ip,
                                ip_dim=self.ip_dim, 
                                ip_weight=self.ip_weight
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
                
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer3D(  # always uses a self-attn
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
                with_ip=self.with_ip,
                ip_dim=self.ip_dim, 
                ip_weight=self.ip_weight
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or i < num_attention_blocks[level]
                    ):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer3D(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                                with_ip=self.with_ip,
                                ip_dim=self.ip_dim, 
                                ip_weight=self.ip_weight
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
                # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
            )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        camera=None,
        num_frames=1,
        **kwargs,
    ):
        """
        Apply the model to an input batch.
        :param x: an [(N x F) x C x ...] Tensor of inputs. F is the number of frames (views).
        :param timesteps: a 1-D batch of timesteps.
        :param context: a dict conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional, default None.
        :param num_frames: a integer indicating number of frames for tensor reshaping.
        :return: an [(N x F) x C x ...] Tensor of outputs. F is the number of frames (views).
        """
        assert (
            x.shape[0] % num_frames == 0
        ), f"[UNet] input batch size ({x.shape[0]}) must be dividable by num_frames ({num_frames})!"
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False) # shape: torch.Size([B, 320]) mean: 0.18, std: 0.68, min: -1.00, max: 1.00
        emb = self.time_embed(t_emb) # shape: torch.Size([B, 1280]) mean: 0.12, std: 0.57, min: -5.73, max: 6.51

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
            
        # Add camera embeddings
        if camera is not None:
            assert (
                camera.shape[0] == emb.shape[0]
            ), f"[UNet] input camera size ({camera.shape[0]}) must equal to emb size ({emb.shape[0]})!"
            # camera embed: shape: torch.Size([B, 16]) mean: -0.02, std: 0.27, min: -7.23, max: 2.04
            emb = emb + self.camera_embed(camera)
        ip = kwargs.get("ip", None)
        ip_img = kwargs.get("ip_img", None)
        additional_residuals = kwargs.get("additional_residuals", None)

        if ip_img is not None:
            for i_ip in range(ip_img.shape[0]):
                x[num_frames * (i_ip + 1) - 1, :, :, :] = ip_img[i_ip]
            
        if ip is not None:
            ip_emb = self.image_embed(ip) # shape: torch.Size([B, 16, 1024]) mean: -0.00, std: 1.00, min: -11.65, max: 7.31
            context = torch.cat((context, ip_emb), 1).to(dtype=torch.float32) # shape: torch.Size([B, 93, 1024]) mean: -0.00, std: 1.00, min: -11.65, max: 7.31

        h = x.type(self.dtype)
        i = 0
        i_residual = 0
        for module in self.input_blocks:
            #for name, param in module.named_parameters(): 
            #    print(name, param.dtype)
            h = module(h, emb, context, num_frames=num_frames)
            # Add additional feature maps, only add to denoised images not to ip image, becasue it doesn't need to be conditioned
            if additional_residuals is not None and (i+1) % 3 == 0:     # 3 repeated blocks with same feature map size
                batch_add_h = additional_residuals[i_residual]
                i_residual += 1
                
                ip_batch_size = len(batch_add_h)
                add_h_batch_size = batch_add_h[0].shape[0]    # 6 views
                h_all_batch_size = h.shape[0]
                
                i_h_B = 0
                i_residual_B = 0
                while i_h_B < h_all_batch_size:
                    h[i_h_B : i_h_B + add_h_batch_size, :, :, :] += batch_add_h[i_residual_B]
                    i_h_B += add_h_batch_size + 1
                    i_residual_B = (i_residual_B + 1) % ip_batch_size
                    
                    """
                    print(f"[Added Additional Feature Maps:] Original feature maps shape: {h[i_h_B : i_h_B + add_h_batch_size].shape}, Added feature maps shape: {batch_add_h[i_residual_B].shape}")
                    [Added Additional Feature Maps:] Original feature maps shape: torch.Size([6, 320, 32, 32]), Added feature maps shape: torch.Size([6, 320, 32, 32])
                    [Added Additional Feature Maps:] Original feature maps shape: torch.Size([6, 640, 16, 16]), Added feature maps shape: torch.Size([6, 640, 16, 16])
                    [Added Additional Feature Maps:] Original feature maps shape: torch.Size([6, 1280, 8, 8]), Added feature maps shape: torch.Size([6, 1280, 8, 8])
                    [Added Additional Feature Maps:] Original feature maps shape: torch.Size([6, 1280, 4, 4]), Added feature maps shape: torch.Size([6, 1280, 4, 4])
                    """
                
            hs.append(h)
            
            i += 1
            #print(f"[Down Block Feature Map Shape:] {h.shape}")
            # Add T2I-Adapter output feature maps here
            
        """
        All Down Blocks Feature Map Shape: (Input Image Resolution of 256)
            torch.Size([14, 320, 32, 32])
            torch.Size([14, 320, 32, 32])
            torch.Size([14, 320, 32, 32])
            torch.Size([14, 320, 16, 16])
            torch.Size([14, 640, 16, 16])
            torch.Size([14, 640, 16, 16])
            torch.Size([14, 640, 8, 8])
            torch.Size([14, 1280, 8, 8])
            torch.Size([14, 1280, 8, 8])
            torch.Size([14, 1280, 4, 4])
            torch.Size([14, 1280, 4, 4])
            torch.Size([14, 1280, 4, 4])
        Batch size 14 because: 6 views + ip_img with CFG so 7 * 2
        """
            
        h = self.middle_block(h, emb, context, num_frames=num_frames)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, num_frames=num_frames)
        h = h.type(x.dtype) # shape: torch.Size([10, 320, 32, 32]) mean: -0.67, std: 3.96, min: -42.74, max: 25.58
        if self.predict_codebook_ids: # False
            return self.id_predictor(h)
        else:
            return self.out(h) # shape: torch.Size([10, 4, 32, 32]) mean: -0.00, std: 0.91, min: -3.65, max: 3.93
        
class MultiViewUNetModelHyper(nn.Module):
    """
    The full multi-view UNet model with attention, timestep embedding and camera embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :param camera_dim: dimensionality of camera input.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_branches=3,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        camera_dim=None,
        with_ip=False,  # wether add image prompt images
        ip_dim=0,  # number of extra token, 4 for global 16 for local
        ip_weight=1.0,  # weight for image prompt context
        ip_mode="local_resample", # which mode of adaptor, global or local
    ):
        super().__init__()
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dims = dims
        self.num_branches = num_branches
        self.legacy = legacy
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_spatial_transformer = use_spatial_transformer
        self.disable_self_attentions = disable_self_attentions
        self.use_new_attention_order = use_new_attention_order
        self.use_linear_in_transformer = use_linear_in_transformer
        
        self.channel_mult_len = len(channel_mult)
        
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = self.channel_mult_len * [num_res_blocks]
        else:
            if len(num_res_blocks) != self.channel_mult_len:
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == self.channel_mult_len
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.num_attention_blocks = num_attention_blocks
        self.transformer_depth = transformer_depth
        self.context_dim = context_dim

        self.with_ip = with_ip  # wether there is image prompt
        self.ip_dim = ip_dim  # num of extra token, 4 for global 16 for local
        self.ip_weight = ip_weight
        assert ip_mode in ["global", "local_resample"]
        self.ip_mode = ip_mode  # which mode of adaptor

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if camera_dim is not None:
            time_embed_dim = model_channels * 4
            self.camera_embed = nn.Sequential(
                linear(camera_dim, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()
            
        self.time_embed_dim = time_embed_dim

        if self.with_ip and (context_dim is not None) and ip_dim > 0:
            if self.ip_mode == "local_resample":
                # ip-adapter-plus
                hidden_dim = 1280
                self.image_embed = Resampler(
                    dim=context_dim,
                    depth=4,
                    dim_head=64,
                    heads=12,
                    num_queries=ip_dim,  # num token
                    embedding_dim=hidden_dim,
                    output_dim=context_dim,
                    ff_mult=4,
                )
            elif self.ip_mode == "global":
                self.image_embed = ImageProjModel(
                    cross_attention_dim=context_dim, 
                    clip_extra_context_tokens=ip_dim)         
            else:
                raise ValueError(f"{self.ip_mode} is not supported")
            
        # Create Expert Head Branches
        self.expert_head_block = nn.ModuleList([])
        expert_head_block_chans = [model_channels]
        self._feature_size = model_channels * num_branches
        
        for i in range(num_branches):
            expert_head_branch = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(dims, in_channels, model_channels, 3, padding=1)
                    )
                ]
            )
            ds = 1
            ch_in = model_channels
            for nr in range(self.num_res_blocks[0]):
                is_downsample = nr == self.num_res_blocks[0]-1
                layers, ch_in, ds = self.create_input_layer(0, channel_mult[0], nr, ch_in, ds, is_downsample)

                expert_head_branch.append(TimestepEmbedSequential(*layers[:2]))
                if is_downsample:
                    expert_head_branch.append(TimestepEmbedSequential(*layers[2:]))
                    
                self._feature_size += ch_in * 2
                expert_head_block_chans.append(ch_in)
                
            self.expert_head_block.append(expert_head_branch)
            expert_head_block_chans.append(ch_in)
            
        self.expert_head_branch_length = len(self.expert_head_block[0])
        self.expert_head_branch_max_idx = self.expert_head_branch_length - 1
        
        # Create Input Blocks
        self.input_blocks = nn.ModuleList([])
        
        input_block_chans = [model_channels]
        ch = model_channels
        for level in range(1, self.channel_mult_len):
            mult = channel_mult[level]
            is_downsample_level = level < self.channel_mult_len - 1
            for nr in range(self.num_res_blocks[level]):
                is_downsample = is_downsample_level and nr == self.num_res_blocks[level]-1
                layers, ch, ds = self.create_input_layer(level, mult, nr, ch, ds, is_downsample)

                self.input_blocks.append(TimestepEmbedSequential(*layers[:2]))
                if is_downsample:
                    self.input_blocks.append(TimestepEmbedSequential(*layers[2:]))
                    
                self._feature_size += ch * 2
                input_block_chans.append(ch)
                
            if is_downsample:
                input_block_chans.append(ch)

        # Create Middle Block
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer3D(  # always uses a self-attn
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
                with_ip=self.with_ip,
                ip_dim=self.ip_dim, 
                ip_weight=self.ip_weight
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # Create Output Blocks
        self.output_blocks = nn.ModuleList([])
        for level in range(self.channel_mult_len-1, 0, -1):
            mult = channel_mult[level]
            for nr in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers, ch, ds = self.create_output_layer(level, mult, nr, ch, ich, ds, nr == self.num_res_blocks[level])
                
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch * 2
                
        # Create Expert Tail Branches
        self.expert_tail_block = nn.ModuleList([])
        for i in range(num_branches):
            expert_tail_branch = nn.ModuleList([])
            ch_out = ch
            ds_out = ds
            for nr in range(self.num_res_blocks[0] + 1):
                ich = expert_head_block_chans.pop()
                layers, ch_out, ds_out = self.create_output_layer(0, channel_mult[0], nr, ch_out, ich, ds_out, False)

                expert_tail_branch.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch * 2
            
            expert_tail_branch.append(
                TimestepEmbedSequential(
                    nn.Sequential(
                        normalization(ch_out),
                        nn.SiLU(),
                        zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
                    )
                    if not self.predict_codebook_ids
                    else nn.Sequential(
                        normalization(ch_out),
                        conv_nd(dims, model_channels, n_embed, 1),
                        # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
                    )
                )
            )
            
            self.expert_tail_block.append(expert_tail_branch)
            
        self.expert_tail_branch_length = len(self.expert_tail_block[0])
        self.expert_tail_branch_max_idx = self.expert_tail_branch_length - 1
            
    def create_input_layer(self, level, mult, nr, ch, ds, is_downsample):
        layers = [
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                out_channels=mult * self.model_channels,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
            )
        ]
        ch = mult * self.model_channels
        if ds in self.attention_resolutions:
            if self.num_head_channels == -1:
                dim_head = ch // num_heads
            else:
                num_heads = ch // self.num_head_channels
                dim_head = self.num_head_channels
            if self.legacy:
                # num_heads = 1
                dim_head = (
                    ch // num_heads
                    if self.use_spatial_transformer
                    else self.num_head_channels
                )
            if exists(self.disable_self_attentions):
                disabled_sa = self.disable_self_attentions[level]
            else:
                disabled_sa = False

            if (
                not exists(self.num_attention_blocks)
                or nr < self.num_attention_blocks[level]
            ):
                layers.append(
                    AttentionBlock(
                        ch,
                        use_checkpoint=self.use_checkpoint,
                        num_heads=num_heads,
                        num_head_channels=dim_head,
                        use_new_attention_order=self.use_new_attention_order,
                    )
                    if not self.use_spatial_transformer
                    else SpatialTransformer3D(
                        ch,
                        num_heads,
                        dim_head,
                        depth=self.transformer_depth,
                        context_dim=self.context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=self.use_linear_in_transformer,
                        use_checkpoint=self.use_checkpoint,
                        with_ip=self.with_ip,
                        ip_dim=self.ip_dim, 
                        ip_weight=self.ip_weight
                    )
                )
                
        if is_downsample:
            layers.append(
                ResBlock(
                    ch,
                    self.time_embed_dim,
                    self.dropout,
                    out_channels=ch,
                    dims=self.dims,
                    use_checkpoint=self.use_checkpoint,
                    use_scale_shift_norm=self.use_scale_shift_norm,
                    down=True,
                )
                if self.resblock_updown
                else Downsample(
                    ch, self.conv_resample, dims=self.dims, out_channels=ch
                )
            )
            ds *= 2
                
        return layers, ch, ds
    
    def create_output_layer(self, level, mult, nr, ch, ich, ds, is_upsample):
        layers = [
            ResBlock(
                ch + ich,
                self.time_embed_dim,
                self.dropout,
                out_channels=self.model_channels * mult,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
            )
        ]
        ch = self.model_channels * mult
        if ds in self.attention_resolutions:
            if self.num_head_channels == -1:
                dim_head = ch // num_heads
            else:
                num_heads = ch // self.num_head_channels
                dim_head = self.num_head_channels
            if self.legacy:
                # num_heads = 1
                dim_head = (
                    ch // num_heads
                    if self.use_spatial_transformer
                    else self.num_head_channels
                )
            if exists(self.disable_self_attentions):
                disabled_sa = self.disable_self_attentions[level]
            else:
                disabled_sa = False

            if (
                not exists(self.num_attention_blocks)
                or nr < self.num_attention_blocks[level]
            ):
                layers.append(
                    AttentionBlock(
                        ch,
                        use_checkpoint=self.use_checkpoint,
                        num_heads=self.num_heads_upsample,
                        num_head_channels=dim_head,
                        use_new_attention_order=self.use_new_attention_order,
                    )
                    if not self.use_spatial_transformer
                    else SpatialTransformer3D(
                        ch,
                        num_heads,
                        dim_head,
                        depth=self.transformer_depth,
                        context_dim=self.context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=self.use_linear_in_transformer,
                        use_checkpoint=self.use_checkpoint,
                        with_ip=self.with_ip,
                        ip_dim=self.ip_dim, 
                        ip_weight=self.ip_weight
                    )
                )
                
        if is_upsample:
            layers.append(
                ResBlock(
                    ch,
                    self.time_embed_dim,
                    self.dropout,
                    out_channels=ch,
                    dims=self.dims,
                    use_checkpoint=self.use_checkpoint,
                    use_scale_shift_norm=self.use_scale_shift_norm,
                    up=True,
                )
                if self.resblock_updown
                else Upsample(ch, self.conv_resample, dims=self.dims, out_channels=ch)
            )
            ds //= 2
                
        return layers, ch, ds

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.expert_head_block.apply(convert_module_to_f16)
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.expert_tail_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.expert_head_block.apply(convert_module_to_f32)
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.expert_tail_block.apply(convert_module_to_f32)
        
    def add_residuals(self, h, additional_residuals, i_residual):
        batch_add_h = additional_residuals[i_residual]
        
        ip_batch_size = len(batch_add_h)
        add_h_batch_size = batch_add_h[0].shape[0]    # 6 views
        h_all_batch_size = h.shape[0]
        
        i_h_B = 0
        i_residual_B = 0
        while i_h_B < h_all_batch_size:
            h[i_h_B : i_h_B + add_h_batch_size, :, :, :] += batch_add_h[i_residual_B]
            i_h_B += add_h_batch_size + 1
            i_residual_B = (i_residual_B + 1) % ip_batch_size
            
            """
            print(f"[Added Additional Feature Maps:] Original feature maps shape: {h[i_h_B : i_h_B + add_h_batch_size].shape}, Added feature maps shape: {batch_add_h[i_residual_B].shape}")
            [Added Additional Feature Maps:] Original feature maps shape: torch.Size([6, 320, 32, 32]), Added feature maps shape: torch.Size([6, 320, 32, 32])
            [Added Additional Feature Maps:] Original feature maps shape: torch.Size([6, 640, 16, 16]), Added feature maps shape: torch.Size([6, 640, 16, 16])
            [Added Additional Feature Maps:] Original feature maps shape: torch.Size([6, 1280, 8, 8]), Added feature maps shape: torch.Size([6, 1280, 8, 8])
            [Added Additional Feature Maps:] Original feature maps shape: torch.Size([6, 1280, 4, 4]), Added feature maps shape: torch.Size([6, 1280, 4, 4])
            """

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        camera=None,
        num_frames=1,
        **kwargs,
    ):
        """
        Apply the model to an input batch.
        :param x: an [(N x F) x C x ...] Tensor of inputs. F is the number of frames (views).
        :param timesteps: a 1-D batch of timesteps.
        :param context: a dict conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional, default None.
        :param num_frames: a integer indicating number of frames for tensor reshaping.
        :return: an [(N x F) x C x ...] Tensor of outputs. F is the number of frames (views).
        """
        assert (
            x.shape[0] % num_frames == 0
        ), f"[UNet] input batch size ({x.shape[0]}) must be dividable by num_frames ({num_frames})!"
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False) # shape: torch.Size([B, 320]) mean: 0.18, std: 0.68, min: -1.00, max: 1.00
        emb = self.time_embed(t_emb) # shape: torch.Size([B, 1280]) mean: 0.12, std: 0.57, min: -5.73, max: 6.51

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
            
        # Add camera embeddings
        if camera is not None:
            assert (
                camera.shape[0] == emb.shape[0]
            ), f"[UNet] input camera size ({camera.shape[0]}) must equal to emb size ({emb.shape[0]})!"
            # camera embed: shape: torch.Size([B, 16]) mean: -0.02, std: 0.27, min: -7.23, max: 2.04
            emb = emb + self.camera_embed(camera)
        ip = kwargs.get("ip", None)
        ip_img = kwargs.get("ip_img", None)
        additional_residuals = kwargs.get("additional_residuals", None)

        if ip_img is not None:
            ip_img = ip_img.repeat(1, 3, 1, 1)
            for i_ip in range(ip_img.shape[0]):
                x[num_frames * (i_ip + 1) - 1, :, :, :] = ip_img[i_ip]
            
        if ip is not None:
            ip_emb = self.image_embed(ip) # shape: torch.Size([B, 16, 1024]) mean: -0.00, std: 1.00, min: -11.65, max: 7.31
            context = torch.cat((context, ip_emb), 1).to(dtype=torch.float32) # shape: torch.Size([B, 93, 1024]) mean: -0.00, std: 1.00, min: -11.65, max: 7.31

        # Expert head branches
        h_x = x.type(self.dtype)
        h_x = h_x.chunk(self.num_branches, dim=1)
        h = 0.
        i_residual = 0
        for i_branch in range(self.num_branches):
            expert_head_branch = self.expert_head_block[i_branch]
            h_in = h_x[i_branch]
            all_h_in = []
            for i in range(self.expert_head_branch_length):
                h_in = expert_head_branch[i](h_in, emb, context, num_frames=num_frames)
                if i < self.expert_head_branch_max_idx:
                    all_h_in.append(h_in)
                
                if additional_residuals is not None and (i+1) % 3 == 0:
                    self.add_residuals(h_in, additional_residuals, i_residual)
            
            hs = all_h_in + hs
            h += h_in
            
        h /= self.num_branches
        hs.append(h)
        # Unet input blocks
        i = 1
        i_residual = 1
        for module in self.input_blocks:
            #for name, param in module.named_parameters(): 
            #    print(name, param.dtype)
            h = module(h, emb, context, num_frames=num_frames)
            # Add additional feature maps, only add to denoised images not to ip image, becasue it doesn't need to be conditioned
            if additional_residuals is not None and (i+1) % 3 == 0:     # 3 repeated blocks with same feature map size
                self.add_residuals(h, additional_residuals, i_residual)
                i_residual += 1
            
            hs.append(h)
            
            i += 1
            #print(f"[Down Block Feature Map Shape:] {h.shape}")
            # Add T2I-Adapter output feature maps here
            
        """
        All Down Blocks Feature Map Shape: (Input Image Resolution of 256)
            torch.Size([14, 320, 32, 32])
            torch.Size([14, 320, 32, 32])
            torch.Size([14, 320, 32, 32])
            torch.Size([14, 320, 16, 16])
            torch.Size([14, 640, 16, 16])
            torch.Size([14, 640, 16, 16])
            torch.Size([14, 640, 8, 8])
            torch.Size([14, 1280, 8, 8])
            torch.Size([14, 1280, 8, 8])
            torch.Size([14, 1280, 4, 4])
            torch.Size([14, 1280, 4, 4])
            torch.Size([14, 1280, 4, 4])
        Batch size 14 because: 6 views + ip_img with CFG so 7 * 2
        """
        
        # Unet middle blocks
        h = self.middle_block(h, emb, context, num_frames=num_frames)
        # Unet output blocks
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, num_frames=num_frames)  # shape: torch.Size([10, 320, 32, 32]) mean: -0.67, std: 3.96, min: -42.74, max: 25.58
        
        # Expert tail branches
        results = []
        for expert_tail_branch in self.expert_tail_block:
            h_out = h
            for i in range(self.expert_tail_branch_length):
                if i < self.expert_tail_branch_max_idx:
                    h_out = th.cat([h_out, hs.pop()], dim=1)
                h_out = expert_tail_branch[i](h_out, emb, context, num_frames=num_frames)
            h_out = h_out.type(x.dtype)
            results.append(h_out)    # shape: torch.Size([10, 4, 32, 32]) mean: -0.00, std: 0.91, min: -3.65, max: 3.93

        #f_test = 0.
        #for h_out in results:
        #    f_test += h_out
        #f_test /= len(results)
        
        return results

class MultiViewUNetModelStage2(MultiViewUNetModel):
    """
    The full multi-view UNet model with attention, timestep embedding and camera embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :param camera_dim: dimensionality of camera input.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        camera_dim=None,
        with_ip=False,  # wether add image prompt images
        ip_dim=0,  # number of extra token, 4 for global 16 for local
        ip_weight=1.0,  # weight for image prompt context
        ip_mode="local_resample", # which mode of adaptor, global or local
    ):
        super().__init__(
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            num_classes,
            use_checkpoint,
            use_fp16,
            use_bf16,
            num_heads,
            num_head_channels,
            num_heads_upsample,
            use_scale_shift_norm,
            resblock_updown,
            use_new_attention_order,
            use_spatial_transformer,
            transformer_depth,
            context_dim,
            n_embed,
            legacy,
            disable_self_attentions,
            num_attention_blocks,
            disable_middle_self_attn,
            use_linear_in_transformer,
            adm_in_channels,
            camera_dim,
            with_ip,
            ip_dim,
            ip_weight,
            ip_mode,
        )

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        camera=None,
        num_frames=1,
        **kwargs,
    ):
        """
        Apply the model to an input batch.
        :param x: an [(N x F) x C x ...] Tensor of inputs. F is the number of frames (views).
        :param timesteps: a 1-D batch of timesteps.
        :param context: a dict conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional, default None.
        :param num_frames: a integer indicating number of frames for tensor reshaping.
        :return: an [(N x F) x C x ...] Tensor of outputs. F is the number of frames (views).
        """
        assert (
            x.shape[0] % num_frames == 0
        ), "[UNet] input batch size must be dividable by num_frames!"
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False) # shape: torch.Size([B, 320]) mean: 0.18, std: 0.68, min: -1.00, max: 1.00
        emb = self.time_embed(t_emb) # shape: torch.Size([B, 1280]) mean: 0.12, std: 0.57, min: -5.73, max: 6.51

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
            
        # Add camera embeddings
        if camera is not None:
            assert camera.shape[0] == emb.shape[0]
            # camera embed: shape: torch.Size([B, 1280]) mean: -0.02, std: 0.27, min: -7.23, max: 2.04
            emb = emb + self.camera_embed(camera)
        ip = kwargs.get("ip", None)
        ip_img = kwargs.get("ip_img", None)
        pixel_images = kwargs.get("pixel_images", None)

        if ip_img is not None:
            x[(num_frames-1)::num_frames, :, :, :] = ip_img
            
        x = torch.cat((x, pixel_images), dim=1)
            
        if ip is not None:
            ip_emb = self.image_embed(ip) # shape: torch.Size([B, 16, 1024]) mean: -0.00, std: 1.00, min: -11.65, max: 7.31
            context = torch.cat((context, ip_emb), 1) # shape: torch.Size([B, 93, 1024]) mean: -0.00, std: 1.00, min: -11.65, max: 7.31

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context, num_frames=num_frames)
            hs.append(h)
        h = self.middle_block(h, emb, context, num_frames=num_frames)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, num_frames=num_frames)
        h = h.type(x.dtype) # shape: torch.Size([10, 320, 32, 32]) mean: -0.67, std: 3.96, min: -42.74, max: 25.58
        if self.predict_codebook_ids: # False
            return self.id_predictor(h)
        else:
            return self.out(h) # shape: torch.Size([10, 4, 32, 32]) mean: -0.00, std: 0.91, min: -3.65, max: 3.93
        