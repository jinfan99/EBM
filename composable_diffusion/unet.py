import math
from abc import abstractmethod
from collections import OrderedDict

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import avg_pool_nd, conv_nd, linear, normalization, timestep_embedding, zero_module


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

    def forward(self, x, emb, encoder_out=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, encoder_out)
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

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
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
            normalization(channels, swish=1.0),
            nn.Identity(),
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
            normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
            nn.SiLU() if use_scale_shift_norm else nn.Identity(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).to(h.dtype)
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
        encoder_channels=None,
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
        self.norm = normalization(channels, swish=0.0)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, encoder_out=None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).to(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

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
    """

    def __init__(
        self,
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
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        encoder_channels=None,
        dataset=''
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dataset = dataset
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes:
            print("Using class-conditional UNet")
            layer_classes = self.num_classes.split(',')
            if self.dataset in {'coco', 'ade20k', 'flowers102', 'cup200', 'ffhq_sub', 'online', 'birds_flowers', 'celeba'}:
                assert len(layer_classes) == 1
                self.label_emb = nn.Embedding(int(layer_classes[0]), self.time_embed_dim)
            elif self.dataset == 'ffhq':
                self.label_emb = nn.ModuleList(
                    [nn.Linear(1, self.time_embed_dim) if int(classes) == 73
                     else nn.Embedding(int(classes), self.time_embed_dim)
                     for classes in layer_classes]
                )
                # unconditional label
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            elif self.dataset == 'clevr_pos':
                self.label_emb = nn.Linear(int(layer_classes[0]), self.time_embed_dim)
                # unconditional label
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            elif self.dataset == 'clevr_rel':
                attr_classes = [int(x) for x in layer_classes[:-1]]
                rel_class = int(layer_classes[-1])
                self.obj_emb = nn.ModuleList(
                    [nn.Embedding(num_classes, self.time_embed_dim // len(attr_classes))
                     for num_classes in attr_classes]
                )
                assert self.time_embed_dim // 3 == int(self.time_embed_dim / 3)
                self.fc = nn.Linear(self.time_embed_dim // len(attr_classes) * len(attr_classes),
                                    self.time_embed_dim // 3)
                self.rel_emb = nn.Embedding(rel_class, self.time_embed_dim // 3)
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            else:
                raise NotImplementedError

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
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
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

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
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
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
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                if level and i == num_res_blocks:
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
            normalization(ch, swish=1.0),
            nn.Identity(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        self.use_fp16 = use_fp16

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

    def forward(self, x, timesteps, y=None, masks=None, layer_idx=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param masks: an [N] Tensor of booleans
        :param layer_idx: integer indicates which attribute layer is used
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes:
            # print("Using class-conditional UNet")
            # assert y.shape == (x.shape[0],)   # assert for discrete class labels
            if self.dataset in {'coco', 'ade20k', 'flowers102', 'cup200',
                                'ffhq_sub', 'online', 'birds_flowers', 'celeba'}:
                assert layer_idx is None
                label_emb = self.label_emb(y.long())
                emb = emb + label_emb
            elif self.dataset == 'ffhq':
                assert layer_idx is not None
                label_emb = th.empty((y.shape[0], emb.shape[1]), device=y.device)
                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
                if layer_idx == 2:
                    # age 0-73
                    normalized_y = y[masks].reshape(-1, 1) / 73.
                    label_emb[masks] = self.label_emb[layer_idx](normalized_y)
                else:
                    label_emb[masks] = self.label_emb[layer_idx](y[masks])
                emb = emb + label_emb
            elif self.dataset == 'clevr_pos':
                label_emb = th.empty((y.shape[0], self.time_embed_dim), device=y.device)
                label_emb[masks] = self.label_emb(y[masks])
                # replace null labels with special embeddings
                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
                emb = emb + label_emb
            elif self.dataset == 'clevr_rel':
                assert masks is not None, "masks are not provided for training relational clevr data."
                label_emb = th.empty((y.shape[0], emb.shape[1]), device=y.device)
                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
                # embed objects and relations
                obj_emb_1 = self.fc(th.cat([self.obj_emb[i](y[masks][:, i]) for i in range(3)], dim=1))
                obj_emb_2 = self.fc(th.cat([self.obj_emb[i-3](y[masks][:, i]) for i in range(3, 6)], dim=1))
                rel_emb = self.rel_emb(y[masks][:, -1])
                label_emb[masks] = th.cat((obj_emb_1, obj_emb_2, rel_emb), dim=1)
                emb = emb + label_emb
            else:
                raise NotImplementedError

        h = x.to(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.to(x.dtype)
        return self.out(h)




class UNetModel_full(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
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
    """

    def __init__(
            self,
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
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            encoder_channels=None,
            dataset=''
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dataset = dataset
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes:
            layer_classes = self.num_classes.split(',')
            if self.dataset == 'clevr_pos':
                self.label_emb = nn.Linear(int(layer_classes[0]), self.time_embed_dim)
                # unconditional label
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            elif self.dataset== "clevr_norel":
                self.label_emb = nn.Embedding(int(layer_classes[0]),self.time_embed_dim)
                self.null_emb = nn.Embedding(1, self.time_embed_dim)

            elif self.dataset == 'clevr_rel':
                attr_classes = [int(x) for x in layer_classes[:-1]]
                rel_class = int(layer_classes[-1])
                self.obj_emb = nn.ModuleList(
                    [nn.Embedding(num_classes, self.time_embed_dim // len(attr_classes))
                     for num_classes in attr_classes]
                )
                assert self.time_embed_dim // 3 == int(self.time_embed_dim / 3)
                self.fc = nn.Linear(self.time_embed_dim // len(attr_classes) * len(attr_classes),
                                    self.time_embed_dim // 3)
                self.rel_emb = nn.Embedding(rel_class, self.time_embed_dim // 3)
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            elif self.dataset == 'clevr_count':
                attr_classes = [int(x) for x in layer_classes]
                self.obj_emb = nn.ModuleList(
                    [nn.Embedding(num_classes, self.time_embed_dim // len(attr_classes))
                     for num_classes in attr_classes]
                )
                assert self.time_embed_dim // 3 == int(self.time_embed_dim / 3)
                self.fc = nn.Linear(self.time_embed_dim // len(attr_classes) * len(attr_classes),
                                    self.time_embed_dim)
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            else:
                raise NotImplementedError

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
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
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

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
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
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
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                if level and i == num_res_blocks:
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
            normalization(ch, swish=1.0),
            nn.Identity(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        self.use_fp16 = use_fp16

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

    def forward(self, x, timesteps, y=None, masks=None, layer_idx=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param masks: an [N] Tensor of booleans
        :param layer_idx: integer indicates which attribute layer is used
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes:
            # assert y.shape == (x.shape[0],)   # assert for discrete class labels
            if self.dataset == 'clevr_pos':
                label_emb = th.empty((y.shape[0], self.time_embed_dim), device=y.device)
                label_emb[masks] = self.label_emb(y[masks])
                # replace null labels with special embeddings
                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
                emb = emb + label_emb


            elif self.dataset=="clevr_norel":
                label_emb = th.empty((y.shape[0],  emb.shape[1]), device=y.device)
                # print("CHECK")
                try: 
                    label_emb[masks] = self.label_emb(y[masks])
                except:
                    label_emb[masks] = self.label_emb(y[masks].squeeze(1))
                # print("CHECK")
                # replace null labels with special embeddings
                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
       
                emb = emb + label_emb
                
            elif self.dataset == 'clevr_rel':
                assert masks is not None, "masks are not provided for training relational clevr data."
                label_emb = th.empty((y.shape[0], emb.shape[1]), device=y.device)
                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
                # embed objects and relations
                obj_emb_1 = self.fc(th.cat([self.obj_emb[i](y[masks][:, i]) for i in range(5)], dim=1))
                obj_emb_2 = self.fc(th.cat([self.obj_emb[i - 5](y[masks][:, i]) for i in range(5, 10)], dim=1))
                rel_emb = self.rel_emb(y[masks][:, -1])
                label_emb[masks] = th.cat((obj_emb_1, obj_emb_2, rel_emb), dim=1)
                emb = emb + label_emb
            elif self.dataset == 'clevr_count':
                assert masks is not None, "masks are not provided for training relational clevr data."
                label_emb = th.empty((y.shape[0], emb.shape[1]), device=y.device)
                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
                # embed objects and counts
                label_emb[masks]= self.fc(th.cat([self.obj_emb[i](y[masks][:, i]) for i in range(3)], dim=1))
            
                emb = emb + label_emb
            else:
                raise NotImplementedError

        h = x.to(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.to(x.dtype)
        return self.out(h)



class SuperResUNetModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear", align_corners=False)
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    
class InpaintUNetModel(UNetModel):
    """
    A UNetModel which can perform inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2 + 1
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, inpaint_image=None, inpaint_mask=None, **kwargs):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask], dim=1),
            timesteps,
            **kwargs,
        )


class SuperResInpaintUNetModel(UNetModel):
    """
    A UNetModel which can perform both upsampling and inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 3 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 3 + 1
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x,
        timesteps,
        inpaint_image=None,
        inpaint_mask=None,
        low_res=None,
        **kwargs,
    ):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask, upsampled], dim=1),
            timesteps,
            **kwargs,
        )





class Energy_UNetModel_full(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
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
    """

    def __init__(
            self,
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
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            encoder_channels=None,
            dataset=''
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dataset = dataset
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes:
            layer_classes = self.num_classes.split(',')
            if self.dataset == 'clevr_pos':
                self.label_emb = nn.Linear(int(layer_classes[0]), self.time_embed_dim)
                # unconditional label
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            elif self.dataset== "clevr_norel":
                self.label_emb = nn.Embedding(int(layer_classes[0]),self.time_embed_dim)
                self.null_emb = nn.Embedding(1, self.time_embed_dim)

            elif self.dataset == 'clevr_rel':
                attr_classes = [int(x) for x in layer_classes[:-1]]
                rel_class = int(layer_classes[-1])
                self.obj_emb = nn.ModuleList(
                    [nn.Embedding(num_classes, self.time_embed_dim // len(attr_classes))
                     for num_classes in attr_classes]
                )
                assert self.time_embed_dim // 3 == int(self.time_embed_dim / 3)
                self.fc = nn.Linear(self.time_embed_dim // len(attr_classes) * len(attr_classes),
                                    self.time_embed_dim // 3)
                self.rel_emb = nn.Embedding(rel_class, self.time_embed_dim // 3)
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            elif self.dataset == 'clevr_count':
                attr_classes = [int(x) for x in layer_classes]
                self.obj_emb = nn.ModuleList(
                    [nn.Embedding(num_classes, self.time_embed_dim // len(attr_classes))
                     for num_classes in attr_classes]
                )
                assert self.time_embed_dim // 3 == int(self.time_embed_dim / 3)
                self.fc = nn.Linear(self.time_embed_dim // len(attr_classes) * len(attr_classes),
                                    self.time_embed_dim)
                self.null_emb = nn.Embedding(1, self.time_embed_dim)
            else:
                raise NotImplementedError

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
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
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

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
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
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
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                if level and i == num_res_blocks:
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
            normalization(ch, swish=1.0),
            nn.Identity(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )
        self.use_fp16 = use_fp16

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

    def forward(self, x, timesteps, y=None, masks=None, layer_idx=None,eval=False,mala_sampler=False, energy_only=False):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param masks: an [N] Tensor of booleans
        :param layer_idx: integer indicates which attribute layer is used
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # if not eval:
        x.requires_grad_(True)
        
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes:
            # assert y.shape == (x.shape[0],)   # assert for discrete class labels
            if self.dataset == 'clevr_pos':
                label_emb = th.empty((y.shape[0], self.time_embed_dim), device=y.device)
                label_emb[masks] = self.label_emb(y[masks])
                # replace null labels with special embeddings
                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
                emb = emb + label_emb


            elif self.dataset=="clevr_norel":
                label_emb = th.empty((y.shape[0],  emb.shape[1]), device=y.device)
                # print("CHECK")
                try: 
                    label_emb[masks] = self.label_emb(y[masks])
                except:
                    label_emb[masks] = self.label_emb(y[masks].squeeze(1))

                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
       
                emb = emb + label_emb
                
            elif self.dataset == 'clevr_rel':
                assert masks is not None, "masks are not provided for training relational clevr data."
                label_emb = th.empty((y.shape[0], emb.shape[1]), device=y.device)
                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
                # embed objects and relations
                obj_emb_1 = self.fc(th.cat([self.obj_emb[i](y[masks][:, i]) for i in range(5)], dim=1))
                obj_emb_2 = self.fc(th.cat([self.obj_emb[i - 5](y[masks][:, i]) for i in range(5, 10)], dim=1))
                rel_emb = self.rel_emb(y[masks][:, -1])
                label_emb[masks] = th.cat((obj_emb_1, obj_emb_2, rel_emb), dim=1)
                emb = emb + label_emb
            elif self.dataset == 'clevr_count':
                assert masks is not None, "masks are not provided for training relational clevr data."
                label_emb = th.empty((y.shape[0], emb.shape[1]), device=y.device)
                label_emb[~masks] = self.null_emb.weight[0][None].repeat(label_emb[~masks].shape[0], 1)
                # embed objects and counts
                label_emb[masks]= self.fc(th.cat([self.obj_emb[i](y[masks][:, i]) for i in range(3)], dim=1))
            
                emb = emb + label_emb
            else:
                raise NotImplementedError

        h = x.to(self.dtype)

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.to(x.dtype)

        unet_out = self.out(h)

        # Energy Score = ||x -  f(x)||^2 AE based Energy
        energy_score = x-unet_out
        energy_norm_ = 0.5 * (energy_score ** 2)
        energy_norm= energy_norm_.sum()

        # Energy Score = ||f(x)||^2  l2 norm squared
        # energy_norm_ =  0.5 * (unet_out** 2)
        # energy_norm = energy_norm_.sum()
        
        if energy_only:
            return energy_norm

        if mala_sampler:
            
            eps = th.autograd.grad([energy_norm],[x],create_graph=False)[0]
            return energy_norm_,eps

        if not eval:
            eps = th.autograd.grad([energy_norm],[x],create_graph=True)[0]            
        else:
            # print("Eval Mode")
            eps = th.autograd.grad([energy_norm.sum()],[x],create_graph=False)[0]
            return eps
        return eps



# if __name__=="__main__":
#     channel_mult = ""
#     image_size = 64
#     attention_resolutions = "32,16,8"
#     if channel_mult == "":
#         if image_size == 256:
#             channel_mult = (1, 1, 2, 2, 4, 4)
#         elif image_size == 128:
#             channel_mult = (1, 1, 2, 3, 4)
#         elif image_size == 64:
#             channel_mult = (1, 2, 3, 4)
#         elif image_size == 32:
#             channel_mult = (1, 2, 2, 2)
#         else:
#             raise ValueError(f"unsupported image size: {image_size}")
#     else:
#         channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
#         assert 2 ** (len(channel_mult) + 2) == image_size

#     attention_ds = []
#     for res in attention_resolutions.split(","):
#         attention_ds.append(image_size // int(res))


#     model = Energy_UNetModel_full(in_channels=3,
#             model_channels=192,
#             out_channels=3,
#             num_res_blocks=2,
#             attention_resolutions=tuple(attention_ds),
#             dropout=0,
#             channel_mult=(1, 2, 4, 8),
#             conv_resample=True,
#             dims=2,
#             num_classes="3,",
#             use_checkpoint=False,
#             use_fp16=False,
#             num_heads=1,
#             num_head_channels=-1,
#             num_heads_upsample=-1,
#             use_scale_shift_norm=False,
#             resblock_updown=False,
#             encoder_channels=None,
#             dataset='clevr_norel',
#             )
    
#     x = th.randn(4,3,64,64).cuda()
#     ts = th.tensor( [0]*x.shape[0] ,device="cuda" )
#     y = th.tensor([[1],[0],[2],[1]],device="cuda")
#     masks = th.tensor([True]*x.shape[0],device="cuda")
#     model = model.cuda()    
#     x.requires_grad_(True)
#     # print(x.grad)
#     # print(ts.requires_grad)
#     energy, eps  = model(x,ts,y,masks,mala_sampler=True)
#     print(energy)
#     print(energy.shape,eps.shape)
    
#     # loss =th.nn.functional.mse_loss(out,x)
#     # print(loss)
#     # loss.backward()
