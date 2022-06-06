from .transformer_utils import BertAttention, trans_nd, layer_norm
from transformers import AutoConfig
# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder
import torch
from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    timestep_embedding,
    checkpoint,
)


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

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x



class TransSimpleBlock(TimestepBlock):
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
        config=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        attention_head_size = 64
        assert self.out_channels % attention_head_size == 0
        self.in_layers = nn.Sequential(
            layer_norm(channels),
            SiLU(),
            trans_nd(config, channels, self.out_channels // attention_head_size, attention_head_size),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            layer_norm(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                trans_nd(config, self.out_channels, self.out_channels // attention_head_size, attention_head_size),
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:

            self.skip_connection = trans_nd(config, channels, self.out_channels // attention_head_size,
                                            attention_head_size)
        else:
            self.skip_connection = nn.Sequential(nn.Linear(self.channels, self.out_channels),
                                                 nn.LayerNorm(self.out_channels, eps=config.layer_norm_eps),
                                                 )

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
        # print('-'*30)
        # print(self.in_layers)
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        # print(self.in_layers, h.shape, x.shape, )
        # print(emb.shape, self.emb_layers, emb_out.shape)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out.unsqueeze(1)
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=-1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h





class TransModel(nn.Module):
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
        dims=1,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        config=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if config is None:
            config = AutoConfig.from_pretrained('bert-base-uncased')
            config.position_embedding_type = 'relative_key'
            config.max_position_embeddings = 256

            # print(self.position_embedding_type, config.max_position_embeddings)


        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        attention_head_size = 64
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    trans_nd(config, in_channels, model_channels // attention_head_size, attention_head_size)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    TransformerBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        config=config,
                    )
                ]
                ch = mult * model_channels
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                #         )
                #     )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            TransformerBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                config=config,
            ),
            # AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            TransformerBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                config=config,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    TransformerBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        config=config,
                    )
                ]
                ch = model_channels * mult
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             use_checkpoint=use_checkpoint,
                #             num_heads=num_heads_upsample,
                #         )
                #     )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        attention_head_size_final = 8
        self.out = nn.Sequential(
            layer_norm(ch),
            SiLU(),
            trans_nd(config, model_channels, out_channels // attention_head_size_final, attention_head_size_final),
        # zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        print(self.out, out_channels)

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

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class TransformerNetModel(nn.Module):
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
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        config=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if config is None:
            config = AutoConfig.from_pretrained('bert-base-uncased')

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        attention_head_size = 64
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    trans_nd(config, in_channels, model_channels // attention_head_size, attention_head_size)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    TransSimpleBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        config=config,
                    )
                ]
                ch = mult * model_channels

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            # if level != len(channel_mult) - 1:
            #     self.input_blocks.append(
            #         TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
            #     )
            #     input_block_chans.append(ch)
            #     ds *= 2

        self.middle_block = TimestepEmbedSequential(
            TransSimpleBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                config=config,
            ),
            TransSimpleBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                config=config,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        print(input_block_chans)
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks ):
                layers = [
                    TransSimpleBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        config=config,
                    )
                ]
                ch = model_channels * mult

                # if level and i == num_res_blocks:
                #     layers.append(Upsample(ch, conv_resample, dims=dims))
                #     ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        attention_head_size_final = 8
        self.out = nn.Sequential(
            layer_norm(ch),
            SiLU(),
            zero_module(trans_nd(config, model_channels, out_channels // attention_head_size_final,
                                            attention_head_size_final)),
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

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            # print(hs[-1].shape)
            cat_in = th.cat([h, hs.pop()], dim=-1)
            # print(cat_in.shape, h.shape, )
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class TransformerNetModel2(nn.Module):
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
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        config=None,
        config_name='bert-base-uncased',
        training_mode='emb',
        vocab_size=None,
        experiment_mode='lm',
        init_pretrained=False,
        logits_mode=1,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout
            # config.hidden_size = 512

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.logits_mode = logits_mode

        if training_mode == 'e2e':
            self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
            if self.logits_mode == 2:
                # self.lm_head = nn.Linear(self.in_channels, vocab_size, bias=False)
                self.lm_head = nn.Linear(self.in_channels, vocab_size, bias=True)

            else:
                self.lm_head = nn.Linear(self.in_channels, vocab_size)
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
        elif training_mode == 'e2e-simple':
            self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
            self.lm_head = nn.Linear(self.in_channels, vocab_size)
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight

        if experiment_mode == 'conditional_gen':
            self.conditional_gen = True
            self.encoder_emb = nn.Embedding(vocab_size, config.hidden_size)
            self.encoder = BertEncoder(config)
            print(config, 'conditional_gen')
            config.is_decoder = True
            config.add_cross_attention = True
        elif experiment_mode == 'lm':
            self.conditional_gen = False


        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)


        # self.input_up_proj = trans_nd(config, in_channels, model_channels // attention_head_size, attention_head_size)
        self.input_up_proj = nn.Sequential(nn.Linear(in_channels, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        if init_pretrained:
            from transformers.models.bert.modeling_bert import BertModel
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            del temp_bert.embeddings
            del temp_bert.pooler
            self.input_transformers = temp_bert.encoder
            print('initializing from pretrained bert.')
        else:
            print(config)
            self.input_transformers = BertEncoder(config)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # config2 = config
        # config2.hidden_size = 2 * config.hidden_size
        # self.output_transformers = BertEncoder(config)



        self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, out_channels))


    # def convert_to_fp16(self):
    #     """
    #     Convert the torso of the model to float16.
    #     """
    #     self.input_blocks.apply(convert_module_to_f16)
    #     self.middle_block.apply(convert_module_to_f16)
    #     self.output_blocks.apply(convert_module_to_f16)
    #
    # def convert_to_fp32(self):
    #     """
    #     Convert the torso of the model to float32.
    #     """
    #     self.input_blocks.apply(convert_module_to_f32)
    #     self.middle_block.apply(convert_module_to_f32)
    #     self.output_blocks.apply(convert_module_to_f32)
    #
    # @property
    # def inner_dtype(self):
    #     """
    #     Get the dtype used by the torso of the model.
    #     """
    #     return next(self.input_blocks.parameters()).dtype

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()

            #
            # scores1 = th.cdist(self.lm_head.weight.unsqueeze(0), hidden_repr, p=2)
            # scores1 = -scores1.permute(0, 2, 1).contiguous()
            #
            # print(scores1.shape, scores.shape)
            # print(scores1[0,0], scores[0,0])
            # print(torch.isclose(scores1, scores))

            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps, y=None, src_ids=None, src_mask=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # print(f'real model inputs: {timesteps}')
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.conditional_gen:
            assert src_ids is not None
            # print(src_ids.shape, 'source_ids shape')
            src_emb = self.encoder_emb(src_ids)
            # print(src_ids.shape, src_emb.shape)
            encoder_hidden_states = self.encoder(src_emb).last_hidden_state
            encoder_attention_mask = src_mask.unsqueeze(1).unsqueeze(1)



        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)


        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        # print(emb_x.shape, emb.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        if self.conditional_gen:
            # print(emb_inputs.shape, encoder_hidden_states.shape, encoder_attention_mask.shape)
            input_trans_hidden_states = self.input_transformers(emb_inputs,
                                                                encoder_hidden_states=encoder_hidden_states,
                                                                encoder_attention_mask=encoder_attention_mask,
                                                                ).last_hidden_state
        else:
            input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)
        return h

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result