from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from numbers import Number
from typing import NamedTuple, Tuple, Union, List
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


device = th.device("cuda" if th.cuda.is_available() else "cpu")


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
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
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
    def forward(self, x, emb, cond = None, tab_cond = None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, cond = None, tab_cond = None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb = emb, cond = cond, tab_cond = tab_cond)
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
            # self.conv = nn.ConvTranspose2d(self.channels, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels, (x.shape[1], self.channels)   
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
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
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class fMRIEncoder(nn.Module):
    """
    Encoder for 4D fMRI data to feature embeddings
    """
    def __init__(self, input_shape=(91, 109, 91, 200), output_dim=512):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        # 3D CNN for spatial-temporal feature extraction
        self.conv3d_1 = nn.Conv3d(200, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        
        self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened size after pooling
        self.flatten_size = 256 * 4 * 4 * 4
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch, 91, 109, 91, 200)
        # Rearrange to (batch, 200, 91, 109, 91) for conv3d
        x = x.permute(0, 4, 1, 2, 3)
        
        x = F.relu(self.conv3d_1(x))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv3d_2(x))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv3d_3(x))
        
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class FCEncoder(nn.Module):
    """
    Encoder for 2D FC (Functional Connectivity) data to feature embeddings
    """
    def __init__(self, input_dim=300, output_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Since FC is symmetric, we can use upper triangular part
        self.upper_tri_size = input_dim * (input_dim - 1) // 2
        
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.upper_tri_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch, 300, 300)
        batch_size = x.shape[0]
        
        # Extract upper triangular part (excluding diagonal)
        upper_tri_indices = th.triu_indices(self.input_dim, self.input_dim, offset=1)
        x_upper_tri = x[:, upper_tri_indices[0], upper_tri_indices[1]]
        
        # x_upper_tri shape: (batch, upper_tri_size)
        x = self.fc_encoder(x_upper_tri)
        
        return x


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
        condition_dim = None, 
        cond_emb_channels = 512,
        tab_cond_dim = 9,
        use_conv=False,
        use_scale_shift_norm=True,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        use_time_condition = True,
        cond_apply_method = 'multi',
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.condition_dim = condition_dim
        self.cond_emb_channels = cond_emb_channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_time_condition = use_time_condition
        self.cond_apply_method = cond_apply_method

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, True, dims)
            self.x_upd = Upsample(channels, True, dims)
        elif down:
            self.h_upd = Downsample(channels, True, dims)
            self.x_upd = Downsample(channels, True, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        
        if use_time_condition:
            self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        if self.condition_dim is not None:
            if True:
                self.tab_embed = nn.Sequential(
                    linear(tab_cond_dim, self.condition_dim), 
                    nn.SiLU(),
                    linear(
                        self.condition_dim,
                        self.out_channels,
                    ),
                )
            else:
                self.tab_embed = nn.Sequential(
                    linear(tab_cond_dim, self.out_channels), ### AdaGN
                    nn.SiLU(),
                    linear(
                        self.out_channels, 
                        2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                    ),
                )
        
        self.input_out_channels = self.out_channels * 2 if (cond_apply_method == 'concat' and condition_dim is not None) else self.out_channels
        
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.input_out_channels, self.out_channels, 3, padding=1)
            ), # zero out the weights, it seems to help training
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        

    def forward(self, x, emb, cond=None, tab_cond=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, cond, tab_cond), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb=None, cond=None, tab_cond=None):
        
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.use_time_condition:
            if emb is not None:
                emb_out = self.emb_layers(emb).type(h.dtype)
                while len(emb_out.shape) < len(h.shape):
                    emb_out = emb_out[..., None]
            else:
                emb_out = None
        else:
            emb_out = None

        if self.condition_dim is not None:
            if cond is not None:
                cond_out = cond.type(h.dtype)
                while len(cond_out.shape) < len(h.shape):
                    cond_out = cond_out[..., None]
            else:
                cond_out = None
    
            
            if tab_cond is not None:
                tab_cond = self.tab_embed(tab_cond).type(h.dtype)
                while len(tab_cond.shape) < len(h.shape):            
                    tab_cond = tab_cond[..., None]
                tab_cond_out = tab_cond
            else:
                tab_cond_out = None
        
        else:
            cond_out = None    
            tab_cond_out = None

        if self.use_scale_shift_norm and (self.condition_dim is not None):
            h = self.apply_conditions(
                h=h,
                emb=emb_out,
                cond=cond_out,
                tab_cond=tab_cond_out,
                layers=self.out_layers,
                scale_bias=1,
                in_channels=self.out_channels,
                up_down_layer=None,
                cond_apply_method = self.cond_apply_method,
            )
        
        else:
            h = self.out_layers(h)
         
        return (self.skip_connection(x) + h) # * (1 / math.sqrt(2))
    
    def apply_conditions(
        self,
        h,
        emb=None,
        cond=None,
        tab_cond=None,
        layers: nn.Sequential = None,
        scale_bias: float = 1,
        in_channels: int = 512,
        up_down_layer: nn.Module = None,
        cond_apply_method = None,
    ):
        """
        apply conditions on the feature maps

        Args:
            emb: time conditional (ready to scale + shift)
            cond: encoder's conditional (ready to scale + shift)
            tab_cond: tabular conditional (ready to scale + shift)
        """
        two_cond = emb is not None and cond is not None
        three_cond = emb is not None and cond is not None and tab_cond is not None

        if emb is not None:
            # adjusting shapes
            while len(emb.shape) < len(h.shape):
                emb = emb[..., None]
        
        if two_cond or three_cond:
            scale_shifts = [emb]
            
            if cond is not None:
                # adjusting shapes
                while len(cond.shape) < len(h.shape):
                    cond = cond[..., None]

                if cond_apply_method == 'concat':
                    scale_shifts = [emb]
                else:
                    # time first
                    scale_shifts = [emb, cond]
            
            if tab_cond is not None:
                while len(tab_cond.shape) < len(h.shape):
                    tab_cond = tab_cond[..., None]
                scale_shifts.append(tab_cond)
        
        else:
            # "cond" is not used with single cond mode
            scale_shifts = [emb]

        # support scale, shift or shift only
        for i, each in enumerate(scale_shifts):
            if each is None:
                # special case: the condition is not provided
                a = None
                b = None
            else:
                if each.shape[1] == in_channels * 2:
                    a, b = th.chunk(each, 2, dim=1)
                else:
                    a = each
                    b = None
            scale_shifts[i] = (a, b)

        # condition scale bias could be a list
        if isinstance(scale_bias, Number):
            biases = [scale_bias] * len(scale_shifts)
        else:
            # a list
            biases = scale_bias

        # default, the scale & shift are applied after the group norm but BEFORE SiLU
        pre_layers, post_layers = layers[0], layers[1:] # GN as pre_layer

        # spilt the post layer to be able to scale up or down before conv
        # post layers will contain only the conv
        mid_layers, post_layers = post_layers[:-2], post_layers[-2:]


        h = pre_layers(h)

        # scale and shift for each condition
        for i, (scale, shift) in enumerate(scale_shifts):
            # if scale is None, it indicates that the condition is not provided
            if scale is not None:
                h = h * (biases[i] + scale)
                if shift is not None:
                    h = h + shift
        
        if self.input_out_channels == self.out_channels * 2 and self.cond_apply_method == 'concat':
            if cond is not None:
                h = th.cat([h, cond], dim=1)
            elif cond is None:
                _zeros = th.zeros_like(h)
                h = th.cat([h, _zeros], dim=1)
        
        h = mid_layers(h)

        # upscale or downscale if any just before the last conv
        if up_down_layer is not None:
            h = up_down_layer(h)
        h = post_layers(h)
        return h



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
        use_new_attention_order=True,
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
        return checkpoint(self._forward, (x,), self.parameters(), True)

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
    matmul_ops = 2 * b * (num_spatial ** 2) * c
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


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding, optional SE block.

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
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.2,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,  # True
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False, # False
        use_new_attention_order=False,
        use_condition = True,
        cond_emb_channels = 512,
        use_tabular_cond = 'bottleneck', # 'bottleneck', 'condition', 'None'
        tab_cond_dim = 9,
        use_time_condition = True,
        with_attention = True,
        cond_apply_method = 'adaN',
        # New parameters for fMRI and FC support
        use_fmri_cond = False,
        use_fc_cond = False,
        fmri_feature_dim = 512,
        fc_feature_dim = 512,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
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
        self.use_fp16 = use_fp16
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.use_condition = use_condition
        self.use_time_condition = use_time_condition
        self.use_tabular_cond = use_tabular_cond
        
        # New fMRI and FC conditioning parameters
        self.use_fmri_cond = use_fmri_cond
        self.use_fc_cond = use_fc_cond
        self.fmri_feature_dim = fmri_feature_dim
        self.fc_feature_dim = fc_feature_dim

        self.cond_apply_method = cond_apply_method

        self.condition_dim = None
        self.cond_emb_channels = None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Initialize fMRI encoder if enabled
        if self.use_fmri_cond:
            self.fmri_encoder = fMRIEncoder(output_dim=self.fmri_feature_dim)
            # Add fMRI embedding layer to project to time embedding dimension
            self.fmri_embed = nn.Sequential(
                linear(self.fmri_feature_dim, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        # Initialize FC encoder if enabled
        if self.use_fc_cond:
            self.fc_encoder = FCEncoder(output_dim=self.fc_feature_dim)
            # Add FC embedding layer to project to time embedding dimension
            self.fc_embed = nn.Sequential(
                linear(self.fc_feature_dim, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        if self.use_tabular_cond == 'bottleneck':
            self.tab_embed_dim = model_channels * channel_mult[-1]
            self.tab_embed = nn.Sequential(
                linear(tab_cond_dim, self.tab_embed_dim),
                nn.SiLU(),
                linear(self.tab_embed_dim, self.tab_embed_dim),
            )

        if self.use_condition:
            cond_embed_dim = model_channels * 4
            self.condition_dim = cond_embed_dim 
            self.cond_emb_channels = cond_emb_channels

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        
        input_in_channels = self.in_channels
        
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, input_in_channels, ch, 3, padding=1))]
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
                        condition_dim=self.condition_dim,
                        cond_emb_channels=self.cond_emb_channels,
                        use_time_condition=self.use_time_condition,
                        cond_apply_method = self.cond_apply_method,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    if dims == 2:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                            )
                        )

                    elif dims == 3:
                        # Add 3D attention or other 3D-specific layers here if needed
                        pass
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
                            condition_dim=None,
                            cond_emb_channels=self.cond_emb_channels,
                            cond_apply_method = self.cond_apply_method,
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

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                condition_dim=self.condition_dim,
                cond_emb_channels=self.cond_emb_channels,
                cond_apply_method = self.cond_apply_method,
            ),
            
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ) if with_attention else nn.Identity(),

            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                condition_dim=self.condition_dim,
                cond_emb_channels=self.cond_emb_channels,
                cond_apply_method = self.cond_apply_method,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        self.output_upsample_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                if i == 0 and level == (len(list(channel_mult)) - 1) and self.use_tabular_cond == 'bottleneck':
                    in_ch = ch + ich + self.tab_embed_dim
                else:
                    in_ch = ch + ich
                layers = [
                    ResBlock(
                        in_ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        condition_dim=self.condition_dim,
                        cond_emb_channels=self.cond_emb_channels,
                        cond_apply_method = self.cond_apply_method,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    if dims == 2:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                            )
                        )

                    elif dims == 3:
                        # Add 3D attention or other 3D-specific layers here if needed
                        pass
                if level and i == num_res_blocks:
                    out_ch = ch
                    upsample_layers = [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            condition_dim=None,
                            cond_emb_channels=self.cond_emb_channels,
                            cond_apply_method = self.cond_apply_method,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    ]
                    self.output_upsample_blocks.append(TimestepEmbedSequential(*upsample_layers))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, self.out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.output_upsample_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.output_upsample_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, cond=None, tab_cond=None, y=None, return_features=False, fmri=None, fc=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param cond: conditioning input (e.g., MRI data).
        :param tab_cond: tabular conditioning data.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param return_features: whether to return intermediate features.
        :param fmri: an [N x 91 x 109 x 91 x 200] Tensor of fMRI data.
        :param fc: an [N x 300 x 300] Tensor of FC data.
        :return: an [N x C x ...] Tensor of outputs.
        """

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if isinstance(cond, List):
            len_cond = len(cond) - 1
            original_input_cond = cond[0]
            encoder_cond = cond[1 : len_cond//2 + 1]
            mid_cond = cond[len_cond//2 + 1]
            decoder_cond = cond[len_cond//2 + 2:]

            assert len(encoder_cond) == len(decoder_cond), (len(encoder_cond), len(decoder_cond))
        else:
            original_input_cond = cond
        
        if return_features:
            features = []
            features.append(x)

        hs = []
        if timesteps is not None:
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        else:
            emb = None

        # Process fMRI conditioning if enabled and provided
        if self.use_fmri_cond and fmri is not None:
            fmri_features = self.fmri_encoder(fmri)
            fmri_emb = self.fmri_embed(fmri_features)
            if emb is not None:
                emb = emb + fmri_emb
            else:
                emb = fmri_emb

        # Process FC conditioning if enabled and provided
        if self.use_fc_cond and fc is not None:
            fc_features = self.fc_encoder(fc)
            fc_emb = self.fc_embed(fc_features)
            if emb is not None:
                emb = emb + fc_emb
            else:
                emb = fc_emb

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        if self.use_tabular_cond == 'bottleneck' and tab_cond is not None:
            assert tab_cond.shape[1] == self.tab_embed[0].in_features
            self.tab_cond_embedding = self.tab_embed(tab_cond)
            tab_condition = None 
        elif self.use_tabular_cond == 'condition' and tab_cond is not None:
            tab_condition = tab_cond
        else:
            self.tab_cond_embedding = None
            tab_condition = None        

        h = x.type(self.dtype)

        # Process input blocks
        for num, module in enumerate(self.input_blocks):
            if isinstance(cond, List):
                h = module(h, emb, encoder_cond[num], tab_condition)
            else:
                h = module(h, emb, cond, tab_condition)
            
            hs.append(h)

            if return_features:
                feature = h
                features.append(feature)
        
        # Process middle block
        if isinstance(cond, List):
            h = self.middle_block(h, emb, mid_cond, tab_condition)
        else:
            h = self.middle_block(h, emb, cond, tab_condition)

        if return_features:
            feature = h
            features.append(feature)
        
        # Add tabular conditioning at bottleneck if enabled
        if self.use_tabular_cond == 'bottleneck' and tab_cond is not None:
            self.tab_cond_embedding = self.tab_cond_embedding.reshape(h.shape[0], -1, 1, 1)
            # expand tabular condition to match the feature map size
            self.tab_cond_embedding = self.tab_cond_embedding.repeat(1, 1, h.shape[2], h.shape[3])
            h = th.cat([h, self.tab_cond_embedding], dim=1)

        # Process output blocks
        for num, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)

            if isinstance(cond, List):
                h = module(h, emb, decoder_cond[num], tab_condition)
            else:
                h = module(h, emb, cond, tab_condition)
            
            if return_features:
                feature = h
                features.append(feature)
            
            if num != 0 and (num + 1) % (self.num_res_blocks + 1) == 0 and ((num + 1) // (self.num_res_blocks + 1) < (len(self.output_upsample_blocks) + 1)):
                if isinstance(cond, List):
                    h = self.output_upsample_blocks[(num + 1) // (self.num_res_blocks + 1) - 1](h, emb, decoder_cond[num], tab_condition)
                else:
                    h = self.output_upsample_blocks[(num + 1) // (self.num_res_blocks + 1) - 1](h, emb, cond, tab_condition)
                
              
        h = h.type(x.dtype)

        pred = self.out(h)
        if return_features:
            return pred, features

        return pred

class Return(NamedTuple):
    pred: th.Tensor