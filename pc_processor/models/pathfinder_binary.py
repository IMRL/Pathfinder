# !/usr/bin/env python3
import logging
import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from .weight_init import named_apply, lecun_normal_
from .utils_quant import *

_logger = logging.getLogger(__name__)


# Binary

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class LearnableBiasnn(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBiasnn, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, out_chn, 1, 1]), requires_grad=True)  # BCHW

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class LearnableBiastt(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBiastt, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, 1, out_chn]), requires_grad=True)  # BNC，N=H*W

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class RPReLU(nn.Module):
    def __init__(self, out_chn, toc=1):
        super(RPReLU, self).__init__()
        self.toc = toc
        if self.toc == 1:
            self.move1 = LearnableBiasnn(out_chn)
            self.move2 = LearnableBiasnn(out_chn)
        else:
            self.move1 = LearnableBiastt(out_chn)
            self.move2 = LearnableBiastt(out_chn)
        self.act = nn.PReLU(out_chn)

    def forward(self, x):
        x = self.move1(x)
        if self.toc == 1:
            x = self.act(x)
        else:
            x = self.act(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.move2(x)
        return x


def binary_weight(x, linear):
    if linear:
        scaling_factor = torch.mean(abs(x), dim=1, keepdim=True)
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = x / (torch.sqrt(x.var(dim=-1, keepdim=True) + 1e-5) / 2 / np.sqrt(2))
    else:
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(x), dim=3, keepdim=True), dim=2, keepdim=True), dim=1,
                                    keepdim=True)
        x = x - x.mean([1, 2, 3], keepdim=True)
        x = x / (torch.sqrt(x.var([1, 2, 3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))

        EW = torch.mean(torch.abs(x))
        Q_tau = (- EW * np.log(2 - 2 * 0.92)).detach().cpu().item()
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(x)
        cliped_weights = torch.clamp(x, -Q_tau, Q_tau)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
    return binary_weights


class LayerScale(nn.Module):
    def __init__(self, hidden_size, init_ones=True):
        super().__init__()
        if init_ones:
            self.alpha = nn.Parameter(torch.ones(hidden_size) * 0.1)
        else:
            self.alpha = nn.Parameter(torch.zeros(hidden_size))
        self.move = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        out = x * self.alpha + self.move
        return out


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.dense = QuantizeLinear(in_features, hidden_features)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.move1 = nn.Parameter(torch.zeros(in_features))
        self.rprelu1 = RPReLU(hidden_features, toc=2)
        self.dense2 = QuantizeLinear(hidden_features, out_features)
        self.move2 = nn.Parameter(torch.zeros(hidden_features))
        self.norm2 = nn.LayerNorm(out_features)
        self.rprelu2 = RPReLU(out_features, toc=2)
        self.pooling = nn.AvgPool1d(4)
        self.layerscale = LayerScale(out_features)

    def forward(self, x):
        if self.in_features == self.hidden_features:
            hidden = self.norm1(self.dense(x + self.move1)) + x
            hidden = self.rprelu1(hidden)
            out = self.norm2(self.dense2(hidden + self.move2)) + hidden
        else:
            hidden = self.norm1(self.dense(x + self.move1)) + torch.concat([x for _ in range(4)], dim=-1)
            hidden = self.rprelu1(hidden)
            out = self.norm2(self.dense2(hidden + self.move2)) + self.pooling(hidden)
        out = self.rprelu2(out)
        out = self.layerscale(out)
        return out


class token_mixer(nn.Module):
    def __init__(self, in_chn, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1, padding='same'):
        super(token_mixer, self).__init__()
        self.scalef = nn.Parameter(torch.ones([1, in_chn, 1, 1]), requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.dilation1 = dilation1
        self.dilation2 = dilation2
        self.dilation3 = dilation3
        self.number_of_weights = in_chn * kernel_size * kernel_size
        self.shape = (in_chn, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)
        self.weights2 = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)
        self.weights3 = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)
        self.norm = nn.BatchNorm2d(in_chn)
        self.act1 = RPReLU(in_chn, toc=1)
        self.act2 = RPReLU(in_chn, toc=1)
        self.act3 = RPReLU(in_chn, toc=1)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        binary_weight1 = binary_weight(real_weights, False)
        real_weights2 = self.weights2.view(self.shape)
        binary_weight2 = binary_weight(real_weights2, False)
        real_weights3 = self.weights3.view(self.shape)
        binary_weight3 = binary_weight(real_weights3, False)
        x = act_quant_fn(x) * self.scalef
        x1 = F.conv2d(x, binary_weight1, stride=self.stride, padding=self.padding, dilation=self.dilation1,
                      groups=self.shape[0])
        x1 = self.act1(x1)
        x2 = F.conv2d(x, binary_weight2, stride=self.stride, padding=self.padding, dilation=self.dilation2,
                      groups=self.shape[0])
        x2 = self.act2(x2)
        x3 = F.conv2d(x, binary_weight3, stride=self.stride, padding=self.padding, dilation=self.dilation3,
                      groups=self.shape[0])
        x3 = self.act3(x3)
        x = self.norm(x1 + x2 + x3)
        return x


class Token_for_Attention(nn.Module):
    def __init__(self, dim, window_size=8):
        super(Token_for_Attention, self).__init__()
        self.window_size = window_size
        self.merge_avg = nn.AvgPool2d(kernel_size=window_size, stride=window_size, padding=0)
        self.merge_max = nn.MaxPool2d(kernel_size=window_size, stride=window_size, padding=0)
        self.mlp = Mlp(dim, hidden_features=dim)
        self.a1 = nn.Parameter(0.25 * torch.ones([1, 1, dim]), requires_grad=True)
        self.a2 = nn.Parameter(0.25 * torch.ones([1, 1, dim]), requires_grad=True)
        self.norm = nn.LayerNorm(dim)

    def Merge_token(self, x):
        merge_token_avage = self.merge_avg(x).permute(0, 2, 3, 1).flatten(1, 2)
        merge_token_max = self.merge_max(x).permute(0, 2, 3, 1).flatten(1, 2)
        merge_token_mlp = self.mlp(merge_token_avage)
        merge_token = (1.0 - self.a1 - self.a2).expand_as(merge_token_mlp) * merge_token_mlp + self.a1.expand_as(
            merge_token_max) * merge_token_max + self.a2.expand_as(merge_token_avage) * merge_token_avage
        return merge_token

    def forward(self, x):
        windows = windows_split(x, self.window_size)
        rep_ratio = x.shape[2] // self.window_size
        merge_token = self.Merge_token(x)
        merge_token_new = torch.repeat_interleave(merge_token, rep_ratio ** 2, dim=0)
        token_all = torch.cat((windows, merge_token_new), dim=1)
        token_all = self.norm(token_all)
        return token_all, windows.shape[1]


def windows_split(x, window_size):
    B, C, H, W, = x.shape
    x = x.permute(0, 2, 3, 1).view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows
#########
class BHViTSelfAttention(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.token_FA = Token_for_Attention(dim=dim, window_size=8)
        self.windows_size = 8
        self.num_attention_heads = 8
        self.attention_head_size = int(dim / 8.0)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.moveq = nn.Parameter(torch.zeros(dim))
        self.movek = nn.Parameter(torch.zeros(dim))
        self.movev = nn.Parameter(torch.zeros(dim))

        self.query = QuantizeLinear(dim, dim)
        self.key = QuantizeLinear(dim, dim)
        self.value = QuantizeLinear(dim, dim)

        self.normq = nn.LayerNorm(dim)
        self.normk = nn.LayerNorm(dim)
        self.normv = nn.LayerNorm(dim)

        self.rpreluq = RPReLU(dim, toc=2)
        self.rpreluk = RPReLU(dim, toc=2)
        self.rpreluv = RPReLU(dim, toc=2)

        self.moveq2 = nn.Parameter(torch.zeros(dim))
        self.movek2 = nn.Parameter(torch.zeros(dim))
        self.movev2 = nn.Parameter(torch.zeros(dim))

        self.act_quantizer = None
        self.att_prob_quantizer = None
        self.att_prob_clip = None

        self.att_prob_quantizer = BinaryActivation_Attention(self.num_attention_heads, 3)
        self.att_prob_clip2 = None
        self.norm_context = nn.LayerNorm(dim)

        self.rprelu_context = RPReLU(dim, toc=2)

        self.dropout = nn.Dropout()

        self.parm = nn.Parameter(0.5 * torch.ones([1, dim, 1, 1]), requires_grad=True)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def window_reverse(self, windows, window_size, H, W, B):
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
        return x

    def window_reverse_high(self, windows, window_size, H, W, B):
        x = windows.view(B, H // window_size * W // window_size, H // window_size, W // window_size, -1)
        x = torch.mean(x, dim=1).permute(0, 3, 1, 2)
        x = torch.nn.functional.interpolate(x, size=H, mode='nearest')
        return x

    def token_split(self, x, split_dim, H, W, B):
        x1, x2 = torch.split(x, split_dim, dim=1)
        x1 = self.window_reverse(x1, self.windows_size, H, W, B)
        x2 = self.window_reverse_high(x2, self.windows_size, H, W, B)
        return x1 * self.parm.expand_as(x1) + x2 * (1.0 - self.parm).expand_as(x2)

    def forward(self, hidden_states):
        B, C, H, W = hidden_states.shape
        hidden_states, split_dim = self.token_FA(hidden_states)
        mixed_query_layer = self.normq(self.query(hidden_states + self.moveq)) + hidden_states
        mixed_key_layer = self.normk(self.key(hidden_states + self.movek)) + hidden_states
        mixed_value_layer = self.normv(self.value(hidden_states + self.movev)) + hidden_states
        mixed_query_layer = self.rpreluq(mixed_query_layer)
        mixed_key_layer = self.rpreluk(mixed_key_layer)
        mixed_value_layer = self.rpreluv(mixed_value_layer)
        query_layer = mixed_query_layer + self.moveq2
        key_layer = mixed_key_layer + self.movek2
        value_layer = mixed_value_layer + self.movev2

        if self.act_quantizer is not None:
            query_layer = self.act_quantizer.apply(query_layer)
            key_layer = self.act_quantizer.apply(key_layer)
            value_layer = self.act_quantizer.apply(value_layer)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.att_prob_quantizer(attention_probs * 3)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = self.norm_context(context_layer) + mixed_query_layer + mixed_key_layer + mixed_value_layer
        context_layer = self.rprelu_context(context_layer)
        context_layer = self.token_split(context_layer, split_dim, H, W, B)
        outputs = (context_layer,)

        return outputs


class BHViTSelfOutput(nn.Module):
    def __init__(self, embed_dims) -> None:
        super().__init__()
        self.dense = QuantizeLinear(embed_dims, embed_dims)
        self.dropout = nn.Dropout()
        self.move = nn.Parameter(torch.zeros(embed_dims))
        self.norm = nn.LayerNorm(embed_dims)
        self.rprelu = RPReLU(embed_dims, toc=2)
        self.layerscale = LayerScale(embed_dims)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.norm(self.dense(hidden_states + self.move)) + hidden_states
        out = self.rprelu(out)
        out = self.dropout(out)
        out = self.layerscale(out)
        return out


class BHViTAttention(nn.Module):
    def __init__(self, embed_dims) -> None:
        super().__init__()
        self.attention = BHViTSelfAttention(embed_dims)
        self.output = BHViTSelfOutput(embed_dims)

    def forward(self, hidden_states):
        self_outputs = self.attention(hidden_states)
        outputs = self.output(self_outputs[0].permute(0, 2, 3, 1).flatten(1, 2))
        return outputs


class Block(nn.Module):
    def __init__(self, embed_dims, mlp_ratio, toa, drop_path=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(embed_dims)
        self.toa = toa
        if self.toa == 0:
            self.token_mix = token_mixer(in_chn=embed_dims)
        else:
            self.token_mix = BHViTAttention(embed_dims)
        self.norm2 = nn.BatchNorm2d(embed_dims)
        self.mlp = Mlp(in_features=embed_dims, hidden_features=mlp_ratio * embed_dims)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.toa == 0:
            x = x + self.drop_path(self.token_mix(self.norm1(x)))
        else:
            x = x + self.drop_path(self.token_mix(self.norm1(x))).permute(0, 2, 1).view(-1, C, H, W).contiguous()
        x = x + self.drop_path(self.mlp(self.norm2(x).permute(0, 2, 3, 1).flatten(1, 2))).permute(0, 2, 1).view(-1, C,
                                                                                                                H,
                                                                                                                W).contiguous()
        return x


class Embedding(nn.Module):
    def __init__(self, in_chans=8, out_chans=32, norm_layer=nn.BatchNorm2d, use_norm=True):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, stride=2, bias=True)
        self.norm = norm_layer(out_chans) if use_norm else nn.Identity()
        self.act = RPReLU(out_chans, toc=1)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Downsample_CNN(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, norm_layer=nn.BatchNorm2d, use_norm=True):
        super().__init__()
        self.proj = nn.Conv2d(
            in_embed_dim,
            out_embed_dim,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        self.norm = norm_layer(
            out_embed_dim) if use_norm else nn.Identity()
        self.act = RPReLU(out_embed_dim, toc=1)

    def forward(self, x):
        x = self.proj(x)
        # B,C,H,W = x.shape
        x = self.norm(x)
        x = self.act(x)
        return x


############################################


############################################
class CFI_ViT_w1a1(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, layers, mlp_ratios, embed_dims, toa, in_chans=3, num_classes=1000, depth=12,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, embed_layer=Embedding,
                 norm_layer=nn.BatchNorm2d, act_layer=RPReLU, weight_init=''):
        """
        Args:

            patch_size (int, tuple): patch size

            num_classes (int): number of classes for classification head
            config.hidden_size (int): embedding dimension
            depth (int): depth of transformer

            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        norm_layer = norm_layer
        act_layer = act_layer
        self.embed = embed_layer(in_chans=in_chans, out_chans=embed_dims[0])
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, (
                depth[0] + depth[1] + depth[2] + depth[3]))]  # stochastic depth decay rule
        self.blocks1 = nn.Sequential(*[
            Block(embed_dims[0], mlp_ratio=mlp_ratios[0], toa=toa[0], drop_path=dpr[i])
            for i in range(layers[0])])
        self.down1 = Downsample_CNN(embed_dims[0], embed_dims[1])
        self.blocks2 = nn.Sequential(*[
            Block(embed_dims[1], mlp_ratio=mlp_ratios[1], toa=toa[1], drop_path=dpr[i + 2])
            for i in range(layers[1])])
        self.down2 = Downsample_CNN(embed_dims[1], embed_dims[2])
        self.blocks3 = nn.Sequential(*[
            Block(embed_dims[2], mlp_ratio=mlp_ratios[2], toa=toa[2], drop_path=dpr[i + 4])
            for i in range(layers[2])])
        self.down3 = Downsample_CNN(embed_dims[2], embed_dims[3])
        self.blocks4 = nn.Sequential(*[
            Block(embed_dims[3], mlp_ratio=mlp_ratios[3], toa=toa[3], drop_path=dpr[i + 8])
            for i in range(layers[3])])
        self.norm = norm_layer(embed_dims[3])
        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        # trunc_normal_(self.embed, std=.02)
        # if self.dist_token is not None:
        #     trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            self.apply(_init_vit_weights)
        if isinstance(mode, nn.Linear):
            trunc_normal_(mode.weight, std=.02)
            if isinstance(mode, nn.Linear) and mode.bias is not None:
                nn.init.constant_(mode.bias, 0)
        elif isinstance(mode, nn.LayerNorm):
            nn.init.constant_(mode.bias, 0)
            nn.init.constant_(mode.weight, 1.0)
        elif isinstance(mode, nn.BatchNorm2d):
            nn.init.ones_(mode.weight)
            nn.init.zeros_(mode.bias)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    def forward(self, x):  # BCHW
        x = self.embed(x)
        for i, layer in enumerate(list(self.blocks1)):
            x = layer(x)
        x1 = x
        x = self.down1(x)
        for i, layer in enumerate(list(self.blocks2)):
            x = layer(x)
        x2 = x
        x = self.down2(x)  # BCHW
        for i, layer in enumerate(list(self.blocks3)):
            x = layer(x)
        x3 = x
        x = self.down3(x)

        for i, layer in enumerate(list(self.blocks4)):
            x = layer(x)
        x4 = x
        # 通道数：64，128，256，512
        # 尺寸：256，128，64，32
        return [x1, x2, x3, x4]


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def CFI_ViT_224_w1a1(num_classes, **kwargs):
    layers = [2, 2, 6, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 256, 512]
    toa = [0, 0, 1, 1]
    depth = [2, 2, 6, 2]
    model = CFI_ViT_w1a1(layers, mlp_ratios, embed_dims, toa=toa, num_classes=num_classes, depth=depth, qkv_bias=True,
                         norm_layer=nn.BatchNorm2d, **kwargs)
    model.default_cfg = _cfg()
    return model


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class ResidualBasedFusionBlock(nn.Module):
    def __init__(self, pcd_channels, img_channels):
        super(ResidualBasedFusionBlock, self).__init__()
        self.move1 = LearnableBias(pcd_channels + img_channels)
        self.fuse_conv = QuantizeConv2d(pcd_channels + img_channels, pcd_channels, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(pcd_channels)
        self.rprelu1 = RPReLU(pcd_channels)
        self.move3 = LearnableBias(pcd_channels)
        self.attention_part2 = QuantizeConv2d(pcd_channels, pcd_channels, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(pcd_channels)
        self.attention_score = nn.Sigmoid()

    def forward(self, pcd_feature, img_feature):
        cat_feature = torch.cat((pcd_feature, img_feature), dim=1)
        fuse_out = self.move1(cat_feature)
        fuse_out = self.fuse_conv(fuse_out)
        fuse_out = self.norm1(fuse_out)
        fuse_out = fuse_out + (pcd_feature + img_feature)
        fuse_out = self.rprelu1(fuse_out)
        fuse_out2 = self.move3(fuse_out)
        attention_map = self.attention_part2(fuse_out2)
        attention_map = self.norm3(attention_map)
        attention_map = self.attention_score(attention_map + fuse_out)
        out = fuse_out * attention_map + pcd_feature
        return out


class ASPP_binary(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP_binary, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.move1 = LearnableBias(in_channel)
        self.atrous_block1 = QuantizeConv2d(in_channel, depth, 1, 1)
        self.norm1 = nn.BatchNorm2d(depth)
        self.rprelu1 = RPReLU(depth)
        self.move6 = LearnableBias(in_channel)
        self.atrous_block6 = QuantizeConv2d(
            in_channel, depth, 3, 1, 6, 6)
        self.norm6 = nn.BatchNorm2d(depth)
        self.rprelu6 = RPReLU(depth)
        self.move12 = LearnableBias(in_channel)
        self.atrous_block12 = QuantizeConv2d(
            in_channel, depth, 3, 1, 12, 12)
        self.norm12 = nn.BatchNorm2d(depth)
        self.rprelu12 = RPReLU(depth)
        self.move18 = LearnableBias(in_channel)
        self.atrous_block18 = QuantizeConv2d(
            in_channel, depth, 3, 1, 18, 18)
        self.norm18 = nn.BatchNorm2d(depth)
        self.rprelu18 = RPReLU(depth)
        self.conv_1x1_output = QuantizeConv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        image_features = F.interpolate(self.mean(x), size=x.shape[2:], mode='nearest')
        atrous_block1 = self.move1(x)
        atrous_block1 = self.atrous_block1(atrous_block1)
        atrous_block1 = self.norm1(atrous_block1) + x
        atrous_block1 = self.rprelu1(atrous_block1)
        atrous_block6 = self.move6(x)
        atrous_block6 = self.atrous_block6(atrous_block6)
        atrous_block6 = self.norm6(atrous_block6) + x
        atrous_block6 = self.rprelu6(atrous_block6)
        atrous_block12 = self.move12(x)
        atrous_block12 = self.atrous_block12(atrous_block12)
        atrous_block12 = self.norm12(atrous_block12) + x
        atrous_block12 = self.rprelu12(atrous_block12)
        atrous_block18 = self.move18(x)
        atrous_block18 = self.atrous_block18(atrous_block18)
        atrous_block18 = self.norm18(atrous_block18) + x
        atrous_block18 = self.rprelu18(atrous_block18)
        net = (image_features + x + atrous_block1 + atrous_block6 + atrous_block12 + atrous_block18) / 6
        return net


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)
        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)
        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)
        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        # self.move1 = LearnableBias(in_filters)
        # self.conv1 = QuantizeConv2d(in_filters, out_filters, (1, 1), 1)
        # self.norm1 = nn.BatchNorm2d(out_filters)
        # self.act1 = RPReLU(out_filters)

        self.move2 = LearnableBias(in_filters)
        self.conv2 = QuantizeConv2d(in_filters, out_filters, (3, 3), 1, 1)
        self.act2 = RPReLU(out_filters)
        self.norm2 = nn.BatchNorm2d(out_filters)

        self.move3 = LearnableBias(out_filters)
        self.conv3 = QuantizeConv2d(out_filters, out_filters, (3, 3), 1, 2, 2)
        self.act3 = RPReLU(out_filters)
        self.norm3 = nn.BatchNorm2d(out_filters)

        self.move4 = LearnableBias(out_filters)
        self.conv4 = QuantizeConv2d(out_filters, out_filters, (2, 2), 1, 1, 2)
        self.act4 = RPReLU(out_filters)
        self.norm4 = nn.BatchNorm2d(out_filters)

        # self.move5 = LearnableBias(out_filters * 3)
        # self.conv5 = QuantizeConv2d(out_filters * 3, out_filters, (1, 1))
        # self.act5 = RPReLU(out_filters)
        # self.norm5 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):

        resA = self.move2(x)
        resA = self.conv2(resA)
        resA = self.norm2(resA) + torch.concat([x for _ in range(2)], dim=1)
        resA1 = self.act2(resA)

        resA = self.move3(resA1)
        resA = self.conv3(resA)
        resA = self.norm3(resA) + resA1
        resA2 = self.act3(resA)

        resA = self.move4(resA2)
        resA = self.conv4(resA)
        resA = self.norm4(resA) + resA2
        resA3 = self.act4(resA)

        resA = (resA1 + resA2 + resA3) / 3

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.pool = nn.AvgPool1d(2, 2)
        self.conv1 = nn.Conv2d(in_filters // 4 + in_filters, out_filters, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.act1 = nn.LeakyReLU()
        self.move2 = LearnableBias(out_filters)
        self.conv2 = QuantizeConv2d(out_filters, out_filters, (3, 3), 1, 2, 2)
        self.act2 = RPReLU(out_filters)
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.move3 = LearnableBias(out_filters)
        self.conv3 = QuantizeConv2d(out_filters, out_filters, (2, 2), 1, 1, 2)
        self.act3 = RPReLU(out_filters)
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        B, C, H, W = skip.shape
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)
        upE = self.conv1(upB)
        upE = self.bn1(upE) + (torch.concat([upA for _ in range(2)], dim=1) + self.pool(
            skip.permute(0, 2, 3, 1).flatten(1, 2)).permute(0, 2, 1).reshape(B, C // 2, H, W))
        upE1 = self.act1(upE)

        upE = self.move2(upE1)
        upE = self.conv2(upE)
        upE = self.bn2(upE) + upE1
        upE2 = self.act2(upE)

        upE = self.move3(upE2)
        upE = self.conv3(upE)
        upE = self.bn3(upE) + upE2
        upE3 = self.act3(upE)


        upE = (upE1 + upE2 + upE3) / 3
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

########################################################################
class SalsaNextFusion(nn.Module):
    def __init__(self, in_channels=8, nclasses=20, base_channels=32, img_feature_channels=[], softmax=True):
        super(SalsaNextFusion, self).__init__()
        # Embedding 32 Bit
        self.downCntx = ResContextBlock(in_channels, base_channels)
        # Embedding 1 Bit
        self.dropout_ratio = 0.2
        self.resBlock1 = ResBlock(base_channels, 2 * base_channels, self.dropout_ratio, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * base_channels, 2 * 2 * base_channels, self.dropout_ratio, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * base_channels, 2 * 4 * base_channels, self.dropout_ratio, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * base_channels, 2 * 8 * base_channels, self.dropout_ratio, pooling=True)

        self.upBlock1 = UpBlock(2 * 8 * base_channels, 8 * base_channels, self.dropout_ratio)
        self.upBlock2 = UpBlock(8 * base_channels, 4 * base_channels, self.dropout_ratio)
        self.upBlock3 = UpBlock(4 * base_channels, 2 * base_channels, self.dropout_ratio)
        self.upBlock4 = UpBlock(2 * base_channels, base_channels, self.dropout_ratio, drop_out=False)
        self.fusionblock_1 = ResidualBasedFusionBlock(base_channels * 2, img_feature_channels[0])  # 64 64
        self.fusionblock_2 = ResidualBasedFusionBlock(base_channels * 4, img_feature_channels[1])  # 128 128
        self.fusionblock_3 = ResidualBasedFusionBlock(base_channels * 8, img_feature_channels[2])  # 256 256
        self.fusionblock_4 = ResidualBasedFusionBlock(base_channels * 16, img_feature_channels[3])  # 512 512
        self.aspp = ASPP_binary(base_channels * 16, base_channels * 16)  # 512 512
        self.logits = nn.Conv2d(base_channels, nclasses, kernel_size=(1, 1))
        self.softmax = softmax

    def forward(self, x, img_feature=[]):
        downCntx = self.downCntx(x)

        down0c, down0b = self.resBlock1(downCntx)  # 32，fbl
        # down0c = self.fusionblock_1(down0c, img_feature[0].detach())

        down1c, down1b = self.resBlock2(down0c)  # 64，
        # down1c = self.fusionblock_2(down1c, img_feature[1].detach())

        down2c, down2b = self.resBlock3(down1c)  # 128，
        # down2c = self.fusionblock_3(down2c, img_feature[2].detach())

        down3c, down3b = self.resBlock4(down2c)  # 256，
        # down3c = self.fusionblock_4(down3c, img_feature[3].detach())

        down5c = self.aspp(down3c) # 32

        up4e = self.upBlock1(down5c, down3b) # 64
        up3e = self.upBlock2(up4e, down2b) # 128
        up2e = self.upBlock3(up3e, down1b) # 256
        up1e = self.upBlock4(up2e, down0b) # 512
        logits = self.logits(up1e)
        if self.softmax:
            logits = F.softmax(logits, dim=1)

        return logits


class RGBDecoder(nn.Module):
    def __init__(self, in_channels=[], nclasses=2, base_channels=64):
        super(RGBDecoder, self).__init__()
        self.upsamle = nn.Upsample(scale_factor=2, mode="nearest")
        self.pool4 = nn.AvgPool1d(32, stride=32)
        self.pool3 = nn.AvgPool1d(16, stride=16)
        self.pool2 = nn.AvgPool1d(8, stride=8)
        self.pool1 = nn.AvgPool1d(4, stride=4)

        self.move4 = LearnableBias(in_channels[3])
        self.conv4 = QuantizeConv2d(in_channels[3], base_channels, 3, 1, 1)
        self.act4 = RPReLU(base_channels)
        self.bn4 = nn.BatchNorm2d(base_channels)

        self.move3 = LearnableBias(in_channels[2] + base_channels)
        self.conv3 = QuantizeConv2d(in_channels[2] + base_channels, base_channels, 3, 1, 1)
        self.act3 = RPReLU(base_channels)
        self.bn3 = nn.BatchNorm2d(base_channels)

        self.move2 = LearnableBias(in_channels[1] + base_channels)
        self.conv2 = QuantizeConv2d(in_channels[1] + base_channels, base_channels, 3, 1, 1)
        self.act2 = RPReLU(base_channels)
        self.bn2 = nn.BatchNorm2d(base_channels)

        self.move1 = LearnableBias(in_channels[0] + base_channels)
        self.conv1 = QuantizeConv2d(in_channels[0] + base_channels, base_channels, 1)
        self.act1 = RPReLU(base_channels)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.token_mix = token_mixer(in_chn=base_channels)
        self.conv = nn.Conv2d(base_channels, 2, kernel_size=1, padding=0)

    def forward(self, inputs):
        up_4 = self.move4(inputs[3])
        up_4 = self.conv4(up_4)
        B, C, H, W = inputs[3].shape
        shorcut1 = self.pool4(inputs[3].permute(0, 2, 3, 1).flatten(1, 2)).permute(0, 2, 1).view(-1, C // 32, H,
                                                                                                 W).contiguous()
        up_4 = self.bn4(up_4) + shorcut1
        up_4 = self.act4(up_4)
        up_4a = self.upsamle(up_4)

        up_3 = self.move3(torch.cat((up_4a, inputs[2]), dim=1))
        up_3 = self.conv3(up_3)

        B, C, H, W = inputs[2].shape
        shorcut2 = self.pool3(inputs[2].permute(0, 2, 3, 1).flatten(1, 2)).permute(0, 2, 1).view(-1, C // 16, H,
                                                                                                 W).contiguous()
        up_3 = self.bn3(up_3) + shorcut2
        up_3a = self.act3(up_3)
        # 1 16 128 128
        up_3a = self.upsamle(up_3a)

        up_2 = self.move2(torch.cat((up_3a, inputs[1]), dim=1))
        up_2 = self.conv2(up_2)

        B, C, H, W = inputs[1].shape
        shorcut3 = self.pool2(inputs[1].permute(0, 2, 3, 1).flatten(1, 2)).permute(0, 2, 1).view(-1, C // 8, H,
                                                                                                 W).contiguous()
        up_2 = self.bn2(up_2) + shorcut3
        up_2a = self.act2(up_2)
        up_2a = self.upsamle(up_2a)

        up_1 = self.move1(torch.cat((up_2a, inputs[0]), dim=1))
        up_1 = self.conv1(up_1)

        B, C, H, W = inputs[0].shape
        shorcut4 = self.pool1(inputs[0].permute(0, 2, 3, 1).flatten(1, 2)).permute(0, 2, 1).view(-1, C // 4, H,
                                                                                                 W).contiguous()
        up_1 = self.bn1(up_1) + shorcut4
        up_1a = self.act1(up_1)
        up_1a = self.upsamle(up_1a)

        out = self.token_mix(up_1a)
        out = self.conv(out)
        out = F.softmax(out, dim=1)
        return out


class Pathfinder_Binary(nn.Module):
    def __init__(self, pcd_channels=5, img_channels=3, nclasses=20, base_channels=32,
                 checkpoint_file=[]):
        super(Pathfinder_Binary, self).__init__()

        self.camera_stream_encoder = CFI_ViT_224_w1a1(num_classes=nclasses)

        self.camera_stream_decoder = RGBDecoder(
            [64, 128, 256, 512],
            nclasses=nclasses, base_channels=1 * 16)

        self.lidar_stream = SalsaNextFusion(
            in_channels=pcd_channels, nclasses=nclasses, base_channels=base_channels,
            img_feature_channels=[64, 128, 256, 512])

    def forward(self, pcd_feature, img_feature):
        img_feature = self.camera_stream_encoder(img_feature)

        lidar_pred = self.lidar_stream(pcd_feature, img_feature)

        camera_pred = self.camera_stream_decoder(img_feature)

        return lidar_pred, camera_pred

