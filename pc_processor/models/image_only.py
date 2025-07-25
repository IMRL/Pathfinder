import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput

from transformers.utils import logging
from transformers import ViTConfig
from timm.models.layers import trunc_normal_, DropPath, to_2tuple


logger = logging.get_logger(__name__)


def get_model(model_config, imagenet_pretrained, checkpoint_file):
    config = ViTConfig.from_pretrained(model_config)
    config.num_hidden_layers = sum(config.depths)
    config.stages = generating_stage_per_depth(config.depths)
    config.drop_path = 0.1
    config.layer_norm_eps = 1e-5
    config.shift3 = False
    config.shift5 = False
    config.norm_layer = nn.LayerNorm
    config.disable_layerscale = False
    config.enable_cls_token = False
    config.gsb = False
    config.recu = False
    config.weight_bits = 1
    config.input_bits = 1
    config.some_fp = True
    model =BHViTModel(config=config)
    if imagenet_pretrained:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        checkpoint_model = checkpoint['model']
        weights_dict = {
              k[len("vit.") :] if "vit." in k else k: v
              for k, v in checkpoint_model.items()
        }

        for k in list(weights_dict.keys()):
            if "patch_embeddings" in k:
                del weights_dict[k]
            if "position_embeddings" in k:
                del weights_dict[k]
            if "classifier" in k:
                del weights_dict[k]
        state_dict = model.state_dict()
        # state_dict = check_dict(state_dict,checkpoint_model)
        state_dict.update(weights_dict)
        # interpolate position embedding
        model.load_state_dict(state_dict, strict=False)
    return model,config


""" PyTorch ViT model."""



########################                    ########################
class QuantizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, bias=True, config=None):
        super(QuantizeConv2d, self).__init__(*kargs, bias=bias)
        self.weight_bits = config.weight_bits
        self.input_bits = config.input_bits

    def forward(self, input):
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True),
                                    dim=1, keepdim=True).transpose(0, 1)
        real_weights = real_weights - real_weights.mean([1, 2, 3], keepdim=True)
        weight = torch.sign(real_weights)
        input = torch.sign(input)
        out = nn.functional.conv2d(input, weight, stride=self.stride, padding=self.padding, dilation=self.dilation,
                                   groups=self.groups) * scaling_factor

        if not self.bias is None:
            out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return out


#####################################################################

class RPReLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.move1 = nn.Parameter(torch.zeros([1, hidden_size, 1, 1]))
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Parameter(torch.zeros([1, hidden_size, 1, 1]))

    def forward(self, x):
        out = self.prelu((x - self.move1)) + self.move2
        return out


class LayerScale(nn.Module):
    def __init__(self, hidden_size, init_ones=True):
        super().__init__()
        if init_ones:
            self.alpha = nn.Parameter(torch.ones([1, hidden_size, 1, 1]) * 0.1)
        else:
            self.alpha = nn.Parameter(torch.zeros([1, hidden_size, 1, 1]))
        self.move = nn.Parameter(torch.zeros([1, hidden_size, 1, 1]))

    def forward(self, x):
        out = x * self.alpha + self.move
        return out


class BHViTEmbeddings(nn.Module):
    """
    Construct position and patch embeddings.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()

        self.patch_embeddings = BHViTPatchEmbeddings(config)
        self.num_patches = config.image_size // 4
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, config.hidden_size[0], self.num_patches, self.num_patches))
        trunc_normal_(self.position_embeddings, std=.02)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x) + self.position_embeddings
        return x


class BHViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, in_chans=3, out_chans=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d(64, eps=config.layer_norm_eps)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


#######################################
class Token_for_Attention(nn.Module):
    def __init__(self, dim, config, window_size=7):
        super(Token_for_Attention, self).__init__()
        self.window_size = window_size
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        windows = windows_split(x, self.window_size)
        token_all = self.norm(windows)
        return token_all


def windows_split(x, window_size):
    B, C, H, W, = x.shape
    x = x.permute(0, 2, 3, 1).view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C).permute(0, 3, 1, 2)
    return windows


#########################################
class BHViTSelfAttention(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()

        self.token_FA = Token_for_Attention(dim=config.hidden_size[config.stages[layer_num]], config=config,
                                            window_size=7)
        self.windows_size = 7
        self.num_attention_heads = config.num_attention_heads[config.stages[layer_num]]
        self.attention_head_size = int(
            config.hidden_size[config.stages[layer_num]] / config.num_attention_heads[config.stages[layer_num]])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.moveq = nn.Parameter(torch.zeros([1, config.hidden_size[config.stages[layer_num]], 1, 1]))
        self.movek = nn.Parameter(torch.zeros([1, config.hidden_size[config.stages[layer_num]], 1, 1]))
        self.movev = nn.Parameter(torch.zeros([1, config.hidden_size[config.stages[layer_num]], 1, 1]))

        self.query = QuantizeConv2d(config.hidden_size[config.stages[layer_num]], self.all_head_size, 1, bias=True,
                                    config=config)
        self.key = QuantizeConv2d(config.hidden_size[config.stages[layer_num]], self.all_head_size, 1, bias=True,
                                  config=config)
        self.value = QuantizeConv2d(config.hidden_size[config.stages[layer_num]], self.all_head_size, 1, bias=True,
                                    config=config)

        self.normq = nn.BatchNorm2d(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.normk = nn.BatchNorm2d(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.normv = nn.BatchNorm2d(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)

        self.rpreluq = RPReLU(config.hidden_size[config.stages[layer_num]])
        self.rpreluk = RPReLU(config.hidden_size[config.stages[layer_num]])
        self.rpreluv = RPReLU(config.hidden_size[config.stages[layer_num]])

        self.moveq2 = nn.Parameter(torch.zeros([1, config.hidden_size[config.stages[layer_num]], 1, 1]))
        self.movek2 = nn.Parameter(torch.zeros([1, config.hidden_size[config.stages[layer_num]], 1, 1]))
        self.movev2 = nn.Parameter(torch.zeros([1, config.hidden_size[config.stages[layer_num]], 1, 1]))

        self.att_prob_clip = nn.Parameter(torch.tensor(0.005))

        self.norm_context = config.norm_layer(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)

        self.rprelu_context = RPReLU(config.hidden_size[config.stages[layer_num]])

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.parm = nn.Parameter(0.5 * torch.ones([1, config.hidden_size[config.stages[layer_num]], 1, 1]),
                                 requires_grad=True)

    def transpose_for_scores(self, x):
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def window_reverse(self, windows, window_size, H, W, B):
        windows = windows.permute(0, 2, 3, 1)
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[3], H, W)
        return x

    def forward(self, hidden_states):
        B, C, H, W = hidden_states.shape
        hidden_states = self.token_FA(hidden_states)
        mixed_query_layer = self.normq(self.query(hidden_states + self.moveq)) + hidden_states
        mixed_key_layer = self.normk(self.key(hidden_states + self.movek)) + hidden_states
        mixed_value_layer = self.normv(self.value(hidden_states + self.movev)) + hidden_states
        mixed_query_layer = self.rpreluq(mixed_query_layer)
        mixed_key_layer = self.rpreluk(mixed_key_layer)
        mixed_value_layer = self.rpreluv(mixed_value_layer)
        query_layer = mixed_query_layer + self.moveq2
        key_layer = mixed_key_layer + self.movek2
        value_layer = mixed_value_layer + self.movev2
        query_layer = torch.sign(query_layer)
        key_layer = torch.sign(key_layer)
        value_layer = torch.sign(value_layer)
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = torch.round(attention_probs / self.att_prob_clip).clamp(0.0, 1.0)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        context_layer = torch.matmul(attention_probs, value_layer) * self.att_prob_clip
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # BHNC1 -> BNHC1
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # BNHC1 -> BN+ C
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = self.norm_context(context_layer).permute(0, 2, 1).view(-1, C, 7,
                                                                               7).contiguous() + mixed_query_layer + mixed_key_layer + mixed_value_layer
        context_layer = self.rprelu_context(context_layer)
        context_layer = self.window_reverse(context_layer, self.windows_size, H, W, B)
        outputs = context_layer

        return outputs


class BHViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config, layer_num):
        super().__init__()

        self.dense = QuantizeConv2d(config.hidden_size[config.stages[layer_num]],
                                    config.hidden_size[config.stages[layer_num]], 1, bias=True, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.move = nn.Parameter(torch.zeros([1, config.hidden_size[config.stages[layer_num]], 1, 1]))
        self.norm = nn.BatchNorm2d(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.rprelu = RPReLU(config.hidden_size[config.stages[layer_num]])

        self.layerscale = LayerScale(
            config.hidden_size[config.stages[layer_num]]) if not config.disable_layerscale else nn.Identity()

    def forward(self, hidden_states):
        out = self.norm(self.dense(hidden_states + self.move)) + hidden_states
        out = self.rprelu(out)
        out = self.dropout(out)

        out = self.layerscale(out)

        return out


class BHViTAttention(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.attention = BHViTSelfAttention(config, layer_num)
        self.output = BHViTSelfOutput(config, layer_num)

    def forward(self, hidden_states):
        self_outputs = self.attention(hidden_states)

        outputs = self.output(self_outputs)
        return outputs


class ViTIntermediate(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()

        self.dense = QuantizeConv2d(config.hidden_size[config.stages[layer_num]],
                                    config.intermediate_size[config.stages[layer_num]], 1, bias=True, config=config)

        self.move = nn.Parameter(torch.zeros([1, config.hidden_size[config.stages[layer_num]], 1, 1]))
        self.norm = nn.BatchNorm2d(config.intermediate_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.rprelu = RPReLU(config.intermediate_size[config.stages[layer_num]])
        self.expansion_ratio = config.intermediate_size[config.stages[layer_num]] // config.hidden_size[
            config.stages[layer_num]]

    def forward(self, hidden_states):
        out = self.norm(self.dense(hidden_states + self.move)) + torch.concat(
            [hidden_states for _ in range(self.expansion_ratio)], dim=1)
        out = self.rprelu(out)

        return out


class ViTOutput(nn.Module):
    def __init__(self, config, layer_num, drop_path=0.0):
        super().__init__()
        self.dense = QuantizeConv2d(config.intermediate_size[config.stages[layer_num]],
                                    config.hidden_size[config.stages[layer_num]], 1, bias=True, config=config)

        self.move = nn.Parameter(torch.zeros([1, config.intermediate_size[config.stages[layer_num]], 1, 1]))
        self.norm = nn.BatchNorm2d(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.rprelu = RPReLU(config.hidden_size[config.stages[layer_num]])
        self.ratio = config.intermediate_size[config.stages[layer_num]] // config.hidden_size[config.stages[layer_num]]
        self.pooling = nn.AvgPool1d(self.ratio)
        self.layerscale = LayerScale(
            config.hidden_size[config.stages[layer_num]]) if not config.disable_layerscale else nn.Identity()

    def forward(self, hidden_states):
        B, C, H, W = hidden_states.shape
        out = self.norm(self.dense(hidden_states + self.move)) + self.pooling(
            hidden_states.permute(0, 2, 3, 1).flatten(1, 2)).permute(0, 2, 1).view(-1, C // self.ratio, H,
                                                                                   W).contiguous()
        out = self.rprelu(out)
        out = self.layerscale(out)
        return out


class LearnableBiasnn(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBiasnn, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, out_chn, 1, 1]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class Shift(nn.Module):
    def __init__(self):
        super(Shift, self).__init__()

    def forward(self, x, dim):
        x1 = torch.roll(x, 1, dims=dim)  # [:,:,1:,:]
        x2 = torch.roll(x, -1, dims=dim)  # [:,:,:-1,:]
        x = x + x1 + x2
        return x / 3


class Shift2(nn.Module):
    def __init__(self):
        super(Shift2, self).__init__()

    def forward(self, x, dim):
        x1 = torch.roll(x, 1, dims=dim)  # [:,:,1:,:]
        x2 = torch.roll(x, -1, dims=dim)  # [:,:,:-1,:]
        x3 = torch.roll(x, 2, dims=dim)  # [:,:,1:,:]
        x4 = torch.roll(x, -2, dims=dim)  # [:,:,:-1,:]
        x = x + x1 + x2 + x3 + x4
        return x / 5


class Shift_channel_mix(nn.Module):
    def __init__(self, shift_size=1):
        super(Shift_channel_mix, self).__init__()
        self.shift_size = shift_size

    def forward(self, x):
        x1, x2, x3, x4 = x.chunk(4, dim=1)

        x1 = torch.roll(x1, self.shift_size, dims=2)  # [:,:,1:,:]

        x2 = torch.roll(x2, -self.shift_size, dims=2)  # [:,:,:-1,:]

        x3 = torch.roll(x3, self.shift_size, dims=3)  # [:,:,:,1:]

        x4 = torch.roll(x4, -self.shift_size, dims=3)  # [:,:,:,:-1]

        x = torch.cat([x1, x2, x3, x4], 1)

        return x


class token_mixer(nn.Module):
    def __init__(self, in_chn, config, pool=2, kernel_size=3, stride=1, padding=1):
        super(token_mixer, self).__init__()
        self.move = LearnableBiasnn(in_chn)
        self.cov1 = QuantizeConv2d(in_chn, in_chn, kernel_size, stride, padding, 1, 1, bias=True, config=config)
        self.pool1 = nn.MaxPool2d(pool, stride=pool)
        self.cov2 = QuantizeConv2d(in_chn, in_chn, kernel_size, stride, padding, 1, 1, bias=True, config=config)
        self.pool2 = nn.AvgPool2d(pool, stride=pool)
        self.cov3 = QuantizeConv2d(in_chn, in_chn, kernel_size, stride, padding, 1, 1, bias=True, config=config)
        self.norm = nn.BatchNorm2d(in_chn, eps=config.layer_norm_eps)
        self.act1 = RPReLU(in_chn)
        self.act2 = RPReLU(in_chn)
        self.act3 = RPReLU(in_chn)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.move(x)
        x1 = self.cov1(x)
        x1 = self.act1(x1)
        x2 = self.pool1(x)
        x2 = self.cov2(x)
        x2 = torch.nn.functional.interpolate(x2, size=H, mode='nearest')
        x2 = self.act2(x2)

        x3 = self.pool2(x)
        x3 = self.cov3(x)
        x3 = torch.nn.functional.interpolate(x3, size=H, mode='nearest')
        x3 = self.act3(x3)

        x = self.norm(x1 + x2 + x3)
        return x


class GCLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config, layer_num, drop_path=0.0):
        super().__init__()
        self.GC = token_mixer(config.hidden_size[config.stages[layer_num]], config)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.config = config
        self.intermediate = ViTIntermediate(config, layer_num)
        self.output = ViTOutput(config, layer_num, drop_path=drop_path)
        self.norm_before = nn.BatchNorm2d(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.norm_after = nn.BatchNorm2d(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)

        self.shift3 = config.shift3
        self.shift5 = config.shift5

        if self.shift5:
            print("Using shift 5 Residual")
            # shift_window = 5
            self.shift_w5 = Shift2()
            self.layerscale_w5 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)
            self.shift_h5 = Shift2()
            self.layerscale_h5 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)
            self.shift_ch5 = Shift_channel_mix(2)
            self.layerscale_ch5 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)
        if self.shift3:
            print("Using shift 3 Residual")
            # shift_window = 3
            self.shift_w3 = Shift()
            self.layerscale_w3 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)
            self.shift_h3 = Shift()
            self.layerscale_h3 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)
            self.shift_ch3 = Shift_channel_mix(1)
            self.layerscale_ch3 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)

    def forward(self, hidden_states):

        B, C, H, W = hidden_states.shape
        hidden_states_norm = self.norm_before(hidden_states)
        self_GC_outputs = self.GC(hidden_states_norm)
        # first residual connection
        hidden_states = self_GC_outputs + hidden_states

        # in ViT, layernorm is also applied after self-attention
        hidden_states_norm = self.norm_after(hidden_states)
        layer_output = self.intermediate(hidden_states_norm)

        # second residual connection is done here
        layer_output = self.output(layer_output) + hidden_states
        if self.shift3:
            layer_output += self.layerscale_h3(self.shift_h3(hidden_states_norm, 2))
            layer_output += self.layerscale_w3(self.shift_w3(hidden_states_norm, 3))
            layer_output += self.layerscale_ch3(self.shift_ch3(hidden_states_norm))
        if self.shift5:
            layer_output += self.layerscale_h5(self.shift_h5(hidden_states_norm, 2))
            layer_output += self.layerscale_w5(self.shift_w5(hidden_states_norm, 3))
            layer_output += self.layerscale_ch5(self.shift_ch5(hidden_states_norm))
        outputs = layer_output

        return outputs


class BHViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config, layer_num, drop_path=0.0):
        super().__init__()
        self.attention = BHViTAttention(config, layer_num)
        self.intermediate = ViTIntermediate(config, layer_num)
        self.output = ViTOutput(config, layer_num, drop_path=drop_path)
        self.norm_before = nn.BatchNorm2d(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)
        self.norm_after = nn.BatchNorm2d(config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps)

        self.shift3 = config.shift3
        self.shift5 = config.shift5

        if self.shift5:
            print("Using shift 5 Residual")
            # shift_window = 5
            self.shift_w5 = Shift2()
            self.layerscale_w5 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)
            self.shift_h5 = Shift2()
            self.layerscale_h5 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)
            self.shift_ch5 = Shift_channel_mix(2)
            self.layerscale_ch5 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)
        if self.shift3:
            print("Using shift 3 Residual")
            # shift_window = 3
            self.shift_w3 = Shift()
            self.layerscale_w3 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)
            self.shift_h3 = Shift()
            self.layerscale_h3 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)
            self.shift_ch3 = Shift_channel_mix(1)
            self.layerscale_ch3 = LayerScale(config.hidden_size[config.stages[layer_num]], init_ones=False)

    def forward(self, hidden_states):
        B, C, H, W = hidden_states.shape
        hidden_states_norm = self.norm_before(hidden_states)
        self_attention_outputs = self.attention(hidden_states_norm)
        # first residual connection
        hidden_states = self_attention_outputs + hidden_states

        # in ViT, layernorm is also applied after self-attention
        hidden_states_norm = self.norm_after(hidden_states)
        layer_output = self.intermediate(hidden_states_norm)

        # second residual connection is done here
        layer_output = self.output(layer_output) + hidden_states
        if self.shift3:
            layer_output += self.layerscale_h3(self.shift_h3(hidden_states_norm, 2))
            layer_output += self.layerscale_w3(self.shift_w3(hidden_states_norm, 3))
            layer_output += self.layerscale_ch3(self.shift_ch3(hidden_states_norm))
        if self.shift5:
            layer_output += self.layerscale_h5(self.shift_h5(hidden_states_norm, 2))
            layer_output += self.layerscale_w5(self.shift_w5(hidden_states_norm, 3))
            layer_output += self.layerscale_ch5(self.shift_ch5(hidden_states_norm))
        return layer_output


class BinaryPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=2, in_dim=3, out_dim=64, config=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0], img_size[1]
        self.num_patches = self.H * self.W

        self.norm0 = nn.BatchNorm2d(in_dim)

        self.move = nn.Parameter(torch.zeros(1, in_dim, 1, 1))
        self.proj = QuantizeConv2d(in_dim, out_dim, self.patch_size, self.patch_size, bias=False, config=config)
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(out_dim)
        self.rprelu = RPReLU(out_dim)

        self.position_embeddings = nn.Parameter(torch.zeros(1, out_dim, img_size[0] // 2, img_size[0] // 2))
        trunc_normal_(self.position_embeddings, std=.02)

    def forward(self, hidden_states):
        B, C, H, W = hidden_states.shape
        hidden_states = self.norm0(hidden_states)
        residual = self.pool(hidden_states)
        hidden_states = self.proj(hidden_states + self.move.expand_as(hidden_states))
        B2, C2, H2, W2 = hidden_states.shape
        residual = torch.concat([residual for _ in range(C2 // C)], dim=1)

        hidden_states = self.norm(hidden_states) + residual
        hidden_states = self.rprelu(hidden_states)

        return hidden_states + self.position_embeddings


class BHViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dpr = [x.item() for x in
               torch.linspace(0, config.drop_path, config.num_hidden_layersA + config.num_hidden_layersB)]
        self.layerA = nn.ModuleList([GCLayer(config, i, drop_path=dpr[i]) for i in range(config.num_hidden_layersA)])
        self.layerB = nn.ModuleList(
            [BHViTLayer(config, i + config.num_hidden_layersA, drop_path=dpr[i + config.num_hidden_layersA]) for i in
             range(config.num_hidden_layersB)])
        self.gradient_checkpointing = False
        self.patch_embed1 = BinaryPatchEmbed(56, in_dim=config.hidden_size[0], out_dim=config.hidden_size[1],
                                             config=config)
        self.patch_embed2 = BinaryPatchEmbed(28, in_dim=config.hidden_size[1], out_dim=config.hidden_size[2],
                                             config=config)
        self.patch_embed3 = BinaryPatchEmbed(14, in_dim=config.hidden_size[2], out_dim=config.hidden_size[3],
                                             config=config)
        self.depths = config.depths

    def forward(self, hidden_states):
        ##### stage 12
        for i, layer_module in enumerate(self.layerA):
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs
            if i == self.depths[0] - 1:
                hidden_states = self.patch_embed1(hidden_states)
            elif i == self.depths[0] + self.depths[1] - 1:
                hidden_states = self.patch_embed2(hidden_states)
        ##### stage 34
        for i, layer_module in enumerate(self.layerB):
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs
            if i == self.depths[2] - 1:
                hidden_states = self.patch_embed3(hidden_states)
        return hidden_states


class BHViTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = BHViTEmbeddings(config)
        self.encoder = BHViTEncoder(config)

        self.layernorm = config.norm_layer(config.hidden_size[3], eps=config.layer_norm_eps)

    def forward(self, x):
        embedding_output = self.embeddings(x)
        encoder_outputs = self.encoder(embedding_output)
        sequence_output = self.layernorm(encoder_outputs.permute(0, 2, 3, 1).flatten(1, 2))
        return sequence_output


def generating_stage_per_depth(depths):
    i = 0
    stage_per_depth = []
    current_stage_depth = depths[i]
    while True:
        current_stage_depth -= 1
        stage_per_depth.append(i)
        if current_stage_depth == 0:
            i += 1
            if i == len(depths):
                break
            current_stage_depth = depths[i]
    return stage_per_depth


class dbViTForImageClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        config.num_hidden_layers = sum(config.depths)
        config.stages = generating_stage_per_depth(config.depths)

        self.vit = BHViTModel(config)
        self.config = config

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size[3],
                                    config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.apply(self.init_weights)

    @torch.no_grad()
    def init_weights(module: nn.Module, name: str = ''):
        """ ViT weight initialization, original timm impl (for reproducibility) """
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.BatchNorm1d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, pixel_values):
        sequence_output = self.vit(pixel_values)
        logits = self.classifier(torch.mean(sequence_output, dim=1))
        return logits

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'position_embeddings', 'cls_token', 'dist_token'}


class Pathfinder_ImageOnly(nn.Module):
    def __init__(self, encoder_config, imagenet_pretrained, checkpoint_file, nclasses=1000):
        super(Pathfinder_ImageOnly, self).__init__()
        self.camera_stream_encoder, config = get_model(model_config=encoder_config,
                                                       imagenet_pretrained=imagenet_pretrained,
                                                       checkpoint_file=checkpoint_file)
    def forward(self, img_feature):
        camera_pred = self.camera_stream_encoder(img_feature)
        return camera_pred