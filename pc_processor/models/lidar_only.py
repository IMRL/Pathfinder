# !/usr/bin/env python3
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from .salsanext import SalsaNext

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from .weight_init import named_apply, lecun_normal_

_logger = logging.getLogger(__name__)


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
        self.bias = nn.Parameter(torch.zeros([1, out_chn, 1, 1]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class LearnableBiastt(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBiastt, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, 1, out_chn]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class RPReLU(nn.Module):
    def __init__(self, out_chn, toc=1):
        super(RPReLU, self).__init__()
        if toc == 1:
            self.move1 = LearnableBiasnn(out_chn)
            self.move2 = LearnableBiasnn(out_chn)
        else:
            self.move1 = LearnableBiastt(out_chn)
            self.move2 = LearnableBiastt(out_chn)
        self.act = nn.PReLU(out_chn)

    def forward(self, x):
        x = self.move1(x)
        x = self.act(x)
        x = self.move2(x)
        return x


class LearnableBias(nn.Module):
    def __init__(self, out_chn, head):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros([head, 1, out_chn // head]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


class Mlp_cnn(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, bias=True)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm2d(hidden_features)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.pool1(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + x1


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
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PosEmb_window(nn.Module):
    def __init__(self, dim, splitdim=49, patch_num=65):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv1d(3, 256, 1, bias=True),
                                 nn.ReLU(),
                                 nn.Conv1d(256, dim, 1, bias=False))
        self.grid_exists = False
        self.pos_emb = None
        self.rp = True
        relative_bias = torch.zeros(1, patch_num, dim)
        depth_mask = torch.cat(torch.zeros(1, splitdim), torch.ones(1, patch_num - self.splitdim), dim=1)
        self.register_buffer("relative_bias", relative_bias)
        self.register_buffer("depth_mask", depth_mask)

    def forward(self, input_tensor):
        patch_num = input_tensor.shape[1]
        if self.rp:
            relative_coords_h = torch.arange(0, patch_num - 1, device=input_tensor.device, dtype=input_tensor.dtype)
            relative_coords_h -= patch_num // 2
            relative_coords_h /= (patch_num // 2)

            relative_coords = (relative_coords_h + self.depth_mask) / 2
            relative_coords_table = relative_coords
            self.pos_emb = self.mlp(relative_coords_table.unsqueeze(0).unsqueeze(2))
            self.relative_bias = self.pos_emb
            return input_tensor + self.pos_emb
        else:
            return input_tensor + self.relative_bias


class token_mixer(nn.Module):
    def __init__(self, in_chn, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1, padding='same'):
        super(token_mixer, self).__init__()
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
        self.act = RPReLU(in_chn)
        self.a = nn.Parameter(0.25 * torch.ones([1, in_chn, 1, 1]), requires_grad=True)
        self.b = nn.Parameter(0.25 * torch.ones([1, in_chn, 1, 1]), requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        real_weights2 = self.weights2.view(self.shape)
        real_weights3 = self.weights3.view(self.shape)
        x1 = F.conv2d(x, real_weights, stride=self.stride, padding=self.padding, dilation=self.dilation1,
                      groups=self.shape[0])
        x2 = F.conv2d(x, real_weights2, stride=self.stride, padding=self.padding, dilation=self.dilation2,
                      groups=self.shape[0])
        x3 = F.conv2d(x, real_weights3, stride=self.stride, padding=self.padding, dilation=self.dilation3,
                      groups=self.shape[0])
        x = self.norm(
            (1.0 - self.a - self.b).expand_as(x1) * x1 + self.a.expand_as(x2) * x2 + self.b.expand_as(x3) * x3)
        x = self.act(x)
        return x


class Token_for_Attention(nn.Module):
    def __init__(self, dim, window_size=8):
        super(Token_for_Attention, self).__init__()
        self.window_size = window_size
        self.merge_avg = nn.AvgPool2d(kernel_size=window_size, stride=window_size, padding=0)
        self.merge_max = nn.MaxPool2d(kernel_size=window_size, stride=window_size, padding=0)
        self.mlp = Mlp(dim)
        self.a1 = nn.Parameter(0.25 * torch.ones([1, 1, dim]), requires_grad=True)
        self.a2 = nn.Parameter(0.25 * torch.ones([1, 1, dim]), requires_grad=True)
        self.norm = nn.LayerNorm(dim)

    def Merge_token(self, x):
        B, C, H, W, = x.shape
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


class Attention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.token_FA = Token_for_Attention(dim=dim, window_size=window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.windows_size = window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.parm = nn.Parameter(0.5 * torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.upsample = nn.Upsample(scale_factor=window_size, mode='nearest')

    def window_reverse(self, windows, window_size, H, W, B):
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
        return x

    def window_reverse_high(self, windows, window_size, H, W, B):
        x = windows.view(B, H // window_size * W // window_size, H // window_size, W // window_size, -1)
        x = torch.mean(x, dim=1).permute(0, 3, 1, 2)
        x = self.upsample(x)
        return x

    def token_split(self, x, split_dim, H, W, B):
        x1, x2 = torch.split(x, split_dim, dim=1)
        x1 = self.window_reverse(x1, self.windows_size, H, W, B)
        x2 = self.window_reverse_high(x2, self.windows_size, H, W, B)
        return x1 * self.parm.expand_as(x1) + x2 * (1.0 - self.parm).expand_as(x2)

    def forward(self, x):
        B0, C0, H0, W0, = x.shape
        x, split_dim = self.token_FA(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.token_split(x, split_dim, H0, W0, B0)
        return x


class Block(nn.Module):
    def __init__(self, embed_dims, mlp_ratio, toa, window_size=8, qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(embed_dims)
        if toa == 0:
            self.token_mix = token_mixer(in_chn=embed_dims)
        else:
            self.token_mix = Attention(dim=embed_dims, window_size=window_size, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                       proj_drop=drop)
        self.norm2 = nn.BatchNorm2d(embed_dims)
        self.mlp = Mlp_cnn(in_features=embed_dims, hidden_features=mlp_ratio * embed_dims)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.token_mix(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Embedding(nn.Module):
    def __init__(self, in_chans=3, out_chans=32, norm_layer=nn.BatchNorm2d, use_norm=True):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, stride=2, bias=True)
        self.norm = norm_layer(out_chans) if use_norm else nn.Identity()
        self.act = RPReLU(out_chans, toc=1)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
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
        x = self.norm(x)
        x = self.act(x)
        return x


class CFI_ViT(nn.Module):
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
        self.dist_token = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, (
                    depth[0] + depth[1] + depth[2] + depth[3]))]  # stochastic depth decay rule
        self.blocks1 = nn.Sequential(*[
            Block(embed_dims[0], mlp_ratio=mlp_ratios[0],
                  qkv_bias=qkv_bias, toa=toa[0], drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer,
                  act_layer=act_layer)
            for i in range(layers[0])])
        self.down1 = Downsample_CNN(embed_dims[0], embed_dims[1])
        self.blocks2 = nn.Sequential(*[
            Block(embed_dims[1], mlp_ratio=mlp_ratios[1],
                  qkv_bias=qkv_bias, toa=toa[1], drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + 2],
                  norm_layer=norm_layer,
                  act_layer=act_layer)
            for i in range(layers[1])])
        self.down2 = Downsample_CNN(embed_dims[1], embed_dims[2])
        self.blocks3 = nn.Sequential(*[
            Block(embed_dims[2], mlp_ratio=mlp_ratios[2], window_size=8,
                  qkv_bias=qkv_bias, toa=toa[2], drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + 4],
                  norm_layer=norm_layer,
                  act_layer=act_layer)
            for i in range(layers[2])])
        self.down3 = Downsample_CNN(embed_dims[2], embed_dims[3])
        self.blocks4 = nn.Sequential(*[
            Block(embed_dims[3], mlp_ratio=mlp_ratios[3], window_size=8,
                  qkv_bias=qkv_bias, toa=toa[3], drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + 8],
                  norm_layer=norm_layer,
                  act_layer=act_layer)
            for i in range(layers[3])])
        self.norm = norm_layer(embed_dims[3])
        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        # trunc_normal_(self.embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
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

    @torch.jit.ignore()
    # def load_pretrained(self, checkpoint_path, prefix=''):
    #     _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

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
        # x=x.permute(0, 2, 3, 1).flatten(1,2)
        for i, layer in enumerate(list(self.blocks3)):
            x = layer(x)
        x3 = x
        x = self.down3(x)
        for i, layer in enumerate(list(self.blocks4)):
            x = layer(x)
        x4 = x
        return [x1, x2, x3, x4]


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
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
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def resize_pos_embed(posemb, posemb_new):
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    out_dict = {}
    if 'model' in state_dict:
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


def CFI_ViT_224(num_classes, imagenet_pretrained, checkpoint_file, **kwargs):
    layers = [2, 2, 6, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 256, 512]
    toa = [0, 0, 1, 1]
    depth = [2, 2, 6, 2]
    model = CFI_ViT(layers, mlp_ratios, embed_dims, toa=toa, num_classes=num_classes, depth=depth, qkv_bias=True,
                    norm_layer=nn.BatchNorm2d, **kwargs)
    model.default_cfg = _cfg()
    if imagenet_pretrained:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        checkpoint_model = checkpoint['model']
        keys_to_remove = ['embed.proj.weight', 'embed.proj.bias', 'embed.norm.weight', 'embed.norm.bias',
                          'embed.norm.running_mean', 'embed.norm.running_var', 'embed.act.move1.bias',
                          'embed.act.move2.bias', 'embed.act.act.weight', 'down1.proj.weight', 'down2.proj.weight',
                          'down3.proj.weight', ]
        for key in keys_to_remove:
            checkpoint_model.pop(key, None)
        state_dict = model.state_dict()
        state_dict.update(checkpoint_model)
        model.load_state_dict(state_dict, strict=False)
    return model


class ResidualBasedFusionBlock(nn.Module):
    def __init__(self, pcd_channels, img_channels):
        super(ResidualBasedFusionBlock, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(pcd_channels + img_channels, pcd_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(pcd_channels)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(pcd_channels, pcd_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(pcd_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(pcd_channels, pcd_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(pcd_channels),
            nn.Sigmoid()
        )

    def forward(self, pcd_feature, img_feature):
        cat_feature = torch.cat((pcd_feature, img_feature), dim=1)
        fuse_out = self.fuse_conv(cat_feature)
        attention_map = self.attention(fuse_out)
        out = fuse_out * attention_map + pcd_feature
        return out


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(
            image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([
            image_features, atrous_block1, atrous_block6,
            atrous_block12, atrous_block18], dim=1))
        return net


class SalsaNextFusion(SalsaNext):
    def __init__(self, in_channels=8, nclasses=20, base_channels=32, img_feature_channels=[]):
        super(SalsaNextFusion, self).__init__(in_channels=in_channels, base_channels=base_channels,
                                              nclasses=nclasses, softmax=True)

        self.fusionblock_1 = ResidualBasedFusionBlock(self.base_channels * 2, img_feature_channels[0])
        self.fusionblock_2 = ResidualBasedFusionBlock(self.base_channels * 4, img_feature_channels[1])
        self.fusionblock_3 = ResidualBasedFusionBlock(self.base_channels * 8, img_feature_channels[2])
        self.fusionblock_4 = ResidualBasedFusionBlock(self.base_channels * 8, img_feature_channels[3])

        self.aspp = ASPP(self.base_channels * 8, self.base_channels * 8)

    def forward(self, x, img_feature=[]):
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)

        down5c = self.aspp(self.resBlock5(down3c))

        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        logits = self.logits(up1e)
        if self.softmax:
            logits = F.softmax(logits, dim=1)

        return logits


class RGBDecoder(nn.Module):
    def __init__(self, in_channels=[], nclasses=2, base_channels=64):
        super(RGBDecoder, self).__init__()

        self.up_4a = nn.Sequential(
            nn.Conv2d(in_channels[3], base_channels, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
        self.up_3a = nn.Sequential(
            nn.Conv2d(in_channels[2] + base_channels, base_channels, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="nearest")

        )
        self.up_2a = nn.Sequential(
            nn.Conv2d(in_channels[1] + base_channels, base_channels, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
        self.up_1a = nn.Sequential(
            nn.Conv2d(in_channels[0] + base_channels, base_channels, 1),
            nn.PReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
        self.token_mix = token_mixer(in_chn=base_channels)
        self.conv = nn.Conv2d(base_channels, 2, kernel_size=1, padding=0)

    def forward(self, inputs):
        up_4a = self.up_4a(inputs[3])
        up_3a = self.up_3a(torch.cat((up_4a, inputs[2]), dim=1))
        up_2a = self.up_2a(torch.cat((up_3a, inputs[1]), dim=1))
        up_1a = self.up_1a(torch.cat((up_2a, inputs[0]), dim=1))
        out = self.token_mix(up_1a)
        out = self.conv(out)
        out = F.softmax(out, dim=1)
        return out


class Pathfinder_LidarOnly(nn.Module):
    def __init__(self, pcd_channels=5, img_channels=3, nclasses=20, base_channels=32,
                 imagenet_pretrained=True, checkpoint_file=[]):
        super(Pathfinder_LidarOnly, self).__init__()

        self.camera_stream_encoder = CFI_ViT_224(num_classes=nclasses, imagenet_pretrained=imagenet_pretrained, checkpoint_file=checkpoint_file)

        self.camera_stream_decoder = RGBDecoder(
            [64,128,256,512],
            nclasses=nclasses, base_channels=1 * 16)

        self.lidar_stream = SalsaNextFusion(
            in_channels=pcd_channels, nclasses=nclasses, base_channels=base_channels,
            img_feature_channels=[64, 128, 256, 512])

    def forward(self, pcd_feature, img_feature):
        img_feature = self.camera_stream_encoder(img_feature)

        lidar_pred = self.lidar_stream(pcd_feature, img_feature)

        camera_pred = self.camera_stream_decoder(img_feature)

        return lidar_pred, camera_pred
