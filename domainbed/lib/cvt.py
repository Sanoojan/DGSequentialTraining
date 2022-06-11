from functools import partial
from itertools import repeat
import collections.abc

import logging
import os
from collections import OrderedDict
import itertools

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from domainbed.lib.visiontransformer import VisionTransformer,_cfg

from timm.models.layers import DropPath, trunc_normal_

# from .registry import register_model
from timm.models.registry import register_model

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


__all__ = [
    "small_cvt"
]

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


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


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 with_di_token=False,
                 attn_sep_mask=False,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token
        self.with_di_token = with_di_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.attn_sep_mask=with_di_token # Fixed mask if Di token introduced
        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        
        if (self.with_cls_token and self.with_di_token):
            cls_token, x,di_token = torch.split(x, [1, h*w,1], 1) 
        elif self.with_cls_token:
            cls_token, x = torch.split(x, [1, h*w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if (self.with_cls_token and self.with_di_token):
            q = torch.cat((cls_token, q,di_token), dim=1)
            k = torch.cat((cls_token, k,di_token), dim=1)
            v = torch.cat((cls_token, v,di_token), dim=1)
        elif self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, h, w):
        
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)
     
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        if(self.attn_sep_mask):
            B,N,C=x.shape
            B,h,kN,kC=k.shape
            permutations=[[0,kN-1],[N-1,0]]
            # mask to avoid attention between token
            for p in permutations:
                attn_score[:,:,p[0],p[1]]-=float('inf')
        
        attn = F.softmax(attn_score, dim=-1)
        # print(attn.shape)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T-1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
            hasattr(module, 'conv_proj_q')
            and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
            hasattr(module, 'conv_proj_k')
            and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
            hasattr(module, 'conv_proj_v')
            and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']
        self.with_di_token = kwargs['with_di_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, h, w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 dtokens=1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']
        with_di_token= kwargs['with_di_token']
        if with_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.cls_token = None
        if with_di_token:
            self.di_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.di_token=None
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if self.di_token is not None:
            trunc_normal_(self.di_token, std=.02)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = None
        if(self.di_token is not None and self.cls_token is not None):
            cls_tokens = self.cls_token.expand(B, -1, -1)
            di_tokens=self.di_token.expand(B,-1,-1)
            x = torch.cat((cls_tokens, x,di_tokens), dim=1)

        elif self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)


        patches = []
        class_tokens = []
        dinv_tokens=[]
        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)
            if (self.di_token is not None and self.cls_token is not None ):
                cls_tokens_, x_,di_tokens_ = torch.split(x, [1, H * W, 1], 1)
                x_ = rearrange(x_, 'b (h w) c -> b c h w', h=H, w=W)
                patches.append(x_)
                class_tokens.append(cls_tokens_) 
                dinv_tokens.append(di_tokens_) 
            elif self.cls_token is not None:
                cls_tokens_, x_ = torch.split(x, [1, H * W], 1)
                x_ = rearrange(x_, 'b (h w) c -> b c h w', h=H, w=W)
                patches.append(x_)
                class_tokens.append(cls_tokens_)
        if self.di_token is not None and self.cls_token is not None:
            cls_tokens, x,di_tokens = torch.split(x, [1, H*W,1], 1)
        elif self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H*W], 1)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, cls_tokens, patches, class_tokens,dinv_tokens


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 dtokens=1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'with_di_token':spec['DI_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                dtokens=dtokens,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec['CLS_TOKEN'][-1]
        self.di_token = spec['DI_TOKEN'][-1]
        # Classifier head
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    if 'pos_embed' in k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()
                        logging.info(
                            '=> load_pretrained: resized variant: {} to {}'
                            .format(size_pretrained, size_new)
                        )

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))

                        logging.info(
                            '=> load_pretrained: grid-size from {} to {}'
                            .format(gs_old, gs_new)
                        )

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = scipy.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = torch.tensor(
                            np.concatenate([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f'stage{i}.pos_embed')
            layers.add(f'stage{i}.cls_token')
            layers.add(f'stage{i}.di_token')
        return layers

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens, patches, class_tokens,dinv_tokens = getattr(self, f'stage{i}')(x) # first two stages produce 0 class tokens


        outputs = []
        outputs_di=[]
        if self.di_token :
            for di_token in dinv_tokens:
                outputs_di.append(torch.squeeze(self.norm(di_token)))
        if self.cls_token:
            for cls_token in class_tokens:
                outputs.append(torch.squeeze(self.norm(cls_token)))

        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = torch.mean(x, dim=1)
        if self.di_token :
            return outputs,outputs_di
        return outputs

    def forward(self, x):
        if self.di_token :
            outputs,outputs_di=self.forward_features(x)
        else:
            outputs = self.forward_features(x)
        x = [self.head(out) for out in outputs]
        # print(x.shape(name=None))
        if self.di_token :
            return x[-1],outputs_di[-1]
        return x[-1]

@register_model
def small_cvt(pretrained=False,add_di_tok=False, index=0, **kwargs):

    msvit_spec = {
    "INIT": 'trunc_norm',
    "NUM_STAGES": 3,
    "PATCH_SIZE": [7, 3, 3],
    "PATCH_STRIDE": [4, 2, 2],
    "PATCH_PADDING": [2, 1, 1],
    "DIM_EMBED": [64, 192, 384],
    "NUM_HEADS": [1, 3, 6],
    "DEPTH": [1, 4, 16],
    "MLP_RATIO": [4.0, 4.0, 4.0],
    "ATTN_DROP_RATE": [0.0, 0.0, 0.0],
    "DROP_RATE": [0.0, 0.0, 0.0],
    "DROP_PATH_RATE": [0.0, 0.0, 0.1],
    "QKV_BIAS": [True, True, True],
    "CLS_TOKEN": [False, False, True],
    "DI_TOKEN":[False, False, add_di_tok],
    "POS_EMBED": [False, False, False],
    "QKV_PROJ_METHOD": ['dw_bn', 'dw_bn', 'dw_bn'],
    "KERNEL_QKV": [3, 3, 3],
    "PADDING_KV": [1, 1, 1],
    "STRIDE_KV": [2, 2, 2],
    "PADDING_Q": [1, 1, 1],
    "STRIDE_Q": [1, 1, 1]
    }
    model = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=1000,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
        spec=msvit_spec
    )
    model.default_cfg = _cfg()
    if pretrained:
        model.load_state_dict(torch.load("domainbed/pretrained/cvt/CvT-21-224x224-IN-1k.pth", map_location="cpu"), strict=False)
    return model



@register_model
def tiny_cvt(pretrained=False,add_di_tok=False, index=0, **kwargs):
    msvit_spec = {
    "INIT": 'trunc_norm',
    "NUM_STAGES": 3,
    "PATCH_SIZE": [7, 3, 3],
    "PATCH_STRIDE": [4, 2, 2],
    "PATCH_PADDING": [2, 1, 1],
    "DIM_EMBED": [64, 192, 384],
    "NUM_HEADS": [1, 3, 6],
    "DEPTH": [1, 2, 10],
    "MLP_RATIO": [4.0, 4.0, 4.0],
    "ATTN_DROP_RATE": [0.0, 0.0, 0.0],
    "DROP_RATE": [0.0, 0.0, 0.0],
    "DROP_PATH_RATE": [0.0, 0.0, 0.1],
    "QKV_BIAS": [True, True, True],
    "CLS_TOKEN": [False, False, True],
    "DI_TOKEN":[False, False, add_di_tok],
    "POS_EMBED": [False, False, False],
    "QKV_PROJ_METHOD": ['dw_bn', 'dw_bn', 'dw_bn'],
    "KERNEL_QKV": [3, 3, 3],
    "PADDING_KV": [1, 1, 1],
    "STRIDE_KV": [2, 2, 2],
    "PADDING_Q": [1, 1, 1],
    "STRIDE_Q": [1, 1, 1]
    }
    model = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=1000,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
        spec=msvit_spec
    )
    model.default_cfg = _cfg()
    if pretrained:
        model.load_state_dict(torch.load("domainbed/pretrained/cvt/CvT-13-224x224-IN-1k.pth", map_location="cpu"), strict=False)
    return model



if __name__ == "__main__":
    model = small_cvt()
    model.load_state_dict(torch.load("../pretrained_models/CvT-13-224x224-IN-1k.pth", map_location="cpu"), strict=False)
    input = torch.randn(1, 3, 224, 224)
    # print(model(input)[-1].shape)
    print(f'num params: {sum(p.numel() for p in model.parameters())/1000000}')

# for tiny there are 10
# tiny_cvt's Top-1 Accuracy: 0.8144
# tiny_cvt's Top-1 Accuracy: 0.8144
# tiny_cvt's Top-1 Accuracy: 0.81142
# tiny_cvt's Top-1 Accuracy: 0.79272
# tiny_cvt's Top-1 Accuracy: 0.6659
# tiny_cvt's Top-1 Accuracy: 0.1121
# tiny_cvt's Top-1 Accuracy: 0.0206
# tiny_cvt's Top-1 Accuracy: 0.01224
# tiny_cvt's Top-1 Accuracy: 0.00424
# tiny_cvt's Top-1 Accuracy: 0.00154

# for small there are 16

