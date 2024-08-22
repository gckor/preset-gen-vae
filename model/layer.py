
"""
Defines some basic layer Classes to be integrated into bigger networks
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from typing import Optional, List
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_

from model.position_encoding import PositionEmbeddingSine, PositionalEncoding1D


class Conv2D(nn.Sequential):
    """ A basic conv layer with activation and batch-norm """
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation=[1, 1],
                 padding_mode='zeros', activation=nn.ReLU, name_prefix='', norm='batchnorm', out_shape=[]):
        """

        :param batch_norm: 'after' activation, 'before' activation, or None
        """
        super().__init__()
        self.add_module(name_prefix + ' conv', nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                                                        padding, dilation, padding_mode=padding_mode))
        self.add_module(name_prefix + ' act', activation)
        
        if norm == 'batchnorm':
            self.add_module(name_prefix + ' bn', nn.BatchNorm2d(out_ch))
        elif norm == 'layernorm':
            h, w = out_shape
            self.add_module(name_prefix + ' ln1', Rearrange('b c h w -> b (h w) c'))
            self.add_module(name_prefix + ' ln2', nn.LayerNorm(out_ch))
            self.add_module(name_prefix + ' ln3', Rearrange('b (h w) c -> b c h w', h=h, w=w))


class TConv2D(nn.Sequential):
    """ A basic Transposed conv layer with activation and batch-norm """
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, output_padding=0, dilation=1,
                 padding_mode='zeros', activation=nn.ReLU, name_prefix='', batch_norm='after'):
        """

        :param batch_norm: 'after' activation, 'before' activation, or None
        """
        super().__init__()
        self.add_module(name_prefix + 'tconv', nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride,
                                                                  padding, output_padding,
                                                                  dilation=dilation, padding_mode=padding_mode))
        bn = nn.BatchNorm2d(out_ch)
        if batch_norm == 'before':
            self.add_module(name_prefix + 'bn', bn)
        self.add_module(name_prefix + 'act', activation)
        if batch_norm == 'after':
            self.add_module(name_prefix + 'bn', bn)


class TransformerEncoder(nn.Module):
    """
    Vanilla transformer encoder
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalize_before: bool = True,
        **kwargs,
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before
        )
        self.layers = _get_clones(encoder_layer, num_encoder_layers)
        self.norm = nn.LayerNorm(d_model) if normalize_before else None
        self._reset_parameters()
        self.pos_embed = PositionEmbeddingSine(d_model // 2)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        pos_embed = self.pos_embed(src).flatten(2).permute(2, 0, 1)
        output = src.flatten(2).permute(2, 0, 1)

        for layer in self.layers:
            output = layer(output, mask, src_key_padding_mask, pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output, pos_embed


class TransformerDecoder(nn.Module):
    """
    Transformer decoder with learnable queries
    """
    def __init__(
        self,
        num_queries: int,
        d_model: int,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalize_before: bool = True,
        **kwargs,
    ):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before
        )
        self.layers = _get_clones(decoder_layer, num_decoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.tgt = nn.Parameter(torch.zeros(num_queries, 1, d_model), requires_grad=True)
        self._reset_parameters()
        self.query_pos_embed = PositionalEncoding1D(d_model, num_queries)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        torch.nn.init.normal_(self.tgt, std=.02)

    def forward(
        self,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_pos_embed: Optional[Tensor] = None,
    ):
        query_pos_embed = self.query_pos_embed(memory)
        output = self.tgt.expand(-1, query_pos_embed.shape[1], -1)

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=memory_pos_embed,
                query_pos=query_pos_embed
            )

        if self.norm is not None:
            output = self.norm(output)

        return output.transpose(0, 1)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self, 
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt2, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution[0], self.resolution[1]
        elif idx == 0:
            H_sp, W_sp = self.resolution[0], self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution[1], self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H, W = self.resolution
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H, W = self.resolution
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H, W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x
    

class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, input_shape, padding, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, padding)
        self.norm = norm_layer(dim_out)
        self.input_shape = input_shape

    def forward(self, x):
        B, HW, C = x.shape
        H, W = self.input_shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        
        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx = -1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim//2, resolution=self.patches_resolution, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:,:,:,:C//2])
            x2 = self.attns[1](qkv[:,:,:,C//2:])
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(
        self,
        d_model: int,
        spectrogram_size: List[int],
        num_heads: List[int] = [2, 4, 8, 16],
        num_encoder_layers: List[int] = [1, 2, 2, 1],
        split_size: List[int] = [1, 2, 8, 8],
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        dropout: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        stage1_shape = self._out_shape(spectrogram_size, [7, 7], [4, 6], [2, 2])
        stage2_shape = self._out_shape(stage1_shape, [3, 3], [2, 2], [1, 4])
        stage3_shape = self._out_shape(stage2_shape, [3, 3], [2, 2], [1, 1])
        stage4_shape = self._out_shape(stage3_shape, [3, 3], [2, 2], [1, 1])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(num_encoder_layers))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=d_model // 8, num_heads=num_heads[0], reso=stage1_shape, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=dropout, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(num_encoder_layers[0])])

        self.merge1 = Merge_Block(d_model // 8, d_model // 4, stage1_shape, [1, 4])
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=d_model // 4, num_heads=num_heads[1], reso=stage2_shape, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=dropout, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(num_encoder_layers[:1])+i], norm_layer=norm_layer)
            for i in range(num_encoder_layers[1])])
        
        self.merge2 = Merge_Block(d_model // 4, d_model // 2, stage2_shape, [1, 1])
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=d_model // 2, num_heads=num_heads[2], reso=stage3_shape, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=dropout, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(num_encoder_layers[:2])+i], norm_layer=norm_layer)
            for i in range(num_encoder_layers[2])])

        self.stage3 = nn.ModuleList(temp_stage3)
        
        self.merge3 = Merge_Block(d_model // 2, d_model, stage3_shape, [1, 1])
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=d_model, num_heads=num_heads[3], reso=stage4_shape, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=dropout, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(num_encoder_layers[:-1])+i], norm_layer=norm_layer, last_stage=True)
            for i in range(num_encoder_layers[-1])])
       
        self.norm = norm_layer(d_model)
        self.apply(self._init_weights)
        self.out_H, self.out_W = stage4_shape
        self.out_pos_embed = PositionEmbeddingSine(d_model // 2)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def forward(self, x):
        x = x.flatten(2).permute(0, 2, 1)

        for blk in self.stage1:
            x = blk(x)

        for pre, blocks in zip([self.merge1, self.merge2, self.merge3], 
                               [self.stage2, self.stage3, self.stage4]):
            x = pre(x)
            for blk in blocks:
                x = blk(x)

        x = self.norm(x)
        out_pos = x.reshape(x.shape[0], x.shape[-1], self.out_H, self.out_W)
        out_pos_embed = self.out_pos_embed(out_pos).flatten(2).permute(2, 0, 1)
        return x.transpose(0, 1), out_pos_embed
    
    def _out_shape(
            self,
            spectrogram_size: List[int],
            kernel_size: List[int],
            stride: List[int],
            padding: List[int],
            dilation: List[int] = [1, 1],
        ):
        H_in, W_in = spectrogram_size
        H_out = (H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        W_out = (W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
        return [H_out, W_out]


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img