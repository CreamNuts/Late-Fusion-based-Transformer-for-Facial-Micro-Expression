from functools import partial
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum

from .create_model import register_model
from .transformer_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    DropPath,
    I3DHead,
    Mlp,
    to_2tuple,
    trunc_normal_,
)


def _cfg(**kwargs):
    # TF-B
    return {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "attention_type": "divided_space_time",
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    }


default_cfgs = {
    "ch1_timesformer": _cfg(in_chans=1),
    "ch3_timesformer": _cfg(in_chans=3),
    "ch4_timesformer": _cfg(in_chans=4),
    "ch6_timesformer": _cfg(in_chans=6),
}


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_weight = None

    def forward(self, x):
        B, N, C = x.shape
        h = self.num_heads
        d_h = C // h
        if self.with_qkv:
            qkv = rearrange(self.qkv(x), "b n (qkv h d_h) -> qkv b h n d_h", qkv=3, h=h, d_h=d_h)
            # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
            #                           C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = rearrange(x, "b n (h d_h) -> b h n d_h", h=h, d_h=d_h)
            # qkv = x.reshape(B, N, self.num_heads, C //
            #                 self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv
        # attn = (q @ rearrange(k, 'b h n d -> b h d n')) * self.scale
        attn = einsum("b h n d, b h j d -> b h n j", q, k) * self.scale
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_weight = attn
        attn = self.attn_drop(attn)

        x = einsum("b h n j, b h j d -> b h n d", attn, v)
        x = rearrange(x, "b h n d_h -> b n (h d_h)")
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attention_type="divided_space_time",
    ):
        super().__init__()
        self.attention_type = attention_type
        assert attention_type in ["divided_space_time", "space_only", "joint_space_time"]

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Temporal Attention Parameters
        if self.attention_type == "divided_space_time":
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.temporal_fc = nn.Linear(dim, dim)

        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ["space_only", "joint_space_time"]:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == "divided_space_time":
            # Temporal
            xt = x[:, 1:, :]
            xt = rearrange(xt, "b (h w t) m -> (b h w) t m", b=B, h=H, w=W, t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(
                res_temporal, "(b h w) t m -> b (h w t) m", b=B, h=H, w=W, t=T
            )
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:, 1:, :] + res_temporal

            # Spatial
            init_cls_token = x[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, "b t m -> (b t) 1 m", b=B, t=T)  # .unsqueeze(1)
            xs = xt
            xs = rearrange(xs, "b (h w t) m -> (b t) (h w) m", b=B, h=H, w=W, t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            # Taking care of CLS token
            cls_token = res_spatial[:, 0, :]
            cls_token = rearrange(cls_token, "(b t) m -> b t m", b=B, t=T)
            # averaging for every frame
            cls_token = torch.mean(cls_token, 1, True)
            res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(
                res_spatial, "(b t) (h w) m -> b (h w t) m", b=B, h=H, w=W, t=T
            )
            res = res_spatial
            x = xt

            # Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.proj(x)
        W = x.size(-1)
        x = rearrange(x, "(b t) c h w -> (b t) (h w) c", b=B)
        # x = x.flatten(2).transpose(1, 2)
        return x, T, W


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        num_frames=8,
        attention_type="divided_space_time",
        dropout=0.0,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != "space_only":
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        # Attention Blocks
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attention_type=self.attention_type,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = I3DHead(in_channels=embed_dim, num_classes=num_classes, spatial_type="")

        # Orig TF
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # initialization of temporal attention weights
        if self.attention_type == "divided_space_time":
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if "Block" in m_str:
                    if i > 0:
                        nn.init.constant_(m.temporal_fc.weight, 0)
                        nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "time_embed"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = I3DHead(
                in_channels=self.embed_dim, num_classes=num_classes, spatial_type=global_pool
            )
            self.head.init_weights()
        else:
            self.head = nn.Identity()

        # self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)  # x : (b t) (h w) c
        cls_tokens = repeat(self.cls_token, "1 1 embed -> bt 1 embed", bt=B * T)
        # cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed  # 1 num_patch+1 embeddim
            cls_pos_embed = rearrange(pos_embed[0, 0, :], "embed -> 1 1 embed")
            # cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = rearrange(
                pos_embed[0, 1:, :], "(np np) embed -> 1 embed np np", np=14
            )
            # other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            H = x.size(1) // W
            # other_pos_embed = other_pos_embed.reshape(1, x.size(2), 14, 14)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode="nearest")
            new_pos_embed = rearrange(new_pos_embed, "1 embed h w -> 1 (h w) embed")
            # new_pos_embed = new_pos_embed.flatten(2)
            # new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # Time Embeddings
        if self.attention_type != "space_only":
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=T)
            x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, "(b n) t m -> b (n t) m", b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        # Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        # Predictions for space-only baseline
        if self.attention_type == "space_only":
            x = rearrange(x, "(b t) n m -> b t n m", b=B, t=T)
            x = torch.mean(x, 1)  # averaging predictions for every frame

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class TimeSFormer(nn.Module):
    def __init__(
        # self, cfg: Dict, in_chans: int, image_size: tuple, num_classes: int, num_frames: int
        self,
        in_chans: int,
        image_size: tuple,
        num_classes: int,
        num_frames: int,
        **kwargs
    ):
        super(TimeSFormer, self).__init__()
        self.model = VisionTransformer(
            in_chans=in_chans,
            image_size=image_size,  # cfg.DATA.TRAIN_CROP_SIZE,
            num_classes=num_classes,  # cfg.MODEL.NUM_CLASSES,
            num_frames=num_frames,  # cfg.DATA.NUM_FRAMES,
            **kwargs
        )
        self.linear_in_features = self.model.embed_dim

    def forward(self, x):
        x = self.model(x)
        return x

    def reset_classifier(self, num_classes):
        self.model.reset_classifier(num_classes, global_pool="")


@register_model
def ch1_timesformer(**kwargs):
    return TimeSFormer(**default_cfgs["ch1_timesformer"], **kwargs)


@register_model
def ch3_timesformer(**kwargs):
    return TimeSFormer(**default_cfgs["ch3_timesformer"], **kwargs)


@register_model
def ch4_timesformer(**kwargs):
    return TimeSFormer(**default_cfgs["ch4_timesformer"], **kwargs)


@register_model
def ch6_timesformer(**kwargs):
    return TimeSFormer(**default_cfgs["ch6_timesformer"], **kwargs)
