"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)
Modified from https://huggingface.co/spaces/Hila/RobustViT/blob/main/ViT/ViT_new.py

"""
import math
import warnings
from functools import partial
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from ..core import register
from ..misc import dist_utils
from .hybrid_encoder import ConvNormLayer_fuse

__all__ = ['ViTAdapter', ]


def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class TileExtractor(nn.Module):
    """Extract overlapping tiles from a high-resolution frame using F.unfold.

    Differentiable operation — gradients flow through normally. Input frames
    must be tile-compatible: (H - tile_size) % stride == 0 and same for W.
    Use PadToMultiple transform to ensure this before calling.

    Args:
        tile_size: Side length of each square tile in pixels.
        stride: Step between adjacent tiles in pixels. Overlap = tile_size - stride.
    """

    def __init__(self, tile_size: int = 448, stride: int = 224):
        super().__init__()
        self.tile_size = tile_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Extract tiles from input frame.

        Args:
            x: Input tensor [B, C, H, W]. H and W must satisfy
               (H - tile_size) % stride == 0 and (W - tile_size) % stride == 0.

        Returns:
            Tuple of (tiles, meta) where:
                tiles: Tensor [B*N, C, tile_size, tile_size] with all tiles.
                meta: Dict with keys B, N, nh, nw, H, W, tile_size, stride.

        Raises:
            AssertionError: If frame is smaller than tile_size or dimensions
                are not tile-compatible.
        """
        B, C, H, W = x.shape
        t = self.tile_size
        s = self.stride

        assert H >= t and W >= t, (
            f"Frame {H}x{W} smaller than tile_size={t}. "
            f"Use a smaller tile_size or pad the input."
        )
        assert (H - t) % s == 0 and (W - t) % s == 0, (
            f"Frame {H}x{W} is not tile-compatible with tile_size={t}, stride={s}. "
            f"Use PadToMultiple to pad before calling TileExtractor."
        )

        patches = F.unfold(x, kernel_size=t, stride=s)
        N = patches.shape[-1]

        tiles = patches.view(B, C, t, t, N).permute(0, 4, 1, 2, 3)
        tiles = tiles.reshape(B * N, C, t, t)

        nh = (H - t) // s + 1
        nw = (W - t) // s + 1

        meta = {
            'B': B, 'N': N, 'nh': nh, 'nw': nw,
            'H': H, 'W': W,
            'tile_size': t, 'stride': s,
        }
        return tiles, meta


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        assert head_dim % 4 == 0, "Head dimension must be divisible by 4 for 2D RoPE"
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = head_dim
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(head_dim // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else: # min
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            coords += torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)[None, :]
        if self.training and self.jitter_coords is not None:
            jitter = (torch.empty(2, **dd).uniform_(-np.log(self.jitter_coords), np.log(self.jitter_coords))).exp()
            coords *= jitter[None, :]
        if self.training and self.rescale_coords is not None:
            rescale = (torch.empty(1, **dd).uniform_(-np.log(self.rescale_coords), np.log(self.rescale_coords))).exp()
            coords *= rescale

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).repeat(1, 2)

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        if self.base is not None:
            periods = self.base ** (2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2))
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)
            periods = self.max_period * (base ** (exponents - 1))
        self.periods.data.copy_(periods)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, sin, cos):
    """Applies RoPE to the input tensor."""
    return (x * cos) + (rotate_half(x) * sin)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.act(self.fc1(x)) 
        x = self.drop(x) 
        x = self.fc2(x) 
        x = self.drop(x)
        return x

    
class ConvPyramidPatchEmbed(nn.Module):
    def __init__(self, embed_dim=192, patch_size=16, act='relu'):
        super().__init__()
        
        assert patch_size==16, "Only support patch_size=16 for ConvPyramidPatchEmbed"
        
        num_stages = int(math.log2(patch_size)) - 1
        ratios = [2 ** i for i in range(num_stages, 0, -1)]
        channels = [embed_dim // r for r in ratios]
        
        self.convs = nn.ModuleList([
            ConvNormLayer_fuse(in_ch, out_ch, 3, 2, act=act)
            for in_ch, out_ch in zip([3] + channels[:-1], channels)
        ])
        
        self.proj = nn.Conv2d(channels[-1], embed_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std); u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1); tensor.erfinv_(); tensor.mul_(std * math.sqrt(2.)); tensor.add_(mean); tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope_sincos=None):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads) # .permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if rope_sincos is not None:
            sin, cos = rope_sincos
            q_cls, q_patch = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patch = k[:, :, :1, :], k[:, :, 1:, :]

            q_patch = apply_rope(q_patch, sin, cos)
            k_patch = apply_rope(k_patch, sin, cos)

            q = torch.cat((q_cls, q_patch), dim=2)
            k = torch.cat((k_cls, k_patch), dim=2)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
        x = x.transpose(1, 2).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.SiLU, norm_layer=nn.LayerNorm, ffn_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(in_features=dim, hidden_features=int(dim * ffn_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, rope_sincos=None):
        attn_output = self.attn(self.norm1(x), rope_sincos=rope_sincos)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=192, depth=12,
        num_heads=3, ffn_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0., return_layers=[3, 7, 11], embed_layer=ConvPyramidPatchEmbed,
        norm_layer=None, act_layer=None, ffn_layer=Mlp
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.return_layers = return_layers
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        if embed_layer == PatchEmbed:
            self.patch_embed = embed_layer(
                img_size=img_size, patch_size=patch_size,
                in_chans=in_chans, embed_dim=embed_dim
            )
        else:
            self.patch_embed = embed_layer(embed_dim=embed_dim, patch_size=patch_size)
        self.patch_size = patch_size

        self.register_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer, ffn_layer=ffn_layer,
            ) for i in range(depth)
        ])

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim, num_heads=num_heads, base=100.0,
            normalize_coords="separate", shift_coords=None, jitter_coords=None,
            rescale_coords=None, dtype=None, device=None,
        )
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_vit_weights)
        self.rope_embed._init_weights()
        trunc_normal_(self.register_token, std=.02)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(
        self, x: torch.Tensor, register_state: torch.Tensor | None = None
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run ViT backbone on input tensor.

        Args:
            x: Input image tensor of shape [B, 3, H, W].
            register_state: Optional accumulated register state of shape
                [B, 1, embed_dim]. When None, uses the learned register_token
                parameter (original behavior).

        Returns:
            Tuple of (outs, final_register) where:
                outs: List of feature tensors from return_layers, each [B, H*W, D].
                final_register: Updated register token [B, 1, D] after all blocks.
        """
        outs = []
        x_embed = self.patch_embed(x)
        _, _, H, W = x_embed.shape

        x_embed = x_embed.flatten(2).transpose(1, 2)

        if register_state is not None:
            register_token = register_state
        else:
            register_token = self.register_token.expand(x_embed.shape[0], -1, -1)

        x = torch.cat((register_token, x_embed), dim=1)
        rope_sincos = self.rope_embed(H=H, W=W)

        for i, blk in enumerate(self.blocks):
            x = blk(x, rope_sincos=rope_sincos)
            if i in self.return_layers:
                outs.append(x[:, 1:])

        final_register = x[:, :1, :]
        return outs, final_register
    
    

EMBED_LAYER_REGISTRY = {
    "ConvPyramidPatchEmbed": ConvPyramidPatchEmbed,
    "PatchEmbed": PatchEmbed,
}

FFN_LAYER_REGISTRY = {
    "mlp": Mlp,
   # "swigluffn": SwiGLUFFN,  # To be implemented
}
    

@register()
class ViTAdapter(nn.Module):
    
    ecvit_url = {
        # detection backbone
        "ecvitt": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvitt.pth",
        "ecvittplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvittplus.pth",
        "ecvits": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvits.pth",
        "ecvitsplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvitsplus.pth",
        
        # segmentation backbone
        "ecseg_vitt": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_vitt.pth",
        "ecseg_vittplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_vittplus.pth",
        "ecseg_vits": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_vits.pth",
        "ecseg_vitsplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_vitsplus.pth",}
    
    def __init__(
        self,
        name,
        weights_path=None,
        interaction_indexes=[10, 11],
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        proj_dim=None,
        num_levels=3,
        embed_layer='ConvPyramidPatchEmbed',
        ffn_layer='mlp',
        ffn_ratio=4,
        skip_load_backbone=False,
        **kwargs
    ):
        super().__init__()
        
        self.name = name
        
        if embed_layer not in EMBED_LAYER_REGISTRY:
            raise ValueError(f"Unknown embed_layer: {embed_layer}. Available: {list(EMBED_LAYER_REGISTRY)}")
        if ffn_layer not in FFN_LAYER_REGISTRY:
            raise ValueError(f"Unknown ffn_layer: {ffn_layer}. Available: {list(FFN_LAYER_REGISTRY)}")
        embed_layer = EMBED_LAYER_REGISTRY[embed_layer]
        ffn_layer = FFN_LAYER_REGISTRY[ffn_layer]
        
        
        self.backbone = VisionTransformer(embed_dim=embed_dim, 
                                          num_heads=num_heads, 
                                          return_layers=interaction_indexes, 
                                          patch_size=patch_size, 
                                          embed_layer=embed_layer,
                                          ffn_layer=ffn_layer,
                                          ffn_ratio=ffn_ratio,
                                          **kwargs)
        if not skip_load_backbone:
            self._load_weights(weights_path)
            
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.num_levels = num_levels
        
        if num_levels != 3:
            raise NotImplementedError("Only support num_levels=3 for ViTAdapter now.")

        self.proj_dim = [proj_dim] * num_levels if proj_dim is not None else [embed_dim]

        self.projector = nn.ModuleList([ConvNormLayer_fuse(embed_dim, dim, kernel_size=1, stride=1) for dim in self.proj_dim])

        # CRITICAL-3 fix: global tile position encoding for register state.
        # Projects normalized (col/nw, row/nh) grid coordinates into embed_dim
        # so the register token knows where each tile is in the frame.
        # Cost: 2*embed_dim + embed_dim = 384+192 = 576 params for ECDet-S.
        self.tile_pos_proj = nn.Linear(2, embed_dim)
        
    def _load_weights(self, weights_path):
        if self.name not in self.ecvit_url:
            raise ValueError(f"Unknown model name: {self.name}. Available: {list(self.ecvit_url)}")
        url = self.ecvit_url[self.name]
        
        if weights_path is None:
            print(
                "="*80 + "\n",
                "❌❌❌ WARNING: Pretrained ViT weights not loaded! ❌❌❌\n"
                "The model is running with randomly initialized parameters.\n"
                "This will severely degrade performance and convergence!\n"
                f"Please download the model manually from {url}.\n"
                "If you want to train from scratch, please set `skip_load_backbone=True` to skip this warning.\n"*2,
                "="*80,
                sep="")
            
            return
        
        path = Path(weights_path)
        if path.exists():
            state = torch.load(path, weights_only=True, map_location="cpu")
            self.backbone.load_state_dict(state, strict=True)
            print(
                "=" * 80 + "\n",
                "✅ Pretrained ViT weights loaded successfully!\n"
                f"📦 Weights file: {path}\n",
                "=" * 80
            )
        else:
            model_dir = Path(__file__).resolve().parents[2] / "ecvits"
            print(
                f"\nTrying to load pretrained ViT weights from {url}. "
                "If this fails, please download the model manually.\n"
                "If you have already downloaded the model, please check that `weights_path` is correct and the file exists.")
            
            if safe_get_rank() == 0:
                torch.hub.load_state_dict_from_url(
                    url, map_location="cpu", model_dir=model_dir, weights_only=True,)
                
            if dist_utils.is_dist_available_and_initialized():
                torch.distributed.barrier()

            model_path = model_dir / url.split("/")[-1]
            state = torch.load(model_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=True)
            print(
                "=" * 80 + "\n",
                "✅ Pretrained ViT weights loaded successfully!\n"
                f"📦 Weights downloaded to: {model_dir}\n",
                "=" * 80,
                sep="")


    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Dispatch to _forward_single or _forward_tiled based on input size.

        Routes to _forward_tiled only when the input is large enough to require
        multiple tiles AND is tile-compatible (padded by PadToMultiple). Falls
        back to _forward_single for standard-size inputs or when tiling is not
        configured.

        Args:
            x: Input image tensor [B, 3, H, W].

        Returns:
            List of 3 projected feature maps at strides 8, 16, 32.
        """
        tile_size = getattr(self, 'tile_size', None)
        tile_stride = getattr(self, 'tile_stride', None)
        if tile_size is not None and tile_stride is not None:
            h, w = x.shape[2], x.shape[3]
            needs_tiling = h > tile_size or w > tile_size
            is_compatible = (
                h >= tile_size and w >= tile_size
                and (h - tile_size) % tile_stride == 0
                and (w - tile_size) % tile_stride == 0
            )
            if needs_tiling and is_compatible:
                return self._forward_tiled(x)
        return self._forward_single(x)

    def _forward_single(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Original forward path. Fully backward-compatible with pretrained weights.

        Args:
            x: Input image tensor [B, 3, H, W].

        Returns:
            List of 3 projected feature maps at strides 8, 16, 32.
        """
        H_c, W_c = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs = x.shape[0]

        return_layers, _ = self.backbone(x)

        fused_feats = torch.mean(torch.stack(return_layers), dim=0)

        fused_feats = fused_feats.transpose(1, 2).contiguous().view(bs, -1, H_c, W_c)
        proj_feats = []
        for i in range(self.num_levels):
            scale = 2 ** (1 - i)
            resize_H = int(H_c * scale)
            resize_W = int(W_c * scale)
            feature = F.interpolate(fused_feats, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            proj_feats.append(feature)

        if len(self.projector) == 1:
            proj_feats[-1] = self.projector[-1](proj_feats[-1])
        else:
            proj_feats = [layer(feat) for layer, feat in zip(self.projector, proj_feats)]

        return proj_feats

    def _forward_tiled(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Process a high-resolution frame through sequential tile passes.

        Extracts overlapping tiles, passes each through the shared ViT backbone,
        then reassembles a global feature map by averaging overlapping tile features.
        Currently stateless (no register accumulation) — each tile uses the default
        register_token parameter independently.

        Args:
            x: Input frame tensor [B, 3, H, W] where H or W > tile_size.
               Must be tile-compatible: (H - tile_size) % tile_stride == 0.

        Returns:
            List of 3 projected feature maps at strides 8, 16, 32 relative to
            the input frame dimensions.

        Note:
            BatchNorm layers in patch_embed are forced to eval() mode during this
            call to preserve pretrained running statistics (CRITICAL-2).
        """
        tile_size = self.tile_size
        tile_stride = self.tile_stride
        embed_dim = self.backbone.embed_dim

        # CRITICAL-2: freeze BN in patch_embed to preserve pretrained stats
        patch_embed_bn_training = {}
        for name, m in self.backbone.patch_embed.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                patch_embed_bn_training[name] = m.training
                m.eval()

        try:
            extractor = TileExtractor(tile_size=tile_size, stride=tile_stride)
            tiles, meta = extractor(x)

            B = meta['B']
            N = meta['N']
            nh = meta['nh']
            nw = meta['nw']
            H = meta['H']
            W = meta['W']

            tile_feat_h = tile_size // self.patch_size
            tile_feat_w = tile_size // self.patch_size
            global_feat_h = H // self.patch_size
            global_feat_w = W // self.patch_size
            stride_feat = tile_stride // self.patch_size

            device = x.device
            dtype = x.dtype
            global_feat = torch.zeros(
                B, embed_dim, global_feat_h, global_feat_w,
                device=device, dtype=dtype,
            )
            global_count = torch.zeros(
                B, 1, global_feat_h, global_feat_w,
                device=device, dtype=dtype,
            )

            register_state = self.backbone.register_token.expand(B, -1, -1).clone()

            for tile_idx in range(N):
                tile_batch = tiles[tile_idx::N]

                row = tile_idx // nw
                col = tile_idx % nw

                # CRITICAL-3 fix: inject global tile position into register state
                tile_pos = torch.tensor(
                    [[col / nw, row / nh]],
                    device=device, dtype=dtype,
                )
                pos_signal = self.tile_pos_proj(tile_pos).unsqueeze(0).expand(B, -1, -1)
                tile_register = register_state + pos_signal

                return_layers, new_register = self.backbone(
                    tile_batch, register_state=tile_register,
                )

                # Truncated BPTT: detach between tiles to prevent gradient explosion
                register_state = new_register.detach()

                fused_tile = torch.mean(torch.stack(return_layers), dim=0)
                fused_tile = fused_tile.transpose(1, 2).contiguous()
                fused_tile = fused_tile.view(B, embed_dim, tile_feat_h, tile_feat_w)

                y0 = row * stride_feat
                x0 = col * stride_feat
                y1 = y0 + tile_feat_h
                x1 = x0 + tile_feat_w

                global_feat[:, :, y0:y1, x0:x1] += fused_tile
                global_count[:, :, y0:y1, x0:x1] += 1.0

            global_count = global_count.clamp(min=1.0)
            global_feat = global_feat / global_count

            proj_feats = []
            for i in range(self.num_levels):
                scale = 2 ** (1 - i)
                resize_H = int(global_feat_h * scale)
                resize_W = int(global_feat_w * scale)
                feature = F.interpolate(
                    global_feat, size=[resize_H, resize_W],
                    mode="bilinear", align_corners=False,
                )
                proj_feats.append(feature)

            if len(self.projector) == 1:
                proj_feats[-1] = self.projector[-1](proj_feats[-1])
            else:
                proj_feats = [layer(feat) for layer, feat in zip(self.projector, proj_feats)]

            return proj_feats

        finally:
            # Restore BN training state
            for name, m in self.backbone.patch_embed.named_modules():
                if isinstance(m, nn.BatchNorm2d) and name in patch_embed_bn_training:
                    m.train(patch_embed_bn_training[name])
        
    
