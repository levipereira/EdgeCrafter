"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from ...core import register
from .._misc import (BoundingBoxes, Image, Mask, SanitizeBoundingBoxes, Video,
                     _boxes_keys, convert_to_tv_tensor)

torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class PadToMultiple(nn.Module):
    """Pad image so tiles fully cover the frame.

    Pads right and bottom only to the nearest size where
    (H - tile_size) % stride == 0 and H >= tile_size. When tile_size is a
    multiple of stride (e.g. 448 = 2*224), this is equivalent to padding H
    to the nearest multiple of stride that is >= tile_size.

    Bbox coordinates are NOT shifted because padding is right+bottom only —
    normalized [0,1] coordinates computed before padding remain valid.

    Must be placed AFTER ConvertPILImage in the transform pipeline (operates
    on tensors, not PIL images).

    Args:
        tile_size: Side length of each tile in pixels.
        stride: Step between adjacent tiles in pixels.
        fill: Fill value for padded pixels. Default 0.
    """

    def __init__(
        self, tile_size: int = 448, stride: int = 224, fill: float = 0,
    ) -> None:
        super().__init__()
        self.tile_size = tile_size
        self.stride = stride
        self.fill = fill

    def _compute_padded_size(self, h: int, w: int) -> tuple[int, int]:
        """Compute the padded dimensions for tile compatibility."""
        t, s = self.tile_size, self.stride
        h_pad = t + math.ceil((h - t) / s) * s if h > t else t
        w_pad = t + math.ceil((w - t) / s) * s if w > t else t
        return h_pad, w_pad

    def forward(self, *inputs):
        """Pad the first input (image) and pass through all other inputs.

        Handles both call conventions used by the Compose container:
        - ``transform(img, target)`` → inputs = (img, target)
        - ``transform((img, target))`` → inputs = ((img, target),)

        Args:
            *inputs: (image, targets_dict) or just (image,). Image must be a
                tensor of shape [C, H, W].

        Returns:
            Same structure as input, with image padded right+bottom.
        """
        # Compose passes a single tuple; unpack it
        if len(inputs) == 1 and isinstance(inputs[0], tuple):
            inputs = inputs[0]

        img = inputs[0]
        h_cur, w_cur = img.shape[-2], img.shape[-1]
        h_pad, w_pad = self._compute_padded_size(h_cur, w_cur)
        pad_bottom = h_pad - h_cur
        pad_right = w_pad - w_cur

        padding = [0, 0, pad_right, pad_bottom]

        if pad_bottom > 0 or pad_right > 0:
            img = F.pad(img, padding=padding, fill=self.fill, padding_mode='constant')

        if len(inputs) == 1:
            return img

        outputs = (img,) + inputs[1:]
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
