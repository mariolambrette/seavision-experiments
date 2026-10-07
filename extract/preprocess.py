"""Geometry, resize and colour normalisation -- shared by every backbone.

The order is fixed and is the lesson of the first embedding experiment:
our geometry first, then a direct resize to the model's input size, then the
model's own colour statistics. The models' bundled preprocessing is never
used: it centre-crops, which would cut letterbox padding off and quietly turn
letterbox into something close to square expansion.

Geometries (a geometry names a shard set plus what is done to its image):
  letterbox   crops set; pad the short side to a square with the model's own
              mean colour, so the padding normalises to zero and carries no
              signal
  distort     crops set; stretch to a square (keeps everything, distorts shape)
  square_m00  square-m00 set; already square, resize only
  square_m10  square-m10 set; already square, resize only
  native      crops set; the crop exactly as cut, aspect ratio kept. Only for
              NaFlex models (SigLIP 2), which take each image at its own
              aspect ratio; a fixed-size model cannot accept it.

NaFlex inputs (SigLIP 2): instead of one square size, the model takes a patch
budget. Each image is resized, keeping its aspect ratio, to the largest
multiple-of-16 size whose patch count fits the budget, cut into patches, and
padded to the budget with a mask. The sizing rule is a frozen copy of the
transformers implementation (below), and the resize is our own, the same
antialiased bicubic every backbone gets -- so the pixels cannot change with a
library upgrade between the WP8 sweep and WP9.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

SHARD_SET = {"letterbox": "crops", "distort": "crops",
             "square_m00": "square-m00", "square_m10": "square-m10",
             "native": "crops"}
RESAMPLE = Image.BICUBIC


def apply_geometry(img: Image.Image, geometry: str, mean) -> Image.Image:
    img = img.convert("RGB")
    if geometry in ("distort", "square_m00", "square_m10", "native"):
        return img
    if geometry == "letterbox":
        w, h = img.size
        s = max(w, h)
        fill = tuple(int(round(255 * m)) for m in mean)
        canvas = Image.new("RGB", (s, s), fill)
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas
    raise ValueError(f"unknown geometry {geometry!r}")


def _normalise(img, mean, std):
    a = np.asarray(img, dtype=np.float32) / 255.0
    return (a - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)


def to_tensor(img: Image.Image, size: int, mean, std) -> torch.Tensor:
    """Direct resize to size x size (antialiased bicubic), then normalise."""
    img = img.resize((size, size), RESAMPLE, reducing_gap=None)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def preprocess(img: Image.Image, geometry: str, size: int, mean, std):
    if geometry == "native":
        raise ValueError("geometry 'native' is for NaFlex models only")
    return to_tensor(apply_geometry(img, geometry, mean), size, mean, std)


# ---------------------------------------------------------------- NaFlex
def naflex_size(h: int, w: int, patch: int, budget: int,
                eps: float = 1e-5) -> tuple[int, int]:
    """Frozen copy of transformers' get_image_size_for_max_num_patches
    (transformers 5.x, models/siglip2/image_processing_siglip2.py): the
    largest aspect-preserving size, each side a multiple of `patch` and at
    least one patch, whose patch count fits `budget`."""
    import math

    def scaled(scale, size):
        s = math.ceil(size * scale / patch) * patch
        return int(max(patch, s))

    lo, hi = eps / 10, 100.0
    while hi - lo >= eps:
        mid = (lo + hi) / 2
        if (scaled(mid, h) / patch) * (scaled(mid, w) / patch) <= budget:
            lo = mid
        else:
            hi = mid
    return scaled(lo, h), scaled(lo, w)


def naflex_inputs(img: Image.Image, geometry: str, budget: int, patch: int,
                  mean, std):
    """-> (patches [budget, patch*patch*3], mask [budget], spatial [2]).
    Patch vector order matches transformers' convert_image_to_patches:
    (row-in-patch, col-in-patch, channel)."""
    img = apply_geometry(img, geometry, mean)
    w, h = img.size
    if geometry == "distort":
        # For fixed-size models the stretch happens in the final square
        # resize; NaFlex never resizes to a square, so without this
        # "distort" would silently equal "native". Size it as a square
        # instead, and the single resize below does the stretch.
        h = w = max(h, w)
    th, tw = naflex_size(h, w, patch, budget)
    img = img.resize((tw, th), RESAMPLE, reducing_gap=None)
    a = torch.from_numpy(_normalise(img, mean, std).transpose(2, 0, 1).copy())
    c, H, W = a.shape
    gh, gw = H // patch, W // patch
    p = a.reshape(c, gh, patch, gw, patch).permute(1, 3, 2, 4, 0)
    p = p.reshape(gh * gw, patch * patch * c)
    n = p.shape[0]
    patches = torch.zeros(budget, p.shape[1], dtype=p.dtype)
    patches[:n] = p
    mask = torch.zeros(budget, dtype=torch.int32)
    mask[:n] = 1
    return patches, mask, torch.tensor([gh, gw], dtype=torch.long)
