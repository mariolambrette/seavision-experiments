"""The interface every backbone adapter implements.

Designed against the measured token layouts of all seven backbones
(`extract/probe_backbones.py`, 2 October 2026), not against one model:

  * layers differ in number (18 to 32) and width (896 to 1280);
  * the number of non-patch tokens differs (0 for SigLIP 2, 1 for the CLIP
    family, 5 for the DINO family: CLS + 4 registers);
  * SigLIP 2 has no CLS token at all, so `has_cls` is part of the contract
    rather than an assumption;
  * pooled outputs differ (projection head, attention pooling, none).

An adapter turns a batch of preprocessed images into a `Readout`: for every
block, the CLS token (if any), the mean of the patch tokens, and the mean of
the patch tokens after the model's own final norm; plus whatever pooled
outputs the model has. Registers and padding are never averaged in.

Everything that is NOT backbone-specific -- geometry, resizing, reading
shards, writing outputs, manifests -- lives outside the adapters, so all
seven are extracted by the same code. That matters because the readout
chosen on the development set is only valid if the main extraction (WP9)
computes exactly the same thing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Readout:
    """One batch. Per-layer arrays are [B, n_layers, width]."""
    cls: torch.Tensor | None            # None when the model has no CLS
    patch_mean: torch.Tensor            # mean of raw patch tokens
    patch_mean_normed: torch.Tensor     # mean of patch tokens after final norm
    pooled: dict[str, torch.Tensor] = field(default_factory=dict)  # [B, d]


class Adapter:
    """Subclass per backbone family. Set the attributes in `load`."""

    name: str = ""
    library: str = ""
    checkpoint: str = ""
    n_layers: int = 0
    width: int = 0
    patch_size: int = 0
    prefix_tokens: int = 0           # non-patch tokens before the grid
    has_cls: bool = True
    # Where the CLS sits among the prefix tokens. Not always 0: DINOv3 is
    # [CLS, registers, patches] but birder (Bio-DINO) is [registers, CLS,
    # patches], so the CLS is token 4 there. Never assume it.
    cls_index: int = 0
    native_size: int = 0
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # True when the requested size is not the checkpoint's native one and the
    # model only accepts it by interpolating its position embeddings. Results
    # at such sizes are labelled, never silently mixed with native ones.
    interpolated: bool = False
    # "fixed": forward(x) takes [B, 3, size, size]. "naflex": `size` is a
    # patch budget and forward(patches, mask, spatial_shapes) takes the
    # padded patch sequences built by preprocess.naflex_inputs.
    input_mode: str = "fixed"

    def load(self, device: torch.device, size: int, no_weights: bool = False):
        raise NotImplementedError

    @torch.inference_mode()
    def forward(self, *x: torch.Tensor) -> Readout:
        raise NotImplementedError

    def self_check(self, *x: torch.Tensor) -> dict:
        """Compare what the adapter reconstructs against the model's own
        forward on the same batch. Called once per run, on the first batch,
        and its result goes in the manifest."""
        return {}

    def describe(self) -> dict:
        return {k: getattr(self, k) for k in (
            "name", "library", "checkpoint", "n_layers", "width",
            "patch_size", "prefix_tokens", "has_cls", "cls_index",
            "native_size",
            "mean", "std", "interpolated", "input_mode")}
