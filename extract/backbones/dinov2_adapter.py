"""DINOv2 with registers (Hugging Face transformers), ViT-L/14. Added after
WP8 closed (schedule WP8b, plan section 5.3).

Token layout: [CLS, 4 registers, patches], batch-first -- the same as DINOv3,
so 5 prefix tokens and registers are never averaged into the patch mean.
Checked on the real checkpoint by the step-0 probe before the sweep.

Position embeddings are LEARNED for a 37 x 37 grid (518 px at patch 14), so
every sweep size (224, 336, 448 px = 16, 24, 32 patches a side) reaches the
model only by interpolating them, and `interpolated` is True for all three.
The interpolation is the library's own (bicubic, antialiased, in float32),
done inside the model's forward; it is not something this adapter controls,
so the transformers version is recorded in every manifest and must not
change between the WP8b sweep and the WP9 extraction.

How the model pools (transformers Dinov2WithRegistersModel.forward): the
final layernorm is applied to every token of the last block, and
`pooler_output` is the CLS of that. So "pooler_output" here is
layernorm(last block)[:, 0], reconstructed from the block outputs and checked
against the model's own forward on the first batch. No projection head.
"""
from __future__ import annotations

import torch

from .base import Adapter, Readout

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2Adapter(Adapter):
    library = "transformers"
    patch_size = 14

    def __init__(self, name, checkpoint, no_weights_cfg=None):
        self.name, self.checkpoint = name, checkpoint
        self.no_weights_cfg = no_weights_cfg or {}

    def load(self, device, size, no_weights=False):
        from transformers import (AutoModel, Dinov2WithRegistersConfig,
                                  Dinov2WithRegistersModel)
        if size % self.patch_size:
            raise ValueError(f"{self.name}: size {size} is not a multiple of "
                             f"the patch size {self.patch_size}")
        if no_weights:
            cfg = Dinov2WithRegistersConfig(patch_size=14, image_size=518,
                                            num_register_tokens=4,
                                            **self.no_weights_cfg)
            model = Dinov2WithRegistersModel(cfg)
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            from transformers import AutoImageProcessor
            model = AutoModel.from_pretrained(self.checkpoint)
            # Colour statistics only; the processor's resize/crop is not used.
            proc = AutoImageProcessor.from_pretrained(self.checkpoint)
            mean = tuple(proc.image_mean)
            std = tuple(proc.image_std)
        if type(model).__name__ != "Dinov2WithRegistersModel":
            raise RuntimeError(f"{self.name}: loaded {type(model).__name__}, "
                               "expected Dinov2WithRegistersModel")
        self.model = model.eval().to(device)
        c = self.model.config
        if c.patch_size != self.patch_size:
            raise RuntimeError(f"{self.name}: patch size {c.patch_size}")
        self.blocks = list(self.model.encoder.layer)
        self.n_layers = len(self.blocks)
        self.width = int(c.hidden_size)
        self.n_registers = int(c.num_register_tokens)
        self.prefix_tokens = 1 + self.n_registers        # CLS + registers
        self.has_cls = True
        self.cls_index = 0
        self.native_size = int(c.image_size)             # 518
        self.interpolated = size != self.native_size
        self.mean, self.std = mean, std
        self._out = [None] * self.n_layers
        for i, b in enumerate(self.blocks):
            b.register_forward_hook(self._hook(i))
        return self

    def _hook(self, i):
        # Reduce inside the hook: no layer's full token grid is kept.
        def h(_m, _inp, o):
            t = (o[0] if isinstance(o, (tuple, list)) else o).float()
            patches = t[:, self.prefix_tokens:]
            self._out[i] = (t[:, 0], patches.mean(1),
                            self.model.layernorm(patches).mean(1))
        return h

    @torch.inference_mode()
    def forward(self, x):
        out = self.model(pixel_values=x)                 # fills self._out
        expect = (x.shape[-2] // 14) * (x.shape[-1] // 14) + self.prefix_tokens
        if out.last_hidden_state.shape[1] != expect:
            raise RuntimeError(f"{self.name}: {out.last_hidden_state.shape[1]} "
                               f"tokens, expected {expect}")
        cls, pm, pmn = (list(z) for z in zip(*self._out))
        self._out = [None] * self.n_layers
        cls = torch.stack(cls, 1)
        pooled = {"pooler_output": self.model.layernorm(cls[:, -1])}
        self._last_pooler = out.pooler_output.float()
        return Readout(cls=cls, patch_mean=torch.stack(pm, 1),
                       patch_mean_normed=torch.stack(pmn, 1), pooled=pooled)

    def self_check(self, x):
        r = self.forward(x)
        mine, theirs = r.pooled["pooler_output"], self._last_pooler
        err = float((mine - theirs).abs().max() /
                    theirs.abs().max().clamp_min(1e-6))
        return {"reconstructed_vs_pooler_output_max_rel_err": err,
                "ok": err < 1e-2}
