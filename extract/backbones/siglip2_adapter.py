"""SigLIP 2 So400m NaFlex (Hugging Face transformers).

What makes it different from the other six (probe, 2 October 2026, and the
transformers source):

  * no CLS token: every token is a patch, plus padding up to the budget, so
    `has_cls` is False and the readout has no CLS axis;
  * NaFlex: each image keeps its own aspect ratio and gets a variable number
    of patches within a budget, padded and masked. "Size" for this model is
    that patch budget, not a square side. Inputs are built by our own
    `preprocess.naflex_inputs`, not the transformers processor, so the pixels
    do not depend on the library version;
  * the 16 x 16 position-embedding grid is resized to every image's patch
    grid by design, so no size is "interpolated" in the sense the CLIP family
    is above 224 px;
  * pooling is an attention-pooling head (`head`) applied after
    `post_layernorm` at the final layer only. It is stored as the model's own
    `pooler_output`; intermediate layers have only patch means.

Every patch mean is masked: padding tokens are never averaged in.

Self-check: the adapter's final-layer masked mean of normed patches is
compared with the masked mean of the model's own `last_hidden_state` (which
is post_layernorm of the encoder output). That checks the hook indexing, the
mask and the norm together.
"""
from __future__ import annotations

import torch

from .base import Adapter, Readout


class Siglip2Adapter(Adapter):
    library = "transformers"
    input_mode = "naflex"
    interpolated = False

    def __init__(self, name, checkpoint, no_weights_cfg=None):
        self.name, self.checkpoint = name, checkpoint
        self.no_weights_cfg = no_weights_cfg or {}

    def load(self, device, size, no_weights=False):
        from transformers import (AutoModel, Siglip2VisionConfig,
                                  Siglip2VisionModel)
        if no_weights:
            vm = Siglip2VisionModel(Siglip2VisionConfig(**self.no_weights_cfg))
            mean = std = (0.5, 0.5, 0.5)
        else:
            from transformers import AutoImageProcessor
            full = AutoModel.from_pretrained(self.checkpoint)
            vm = full.vision_model
            # colour statistics only; the processor's resizing is not used
            proc = AutoImageProcessor.from_pretrained(self.checkpoint)
            mean, std = tuple(proc.image_mean), tuple(proc.image_std)
        self.vm = vm.eval().to(device)
        c = self.vm.config
        self.budget = int(size)                     # patch budget, not pixels
        self.patch_size = int(c.patch_size)
        self.blocks = list(self.vm.encoder.layers)
        self.n_layers = len(self.blocks)
        self.width = int(c.hidden_size)
        self.prefix_tokens, self.has_cls, self.cls_index = 0, False, -1
        self.native_size = int(c.num_patches)        # 256 = the 16x16 grid
        self.mean, self.std = mean, std
        if not getattr(self.vm, "use_head", True) or \
                getattr(self.vm, "head", None) is None:
            raise RuntimeError(f"{self.name}: no attention-pooling head")
        self._mask = None
        self._out = [None] * self.n_layers
        for i, b in enumerate(self.blocks):
            b.register_forward_hook(self._hook(i))
        return self

    def _masked_mean(self, t):
        m = self._mask.unsqueeze(-1).to(t.dtype)    # [B, N, 1]
        return (t * m).sum(1) / m.sum(1).clamp_min(1)

    def _hook(self, i):
        def h(_m, _inp, o):
            t = (o[0] if isinstance(o, (tuple, list)) else o).float()
            self._out[i] = (self._masked_mean(t),
                            self._masked_mean(self.vm.post_layernorm(t)))
        return h

    @torch.inference_mode()
    def forward(self, patches, mask, spatial):
        if patches.shape[1] != self.budget:
            raise RuntimeError(f"{self.name}: {patches.shape[1]} patches, "
                               f"budget {self.budget}")
        self._mask = mask
        out = self.vm(pixel_values=patches, pixel_attention_mask=mask,
                      spatial_shapes=spatial)
        pm, pmn = (list(z) for z in zip(*self._out))
        self._out = [None] * self.n_layers
        self._last_hidden = out.last_hidden_state.float()
        return Readout(cls=None, patch_mean=torch.stack(pm, 1),
                       patch_mean_normed=torch.stack(pmn, 1),
                       pooled={"attention_pool": out.pooler_output.float()})

    def self_check(self, patches, mask, spatial):
        r = self.forward(patches, mask, spatial)
        mine = r.patch_mean_normed[:, -1]
        theirs = self._masked_mean(self._last_hidden)
        err = float((mine - theirs).abs().max() /
                    theirs.abs().max().clamp_min(1e-6))
        valid = mask.sum(1)
        return {"final_normed_patch_mean_vs_last_hidden_state_max_rel_err":
                err, "valid_patches_min_max": [int(valid.min()),
                                               int(valid.max())],
                "ok": err < 1e-2}
