"""open_clip ViTs: CLIP LAION-2B, BioCLIP 2 (both ViT-L/14) and BioCLIP 2.5
Huge (ViT-H/14). One adapter, three checkpoints.

Token layout (probe, 2 October 2026): [CLS, 256 patches] at 224 px, batch-
first, one prefix token. Sizes above 224 need the position embeddings
interpolated (open_clip `force_image_size`), and are labelled as such.

How open_clip pools (pool_type 'tok', final_ln_after_pool False):
ln_post is applied to every token, then the CLS is taken, then multiplied by
`proj`. So "pre-projection" here is ln_post(CLS) and "post-projection" is
that times `proj` -- reconstructed from the block outputs and checked against
`encode_image` on the first batch.
"""
from __future__ import annotations

import torch

from .base import Adapter, Readout

OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


class OpenClipAdapter(Adapter):
    library = "open_clip"

    def __init__(self, name, arch, checkpoint):
        self.name, self.arch, self.checkpoint = name, arch, checkpoint

    def load(self, device, size, no_weights=False):
        import open_clip
        native = 224
        kw = {} if size == native else {"force_image_size": size}
        self.interpolated = size != native
        if no_weights:
            model = open_clip.create_model(self.arch, pretrained=None, **kw)
        elif self.checkpoint.startswith("hf-hub:"):
            model = open_clip.create_model(self.checkpoint, **kw)
        else:
            model = open_clip.create_model(self.arch, pretrained=self.checkpoint,
                                           **kw)
        self.model = model.eval().to(device)
        v = self.visual = self.model.visual
        if getattr(v, "pool_type", "tok") != "tok" or \
                getattr(v, "final_ln_after_pool", False) or \
                getattr(v, "attn_pool", None) is not None:
            raise RuntimeError(f"{self.name}: unexpected pooling "
                               f"({getattr(v, 'pool_type', None)}); this "
                               "adapter assumes CLS pooling after ln_post")
        self.blocks = list(v.transformer.resblocks)
        self.n_layers = len(self.blocks)
        self.width = int(v.transformer.width)
        ps = v.patch_size
        self.patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        self.prefix_tokens, self.has_cls, self.native_size = 1, True, native
        cfg = getattr(v, "preprocess_cfg", {}) or {}
        self.mean = tuple(cfg.get("mean", OPENAI_MEAN))
        self.std = tuple(cfg.get("std", OPENAI_STD))
        self._out = [None] * self.n_layers
        for i, b in enumerate(self.blocks):
            b.register_forward_hook(self._hook(i))
        return self

    def _hook(self, i):
        # Reduce inside the hook, so no layer's full token grid is kept: the
        # whole stack would be ~1.6 GB at batch 128 for a ViT-L, and several
        # times that for a ViT-H at 336 px.
        def h(_m, _inp, o):
            t = (o[0] if isinstance(o, (tuple, list)) else o).float()
            patches = t[:, self.prefix_tokens:]
            self._out[i] = (t[:, 0], patches.mean(1),
                            self.visual.ln_post(patches).mean(1))
        return h

    @torch.inference_mode()
    def forward(self, x):
        post = self.model.encode_image(x)          # fills self._out
        cls, pm, pmn = (list(z) for z in zip(*self._out))
        self._out = [None] * self.n_layers
        cls = torch.stack(cls, 1)
        pre = self.visual.ln_post(cls[:, -1])
        pooled = {"pre_projection": pre}
        if self.visual.proj is not None:
            pooled["post_projection"] = pre @ self.visual.proj.float()
        self._last_encode = post.float()
        return Readout(cls=cls, patch_mean=torch.stack(pm, 1),
                       patch_mean_normed=torch.stack(pmn, 1), pooled=pooled)

    def self_check(self, x):
        r = self.forward(x)
        mine = r.pooled.get("post_projection", r.pooled["pre_projection"])
        theirs = self._last_encode
        err = float((mine - theirs).abs().max() /
                    theirs.abs().max().clamp_min(1e-6))
        return {"reconstructed_vs_encode_image_max_rel_err": err,
                "ok": err < 1e-2}
