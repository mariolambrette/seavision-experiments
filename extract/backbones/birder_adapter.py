"""Bio-DINO (birder): vit_reg4_so150m_p14_ls, DINOv2 self-supervision on
biodiversity images.

Token layout -- read from birder's ViT source, NOT the DINOv3 layout:
  [4 registers, CLS, patches]   (forward_features: "prepend in order
  [REG..., CLS, PATCH...]")
so the CLS is token 4 (`num_reg_tokens`), and patches start at token 5
(`num_special_tokens`). An adapter reading token 0 as the CLS would silently
read a register.

How it pools (ViT._pool with a class token and no attention-pool head):
`embedding()` = embedding_norm(norm(x)[:, num_reg_tokens]), and for this
checkpoint embedding_norm is the identity. So "embedding" here is
norm(last block)[:, cls_index], checked against `net.embedding()` on the
first batch. No projection head.

Resolution: the checkpoint has a fixed position-embedding grid. Any other
size goes through birder's `adjust_size` (bicubic interpolation of the
position embeddings) and is labelled interpolated. birder also publishes
separate 252 px and 336 px checkpoints of the same model; using one of those
instead of interpolating is a resolution decision, made in the config via
`checkpoint_by_size`, not here.
"""
from __future__ import annotations

import torch

from .base import Adapter, Readout


class BirderAdapter(Adapter):
    library = "birder"

    def __init__(self, name, net_name, checkpoint, checkpoint_size,
                 checkpoint_by_size=None):
        self.name, self.net_name = name, net_name
        self.default_checkpoint, self.default_size = checkpoint, checkpoint_size
        # optional {size: (checkpoint, its native size)}
        self.checkpoint_by_size = checkpoint_by_size or {}

    def load(self, device, size, no_weights=False):
        import birder
        from birder.model_registry import registry
        ckpt, ck_size = self.checkpoint_by_size.get(
            size, (self.default_checkpoint, self.default_size))
        self.checkpoint, self.native_size = ckpt, ck_size
        if no_weights:
            net = registry.net_factory(self.net_name, 0,
                                       size=(ck_size, ck_size))
            rgb = None
        else:
            net, info = birder.load_pretrained_model(
                ckpt, inference=True, device=device)
            rgb = getattr(info, "rgb_stats", None)
        if size % net.patch_size:
            raise ValueError(f"{self.name}: size {size} is not a multiple of "
                             f"the patch size {net.patch_size}")
        self.interpolated = size != ck_size
        if self.interpolated:
            net.adjust_size((size, size))
        if net.attn_pool is not None or net.class_token is None:
            raise RuntimeError(f"{self.name}: unexpected pooling (attention "
                               "pool or no class token)")
        if not isinstance(net.embedding_norm, torch.nn.Identity):
            raise RuntimeError(f"{self.name}: embedding_norm is not identity; "
                               "pooled reconstruction would be wrong")
        self.net = net.eval().to(device)
        self.blocks = list(self.net.encoder.block)
        self.n_layers = len(self.blocks)
        self.width = int(self.net.norm.normalized_shape[0])
        self.patch_size = int(self.net.patch_size)
        self.cls_index = int(self.net.num_reg_tokens)
        self.prefix_tokens = int(self.net.num_special_tokens)
        self.has_cls = True
        if rgb is None:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif isinstance(rgb, dict):
            mean, std = tuple(rgb["mean"]), tuple(rgb["std"])
        else:
            mean, std = tuple(rgb.mean), tuple(rgb.std)
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
            self._out[i] = (t[:, self.cls_index], patches.mean(1),
                            self.net.norm(patches).mean(1))
        return h

    @torch.inference_mode()
    def forward(self, x):
        emb = self.net.embedding(x)                      # fills self._out
        cls, pm, pmn = (list(z) for z in zip(*self._out))
        self._out = [None] * self.n_layers
        cls = torch.stack(cls, 1)
        pooled = {"embedding": self.net.norm(cls[:, -1])}
        self._last_embedding = emb.float()
        return Readout(cls=cls, patch_mean=torch.stack(pm, 1),
                       patch_mean_normed=torch.stack(pmn, 1), pooled=pooled)

    def self_check(self, x):
        r = self.forward(x)
        mine, theirs = r.pooled["embedding"], self._last_embedding
        err = float((mine - theirs).abs().max() /
                    theirs.abs().max().clamp_min(1e-6))
        return {"reconstructed_vs_embedding_max_rel_err": err,
                "ok": err < 1e-2}
