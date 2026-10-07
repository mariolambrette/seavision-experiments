#!/usr/bin/env python3
"""Step 0 of the WP8 readout sweep: what does each backbone actually expose?

Loads each of the six backbones, pushes a small synthetic batch through it at
two or more input sizes, and records -- per transformer block -- how many
tokens come out and how wide they are. From that it works out, rather than
assumes:

  * where the patch tokens are, and how many non-patch ("prefix") tokens sit
    in front of them (CLS, registers), solved from two input sizes;
  * the patch size, from how the token count scales with input size;
  * whether a block's output is batch-first ([B, N, D]) or sequence-first;
  * what pooled outputs exist (CLS after norm, projection head, attention
    pooling) and their widths;
  * whether the checkpoint accepts a size other than its native one;
  * the colour statistics the model expects;
  * with --time on a GPU, rough throughput (images/s) per size.

The extraction interface is designed against this report, not against one
model. Nothing here touches the shards or writes outside --out.

    python extract\\probe_backbones.py --out D:\\marineai\\scratch\\wp8_probe
    python extract\\probe_backbones.py --out ... --only clip_laion2b --device cuda:0 --time

--no-weights builds every architecture from its config with random weights
(no download), which checks shapes and plumbing but not colour statistics.
DINOv3's weights are gated on Hugging Face: accept the licence on the model
page and run `huggingface-cli login` once before the first real run.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

# ----------------------------------------------------------------- registry
# One entry per backbone (plan section 5.3). "sizes" are the square input
# sizes to probe; the first is the checkpoint's native size. SigLIP 2 NaFlex
# is probed on aspect ratios and patch budgets instead (see probe_naflex).
BACKBONES = {
    "clip_laion2b": dict(lib="open_clip", arch="ViT-L-14",
                         weights="laion2b_s32b_b82k", sizes=[224, 336]),
    "bioclip2": dict(lib="open_clip", arch="ViT-L-14",
                     weights="hf-hub:imageomics/bioclip-2", sizes=[224, 336]),
    "bioclip25_huge": dict(lib="open_clip", arch="ViT-H-14",
                           weights="hf-hub:imageomics/bioclip-2.5-vith14",
                           sizes=[224, 336]),
    "dinov3_vitl16": dict(lib="hf_dinov3",
                          weights="facebook/dinov3-vitl16-pretrain-lvd1689m",
                          sizes=[256, 512]),
    # Candidate, not yet in the plan: a size ablation for self-supervised
    # training, the counterpart of BioCLIP 2 vs 2.5 Huge. Probed so the
    # inclusion decision rests on measured cost; decided before the sweep.
    "dinov3_vith16plus": dict(lib="hf_dinov3",
                              weights="facebook/dinov3-vith16plus-pretrain-lvd1689m",
                              sizes=[256, 512],
                              no_weights_cfg=dict(hidden_size=1280,
                                                  num_hidden_layers=32,
                                                  num_attention_heads=20,
                                                  intermediate_size=5120)),
    "bio_dino": dict(lib="birder",
                     weights="vit_reg4_so150m_p14_ls_dino-v2-bio-224px",
                     net="vit_reg4_so150m_p14_ls", sizes=[224, 336]),
    "siglip2_naflex": dict(lib="hf_siglip2",
                           weights="google/siglip2-so400m-patch16-naflex",
                           budgets=[256, 576],
                           shapes=[(256, 256), (512, 128), (128, 512)]),
}


def versions():
    out = {"python": platform.python_version(), "torch": torch.__version__}
    for m in ("open_clip", "transformers", "timm", "birder"):
        try:
            mod = importlib.import_module(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception as e:                         # noqa: BLE001
            out[m] = f"not importable ({type(e).__name__})"
    return out


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def find_blocks(root: nn.Module):
    """The transformer blocks: the longest ModuleList/Sequential under root
    whose children are all the same class. Returns (dotted name, list)."""
    best = ("", [])
    for name, mod in root.named_modules():
        if isinstance(mod, (nn.ModuleList, nn.Sequential)) and len(mod) > 1:
            kids = list(mod)
            if len({type(k) for k in kids}) == 1 and len(kids) > len(best[1]):
                best = (name, kids)
    return best


class Recorder:
    """Forward hooks on every block; keeps each block's output tensor."""

    def __init__(self, blocks):
        self.out = [None] * len(blocks)
        self.h = [b.register_forward_hook(self._mk(i))
                  for i, b in enumerate(blocks)]

    def _mk(self, i):
        def hook(_m, _inp, o):
            self.out[i] = (o[0] if isinstance(o, (tuple, list)) else o).detach()
        return hook

    def close(self):
        for h in self.h:
            h.remove()


def layout(t, batch):
    """[B, N, D] or [N, B, D] -> ('BND', N, D)."""
    if t.dim() != 3:
        return ("?", tuple(t.shape), None)
    if t.shape[0] == batch:
        return ("BND", int(t.shape[1]), int(t.shape[2]))
    if t.shape[1] == batch:
        return ("NBD", int(t.shape[0]), int(t.shape[2]))
    return ("?", tuple(t.shape), None)


def solve_patch(tokens_by_size):
    """Token count N(r) = prefix + (r/p)^2. Find p in {14, 16} and prefix
    consistent across every probed size."""
    for p in (14, 16, 8, 32):
        prefixes = {n - (r // p) ** 2 for r, n in tokens_by_size.items()
                    if r % p == 0}
        if len(prefixes) == 1 and len(tokens_by_size) >= 2 and \
                all(r % p == 0 for r in tokens_by_size):
            pre = prefixes.pop()
            if 0 <= pre <= 16:
                return p, pre
    return None, None


def timed(fn, n_warm=2, n_rep=5):
    for _ in range(n_warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_rep


# ------------------------------------------------------------------ loaders
# Each returns (vision_root, forward(x) -> dict of pooled outputs, info).

def load_open_clip(spec, dev, no_weights, force_size=None):
    import open_clip
    kw = {"force_image_size": force_size} if force_size else {}
    if no_weights:
        model = open_clip.create_model(spec["arch"], pretrained=None, **kw)
        pre_cfg = {}
    else:
        if spec["weights"].startswith("hf-hub:"):
            model, _, _ = open_clip.create_model_and_transforms(
                spec["weights"], **kw)
        else:
            model, _, _ = open_clip.create_model_and_transforms(
                spec["arch"], pretrained=spec["weights"], **kw)
        pre_cfg = dict(getattr(model.visual, "preprocess_cfg", {}) or {})
    model = model.eval().to(dev)
    vis = model.visual
    proj = getattr(vis, "proj", None)

    def fwd(x):
        emb = model.encode_image(x)
        return {"encode_image (after projection)": emb}
    info = {"preprocess_cfg": {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in pre_cfg.items()},
            "native_image_size": getattr(vis, "image_size", None),
            "projection": None if proj is None else list(proj.shape),
            "patch_size": list(getattr(vis, "patch_size", ())) or None,
            "params": n_params(vis)}
    return vis, fwd, info


def load_hf_dinov3(spec, dev, no_weights):
    from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel
    if no_weights:
        # approximate architecture for shape checks only
        kw = spec.get("no_weights_cfg") or dict(
            hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
            intermediate_size=4096)
        cfg = DINOv3ViTConfig(patch_size=16, num_register_tokens=4, **kw)
        model = DINOv3ViTModel(cfg)
    else:
        model = AutoModel.from_pretrained(spec["weights"])
    model = model.eval().to(dev)
    c = model.config

    def fwd(x):
        o = model(pixel_values=x)
        return {"pooler_output (CLS after norm)": o.pooler_output}
    info = {"config": {k: getattr(c, k, None) for k in (
        "hidden_size", "num_hidden_layers", "patch_size",
        "num_register_tokens", "image_size")},
        "params": n_params(model)}
    return model, fwd, info


def load_birder(spec, dev, no_weights):
    import birder
    from birder.model_registry import registry
    size = (spec["sizes"][0],) * 2
    if no_weights:
        net = registry.net_factory(spec["net"], 0, size=size)
        rgb = None
    else:
        net, model_info = birder.load_pretrained_model(
            spec["weights"], inference=True, device=dev)
        rgb = getattr(model_info, "rgb_stats", None)
    net = net.eval().to(dev)

    def fwd(x):
        return {"embedding()": net.embedding(x)}
    info = {"rgb_stats": rgb if rgb is None or isinstance(rgb, dict)
            else str(rgb),
            "has_adjust_size": hasattr(net, "adjust_size"),
            "params": n_params(net)}
    return net, fwd, info


def probe_square(name, spec, args, dev):
    loader = {"open_clip": load_open_clip, "hf_dinov3": load_hf_dinov3,
              "birder": load_birder}[spec["lib"]]
    root, fwd, info = loader(spec, dev, args.no_weights)
    bname, blocks = find_blocks(root)
    rep = {"library": spec["lib"], "weights": spec["weights"],
           "blocks_module": bname, "n_blocks": len(blocks),
           "block_class": type(blocks[0]).__name__ if blocks else None,
           **info, "sizes": {}}
    rec = Recorder(blocks)
    tokens = {}
    for r in spec["sizes"]:
        entry = {}
        try:
            if spec["lib"] == "birder" and r != spec["sizes"][0]:
                # birder ViTs carry a fixed position-embedding grid; resizing
                # is an explicit, recorded operation, never silent.
                root.adjust_size((r, r))
                entry["adjusted_size"] = True
            x = torch.randn(args.batch, 3, r, r, device=dev)
            with torch.inference_mode():
                pooled = fwd(x)
            lay = [layout(t, args.batch) for t in rec.out]
            forms = {lo[0] for lo in lay}
            ns = {lo[1] for lo in lay}
            ds = {lo[2] for lo in lay}
            entry.update({
                "block_output_layout": sorted(forms),
                "tokens_per_block": sorted(ns),
                "width_per_block": sorted(d for d in ds if d is not None),
                "pooled": {k: list(v.shape) for k, v in pooled.items()},
            })
            if len(ns) == 1:
                tokens[r] = next(iter(ns))
            if args.time and dev.type == "cuda":
                xb = torch.randn(args.time_batch, 3, r, r, device=dev)

                def go():
                    with torch.inference_mode(), torch.autocast(
                            "cuda", dtype=torch.float16):
                        fwd(xb)
                s = timed(go)
                entry["images_per_s_fp16"] = round(args.time_batch / s, 1)
        except Exception as e:                         # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {e}"[:400]
            if spec["lib"] == "open_clip":
                # Fixed position-embedding grid. open_clip can interpolate it
                # at load time; record that the size needs it, then retry.
                try:
                    root2, fwd2, _ = load_open_clip(spec, dev, args.no_weights,
                                                    force_size=r)
                    _, blocks2 = find_blocks(root2)
                    rec2 = Recorder(blocks2)
                    x = torch.randn(args.batch, 3, r, r, device=dev)
                    with torch.inference_mode():
                        pooled = fwd2(x)
                    lay = [layout(t, args.batch) for t in rec2.out]
                    rec2.close()
                    ns = {lo[1] for lo in lay}
                    entry = {"needs": "position-embedding interpolation "
                                      "(open_clip force_image_size)",
                             "block_output_layout": sorted({lo[0] for lo in lay}),
                             "tokens_per_block": sorted(ns),
                             "width_per_block": sorted({lo[2] for lo in lay}),
                             "pooled": {k: list(v.shape)
                                        for k, v in pooled.items()}}
                    if len(ns) == 1:
                        tokens[r] = next(iter(ns))
                    del root2, fwd2
                except Exception as e2:                # noqa: BLE001
                    entry["retry_error"] = f"{type(e2).__name__}: {e2}"[:400]
        rep["sizes"][str(r)] = entry
    rec.close()
    p, pre = solve_patch(tokens)
    rep["inferred_patch_size"] = p
    rep["inferred_prefix_tokens"] = pre
    rep["prefix_note"] = (
        "tokens before the patch grid (CLS + registers); patches are "
        f"tokens[{pre}:]" if pre is not None else
        "could not solve from the probed sizes (see per-size errors)")
    return rep


def probe_naflex(name, spec, args, dev):
    """SigLIP 2 NaFlex: no CLS; variable patch count per image; padded."""
    from PIL import Image
    from transformers import (Siglip2ImageProcessor, Siglip2VisionConfig,
                              Siglip2VisionModel, AutoModel)
    if args.no_weights:
        cfg = Siglip2VisionConfig(hidden_size=1152, intermediate_size=4304,
                                  num_hidden_layers=27,
                                  num_attention_heads=16, patch_size=16,
                                  num_patches=256)
        vm = Siglip2VisionModel(cfg)
        proc = Siglip2ImageProcessor()
    else:
        full = AutoModel.from_pretrained(spec["weights"])
        vm = full.vision_model if hasattr(full, "vision_model") else full
        proc = Siglip2ImageProcessor.from_pretrained(spec["weights"])
    vm = vm.eval().to(dev)
    bname, blocks = find_blocks(vm)
    rep = {"library": spec["lib"], "weights": spec["weights"],
           "blocks_module": bname, "n_blocks": len(blocks),
           "block_class": type(blocks[0]).__name__ if blocks else None,
           "params": n_params(vm),
           "image_mean": getattr(proc, "image_mean", None),
           "image_std": getattr(proc, "image_std", None),
           "has_head (attention pooling)": hasattr(
               getattr(vm, "vision_model", vm), "head"),
           "budgets": {}}
    rec = Recorder(blocks)
    rng = np.random.default_rng(0)
    imgs = [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
            for (w, h) in spec["shapes"]]
    for b in spec["budgets"]:
        entry = {}
        try:
            inp = proc(images=imgs, return_tensors="pt", max_num_patches=b)
            inp = {k: v.to(dev) for k, v in inp.items()}
            with torch.inference_mode():
                o = vm(**inp)
            lay = [layout(t, len(imgs)) for t in rec.out]
            mask = inp.get("pixel_attention_mask")
            entry = {
                "block_output_layout": sorted({lo[0] for lo in lay}),
                "tokens_per_block (padded)": sorted({lo[1] for lo in lay}),
                "width_per_block": sorted({lo[2] for lo in lay}),
                "valid_patches_per_image": None if mask is None else
                [int(m.sum()) for m in mask],
                "spatial_shapes": inp["spatial_shapes"].tolist()
                if "spatial_shapes" in inp else None,
                "image_shapes_wxh": [list(s) for s in spec["shapes"]],
                "pooler_output": list(o.pooler_output.shape)
                if getattr(o, "pooler_output", None) is not None else None,
            }
        except Exception as e:                         # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {e}"[:400]
        rep["budgets"][str(b)] = entry
    rec.close()
    rep["prefix_note"] = ("no CLS token: tokens are patches plus padding; "
                          "mean-of-patches must use pixel_attention_mask")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"subset of {list(BACKBONES)}")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--time", action="store_true",
                    help="measure fp16 throughput (GPU only)")
    ap.add_argument("--time-batch", type=int, default=64)
    ap.add_argument("--no-weights", action="store_true",
                    help="random-init architectures; no downloads")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    names = args.only or list(BACKBONES)
    report = {"versions": versions(), "device": str(dev),
              "no_weights": args.no_weights, "backbones": {}}
    for name in names:
        spec = BACKBONES[name]
        print(f"\n=== {name} ({spec['lib']}) ===", flush=True)
        t0 = time.time()
        try:
            rep = (probe_naflex if spec["lib"] == "hf_siglip2"
                   else probe_square)(name, spec, args, dev)
        except Exception as e:                         # noqa: BLE001
            rep = {"load_error": f"{type(e).__name__}: {e}"[:600],
                   "traceback": traceback.format_exc()[-2000:]}
        rep["seconds"] = round(time.time() - t0, 1)
        report["backbones"][name] = rep
        # one-screen summary
        if "load_error" in rep:
            print(f"  LOAD FAILED: {rep['load_error']}")
        else:
            print(f"  blocks: {rep['n_blocks']} x {rep['block_class']} "
                  f"at '{rep['blocks_module']}'")
            for k in ("sizes", "budgets"):
                for s, e in rep.get(k, {}).items():
                    if "error" in e:
                        print(f"  {k[:-1]} {s}: ERROR {e['error'][:150]}")
                    else:
                        tok = e.get("tokens_per_block") or \
                            e.get("tokens_per_block (padded)")
                        extra = e.get("valid_patches_per_image", "")
                        print(f"  {k[:-1]} {s}: tokens {tok} width "
                              f"{e['width_per_block']} layout "
                              f"{e['block_output_layout']}"
                              f"{'  valid ' + str(extra) if extra else ''}"
                              f"{'  ' + str(e['images_per_s_fp16']) + ' img/s' if 'images_per_s_fp16' in e else ''}"
                              f"{'  [needs ' + e['needs'] + ']' if 'needs' in e else ''}")
            if "inferred_patch_size" in rep:
                print(f"  patch {rep['inferred_patch_size']}  prefix tokens "
                      f"{rep['inferred_prefix_tokens']}")
    path = os.path.join(args.out, "probe_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
