"""The WP8 readout decision, turned into the arrays WP9 stores.

`configs/extract/wp9_readouts.yaml` names readouts as (token, layer), the
same vocabulary the WP8 sweep was scored in. Storage is per *array*:

  cls@14.npy                 the CLS token after block 14
  patch_mean@19.npy          mean of the raw patch tokens after block 19
  post_projection@pooled.npy one of the model's own pooled outputs

A "+" readout (e.g. cls+patch_mean at 31) is stored as its two halves and
rebuilt at evaluation exactly as WP8 did (L2-normalise each half, then
concatenate), so it is never stored pre-concatenated. `readout_arrays`
makes that expansion, and `select` pulls those arrays out of one batch's
`Readout`, with the per-layer arrays indexed exactly as the sweep's were.
"""
from __future__ import annotations

import yaml

LAYER_TOKENS = ("cls", "patch_mean", "patch_mean_normed")


def load_spec(path, backbone):
    with open(path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)["backbones"]
    if backbone not in spec:
        raise KeyError(f"{backbone} not in {path}; has {sorted(spec)}")
    return spec[backbone]


def array_name(token, layer):
    return f"{token}@{layer}"


def readout_arrays(readouts):
    """[(token, layer)] -> ordered, de-duplicated array names to store."""
    out = []
    for r in readouts:
        tok, layer = r["token"], r["layer"]
        parts = tok.split("+") if "+" in tok else [tok]
        for p in parts:
            n = array_name(p, layer)
            if n not in out:
                out.append(n)
    return out


def check_against(adapter, names):
    """Fail before any work if a requested array cannot exist."""
    for n in names:
        tok, layer = n.split("@")
        if layer == "pooled":
            continue                 # checked on the first batch (see select)
        if tok not in LAYER_TOKENS:
            raise ValueError(f"{n}: unknown token {tok!r}")
        if tok == "cls" and not adapter.has_cls:
            raise ValueError(f"{n}: {adapter.name} has no CLS token")
        if not 0 <= int(layer) < adapter.n_layers:
            raise ValueError(f"{n}: layer out of range 0..{adapter.n_layers-1}")


def select(readout, names):
    """-> {name: tensor [B, d]} for one batch."""
    out = {}
    for n in names:
        tok, layer = n.split("@")
        if layer == "pooled":
            if tok not in readout.pooled:
                raise KeyError(f"{n}: pooled outputs are "
                               f"{sorted(readout.pooled)}")
            out[n] = readout.pooled[tok]
        else:
            out[n] = getattr(readout, tok)[:, int(layer)]
    return out
