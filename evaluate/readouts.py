"""Turn an extraction run's arrays into named feature matrices.

A readout is (token, layer). Tokens:
  cls                    the CLS token (absent for SigLIP 2)
  patch_mean             mean of patch tokens
  patch_mean_normed      mean of patch tokens after the model's final norm
  cls+patch_mean         both, each L2-normalised, then concatenated
  cls+patch_mean_normed  likewise
plus every pooled output the model has (projection, attention pooling,
pooler), as layer "pooled".

Concatenations normalise each half first so neither dominates the cosine by
having the larger norm. Concatenation doubles the width; cosine is not
neutral to width, which is why readouts are compared WITHIN a backbone only.
"""
from __future__ import annotations

import json
import os

import numpy as np


class Run:
    def __init__(self, run_dir):
        self.dir = run_dir
        with open(os.path.join(run_dir, "manifest.json"),
                  encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        with open(os.path.join(run_dir, "keys.txt"), encoding="utf-8") as fh:
            self.keys = [k for k in fh.read().split("\n") if k]
        self.row = {k: i for i, k in enumerate(self.keys)}
        self._arr = {}
        b = self.manifest["backbone"]
        self.n_layers = b["n_layers"]
        self.has_cls = b["has_cls"]
        self.pooled = sorted(n[len("pooled_"):] for n in self.manifest["arrays"]
                             if n.startswith("pooled_"))

    def _a(self, name):
        # Whole array into memory, read once sequentially. Arrays are stored
        # [records, layers, width], so reading one layer from disk would
        # re-read the file once per layer. Peak is about two token arrays
        # (a concatenation): ~10 GB for the largest model on the dev set.
        if name not in self._arr:
            self._arr[name] = np.load(os.path.join(self.dir, f"{name}.npy"))
        return self._arr[name]

    def release(self, keep=()):
        """Free cached arrays except those named in `keep`."""
        for k in [k for k in self._arr if k not in keep]:
            del self._arr[k]

    def readouts(self):
        """All (token, layer) pairs available in this run."""
        toks = ["patch_mean", "patch_mean_normed"]
        if self.has_cls:
            toks = ["cls"] + toks + ["cls+patch_mean", "cls+patch_mean_normed"]
        out = [(t, layer) for layer in range(self.n_layers) for t in toks]
        return out + [(p, "pooled") for p in self.pooled]

    def matrix(self, token, layer, rows):
        """Float32 [len(rows), d], rows in the order given."""
        rows = np.asarray(rows)
        if layer == "pooled":
            return np.asarray(self._a(f"pooled_{token}")[rows], np.float32)
        if "+" in token:
            a, b = token.split("+")
            return np.concatenate([l2(self.matrix(a, layer, rows)),
                                   l2(self.matrix(b, layer, rows))], 1)
        return np.asarray(self._a(token)[rows, layer], np.float32)


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
