"""Read WP9 embeddings by record key.

WP9 layout (extract/run_wp9.py):

  <root>/<backbone>/<size>px/<geometry>/<set>/<tar stem>/
      keys.txt                record keys, in row order
      <token>@<layer>.npy     [rows, d] float16, unnormalised
      done.json               written last; a folder without it is incomplete

A readout is (token, layer) as in configs/extract/wp9_readouts.yaml. A "+"
readout (e.g. cls+patch_mean at 31) is rebuilt exactly as WP8 did: each half
L2-normalised, then concatenated.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

# Copied from extract.preprocess.SHARD_SET so WP10 needs no torch import;
# tests/test_wp10.py asserts the two are equal wherever torch is installed.
SHARD_SET = {"letterbox": "crops", "distort": "crops",
             "square_m00": "square-m00", "square_m10": "square-m10",
             "native": "crops"}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class Store:
    def __init__(self, root, backbone, size, geometry, sets=None):
        """Index every complete tar folder of the given sets (default: the
        animal set this geometry reads)."""
        self.base = os.path.join(root, backbone, f"{size}px", geometry)
        if not os.path.isdir(self.base):
            raise FileNotFoundError(self.base)
        self.sets = sets or [SHARD_SET[geometry]]
        self.loc = {}                       # key -> (folder, row)
        for s in self.sets:
            sd = os.path.join(self.base, s)
            if not os.path.isdir(sd):
                raise FileNotFoundError(sd)
            for stem in sorted(os.listdir(sd)):
                d = os.path.join(sd, stem)
                if stem.endswith(".part") or not os.path.isdir(d):
                    continue
                if not os.path.exists(os.path.join(d, "done.json")):
                    raise RuntimeError(f"{d} has no done.json: incomplete "
                                       f"WP9 output; run verify_wp9 first")
                with open(os.path.join(d, "keys.txt"),
                          encoding="utf-8") as fh:
                    for i, k in enumerate(fh.read().split("\n")):
                        if k:
                            if k in self.loc:
                                raise RuntimeError(f"duplicate key {k}")
                            self.loc[k] = (d, i)

    def has(self, key):
        return key in self.loc

    def _array(self, keys, name):
        by_dir = defaultdict(list)
        for j, k in enumerate(keys):
            d, i = self.loc[k]
            by_dir[d].append((j, i))
        out = None
        for d, pairs in by_dir.items():
            a = np.load(os.path.join(d, f"{name}.npy"), mmap_mode="r")
            js, rows = zip(*pairs)
            vals = np.asarray(a[np.asarray(rows)], np.float32)
            if out is None:
                out = np.empty((len(keys), vals.shape[1]), np.float32)
            out[np.asarray(js)] = vals
        return out

    def matrix(self, keys, token, layer):
        """Float32 [len(keys), d] for one readout, rows in key order."""
        missing = [k for k in keys if k not in self.loc]
        if missing:
            raise KeyError(f"{len(missing)} keys not in {self.base} "
                           f"(e.g. {missing[0]})")
        if "+" in token:
            a, b = token.split("+")
            return np.concatenate([l2(self.matrix(keys, a, layer)),
                                   l2(self.matrix(keys, b, layer))], 1)
        return self._array(keys, f"{token}@{layer}")
