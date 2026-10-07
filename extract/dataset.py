"""Read dev-shard records by key, with no full-tar scan.

`devshards.py` writes an index of member offsets, so each record is a seek
and a read. Map-style, so DataLoader workers can share the work and the
output order is fixed: the index's sorted key order, every run.
"""
from __future__ import annotations

import io
import json
import os

from PIL import Image
from torch.utils.data import Dataset

from .preprocess import naflex_inputs, preprocess


class DevShardDataset(Dataset):
    def __init__(self, dev_dir, set_name, geometry, size, mean, std,
                 limit=None, input_mode="fixed", patch_size=None):
        self.tar = os.path.join(dev_dir, f"dev-{set_name}.tar")
        with open(os.path.join(dev_dir, f"dev-{set_name}.index.json"),
                  encoding="utf-8") as fh:
            ix = json.load(fh)
        keys, self.members = ix["keys"], ix["members"]
        if limit:
            # an even stride through the sorted keys, so a small test run
            # still touches every source rather than the first one only
            step = max(1, len(keys) // limit)
            keys = keys[::step][:limit]
        self.keys = keys
        self.geometry, self.size, self.mean, self.std = geometry, size, mean, std
        # "fixed": one square size; "naflex": `size` is a patch budget
        self.input_mode, self.patch_size = input_mode, patch_size
        self._fh = None

    def __len__(self):
        return len(self.keys)

    def _read(self, off, n):
        if self._fh is None:                 # one handle per worker
            self._fh = open(self.tar, "rb")
        self._fh.seek(off)
        return self._fh.read(n)

    def __getitem__(self, i):
        key = self.keys[i]
        m = self.members[key]
        ext = next(e for e in m if e != "json")
        img = Image.open(io.BytesIO(self._read(*m[ext])))
        if self.input_mode == "naflex":
            return (i, *naflex_inputs(img, self.geometry, self.size,
                                      self.patch_size, self.mean, self.std))
        x = preprocess(img, self.geometry, self.size, self.mean, self.std)
        return i, x
