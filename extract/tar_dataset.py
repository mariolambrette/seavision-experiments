"""Read the records of one main shard tar (WP9 onwards).

The read path is the development-shard one, unchanged: this subclasses
`DevShardDataset` and replaces only where the member offsets come from (a
header scan of the tar, rather than a saved index). `__getitem__` -- decode,
geometry, resize, normalise -- is inherited, so a record is preprocessed by
exactly the code WP8 scored.
"""
from __future__ import annotations

import tarfile

from .dataset import DevShardDataset

IMAGE_EXTS = {"jpg", "jpeg", "png"}


def index_tar(path):
    """-> (sorted keys that have an image, {key: {ext: [offset, size]}},
    keys found without an image). Header scan only; no data is read."""
    members = {}
    with tarfile.open(path, "r:") as tf:
        for m in tf:
            if m.isfile():
                key, _, ext = m.name.partition(".")
                members.setdefault(key, {})[ext] = [m.offset_data, m.size]
    keys = sorted(k for k, e in members.items() if set(e) & IMAGE_EXTS)
    no_image = sorted(set(members) - set(keys))
    # Keep only image members (the reader decodes the first non-json member,
    # so nothing else may be left in the way). Exactly one image per key.
    out = {}
    for k in keys:
        imgs = {e: v for e, v in members[k].items() if e in IMAGE_EXTS}
        if len(imgs) != 1:
            raise RuntimeError(f"{path}: {k} has {len(imgs)} image members")
        out[k] = imgs
    return keys, out, no_image


class TarDataset(DevShardDataset):
    def __init__(self, tar_path, keys, members, geometry, size, mean, std,
                 input_mode="fixed", patch_size=None):
        # Deliberately not calling DevShardDataset.__init__, which loads a
        # dev index; set exactly the attributes its __getitem__ uses.
        self.tar = tar_path
        self.keys, self.members = list(keys), members
        self.geometry, self.size, self.mean, self.std = geometry, size, mean, std
        self.input_mode, self.patch_size = input_mode, patch_size
        self._fh = None
