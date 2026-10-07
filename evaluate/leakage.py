"""The deployment-leakage diagnostic (WP10's probe borrowed early).

For a readout: mean cosine similarity of pairs that share a species but not
a deployment (what should be close), against pairs that share a deployment
but not a species (what should not). A readout that scores well but puts
same-deployment-different-species pairs close is encoding the place, not
the animal. Pairs are sampled with a fixed seed, so every readout is
measured on the same pairs.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def sample_pairs(species, deps, n_pairs=20000, seed=0):
    """species, deps: arrays per row. -> (same_sp_diff_dep, same_dep_diff_sp)
    each an int array [m, 2] of row indices."""
    rng = np.random.default_rng(seed)
    by_sp, by_dep = defaultdict(list), defaultdict(list)
    for i, (s, d) in enumerate(zip(species, deps)):
        by_sp[s].append(i)
        by_dep[d].append(i)
    out = []
    for groups, other in ((by_sp, deps), (by_dep, species)):
        gl = [np.array(v) for v in groups.values() if len(v) > 1]
        pairs = []
        tries = 0
        while len(pairs) < n_pairs and gl and tries < n_pairs * 20:
            tries += 1
            g = gl[rng.integers(len(gl))]
            a, b = rng.choice(g, 2, replace=False)
            if other[a] != other[b]:
                pairs.append((a, b))
        out.append(np.array(pairs, dtype=np.int64).reshape(-1, 2))
    return out[0], out[1]


def leakage(x_normed, ssdd, sdds):
    def m(p):
        if not len(p):
            return float("nan")
        return float((x_normed[p[:, 0]] * x_normed[p[:, 1]]).sum(1).mean())
    a, b = m(ssdd), m(sdds)
    return {"same_species_diff_deployment": a,
            "same_deployment_diff_species": b, "margin": a - b,
            "n_pairs": [int(len(ssdd)), int(len(sdds))]}
