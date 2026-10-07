"""Reference/query draws by deployment -- made once, saved, shared.

For each draw and each evaluable species, the species' deployments are
shuffled and taken in order into the reference side until it holds at least
`k_max` crops; every remaining deployment is the query side. If that leaves
fewer than `q_min` queries, the shuffle is redrawn (a recorded count). The
k = 1, 5, 20 references are nested: the first k of the reference crops in a
fixed shuffled order. So within a draw, comparing k compares more labels,
not different labels.

Draws depend only on the labels, a pool and a seed -- never on a backbone or
a readout -- and are written to a JSON file that every evaluation reads. All
readouts are therefore scored on exactly the same references and queries:
the comparison between them is paired, and later work packages can reuse the
very same draws.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np


def _rng(seed, draw, species):
    h = int(hashlib.sha256(f"{seed}|{draw}|{species}".encode())
            .hexdigest()[:16], 16)
    return np.random.default_rng(h)


def make_draws(labels, species_deps, n_draws, k_max, q_min, seed,
               max_tries=1000):
    """-> {"draws": [ {species: {"ref": [keys], "query": [keys],
                                 "ref_deps": [...], "query_deps": [...]}} ],
           "redraws": int, "failed": [species...]}"""
    keys_by = defaultdict(list)                  # (species, dep) -> keys
    for r in labels:
        sp = r["species"]
        if sp in species_deps:
            keys_by[(sp, (r["source"], r["deployment"]))].append(r["key"])
    draws, redraws, failed = [], 0, set()
    for d in range(n_draws):
        out = {}
        for sp in sorted(species_deps):
            deps = sorted(species_deps[sp])
            rng = _rng(seed, d, sp)
            for _ in range(max_tries):
                order = [deps[i] for i in rng.permutation(len(deps))]
                ref_deps, n = [], 0
                for dep in order:
                    if n >= k_max:
                        break
                    ref_deps.append(dep)
                    n += species_deps[sp][dep]
                q_deps = [x for x in order if x not in ref_deps]
                nq = sum(species_deps[sp][x] for x in q_deps)
                if n >= k_max and nq >= q_min:
                    break
                redraws += 1
            else:
                failed.add(sp)
                continue
            ref = sorted(k for dep in ref_deps for k in keys_by[(sp, dep)])
            ref = [ref[i] for i in rng.permutation(len(ref))]  # nested order
            query = sorted(k for dep in q_deps for k in keys_by[(sp, dep)])
            out[sp] = {"ref": ref, "query": query,
                       "ref_deps": ["|".join(x) for x in ref_deps],
                       "query_deps": ["|".join(x) for x in q_deps]}
        draws.append(out)
    return {"draws": draws, "redraws": redraws, "failed": sorted(failed)}


def check_draws(dr):
    """Deployment-disjointness, every draw and species. Raises on failure."""
    for d, draw in enumerate(dr["draws"]):
        for sp, s in draw.items():
            if set(s["ref_deps"]) & set(s["query_deps"]):
                raise AssertionError(f"draw {d} {sp}: a deployment is on "
                                     "both sides")
            if set(s["ref"]) & set(s["query"]):
                raise AssertionError(f"draw {d} {sp}: a key on both sides")
    return True


def save(dr, path, meta):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({**meta, **dr}, fh)


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
