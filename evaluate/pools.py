"""Which species are evaluable, per pool of sources.

A pool is a set of sources scored together. Three are used in WP8:

  ozfish            OzFish alone. The only pool present in EVERY geometry
                    (FishWIO has no frames, so no square crops), and so the
                    one the geometry stability check must use.
  ozfish_fishwio    OzFish + FishWIO. Letterbox and distort only.
  fathomnet         The FathomNet slice. Inferred deployments; always
                    reported separately, never pooled with the others.

A species is evaluable in a pool if its crops can be split by deployment
into at least `k_max` references and at least `q_min` queries: crop total
>= k_max + q_min, and at least q_min crops outside its largest deployment.
Units are crops (decided 5 October 2026): splitting by deployment keeps both
stereo views and every frame of an animal on one side, so no animal can be on
both sides, but k references may be fewer than k distinct animals.
"""
from __future__ import annotations

from collections import Counter, defaultdict

POOLS = {"ozfish": {"ozfish"},
         "ozfish_fishwio": {"ozfish", "fishwio"},
         "fathomnet": {"fathomnet"}}


def evaluable_species(labels, sources, k_max=20, q_min=10):
    """labels: list of dicts (labels.csv rows). -> {species: Counter(dep)}"""
    by_sp = defaultdict(Counter)
    for r in labels:
        if r["source"] in sources and r["rank"] == "species" and r["species"]:
            by_sp[r["species"]][(r["source"], r["deployment"])] += 1
    return {sp: deps for sp, deps in by_sp.items()
            if sum(deps.values()) >= k_max + q_min
            and sum(deps.values()) - max(deps.values()) >= q_min}
