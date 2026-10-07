"""Macro recall at each taxonomic level, constrained and unconstrained.

Every level is read off ONE classifier over species (decided 5 October
2026): the predicted label at a level is that level's ancestor of the
best-scoring candidate species. A deployed classifier works at species, so
this is how its genus and family calls would actually arise. No pooled genus
or family prototypes.

  level    constrained (parent known)              unconstrained
  species  candidates = species of the query's     all evaluable species
           genus
  genus    candidates = species of the query's     all evaluable species,
           family; answer = genus of the best      answer = genus of the best
  family   candidates = species of the query's     all evaluable species,
           order; answer = family of the best      answer = family of the best

A constrained task is only scored for queries whose parent holds at least two
children among the evaluable species (a genus with two species, a family
with two genera, an order with two families) -- otherwise it is trivial.
Queries whose lineage lacks the rank are left out of that level's tasks.

Macro recall = mean over the true classes AT THAT LEVEL of per-class recall.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

LEVELS = [("species", "genus"), ("genus", "family"), ("family", "order")]


class Taxonomy:
    """Lineage lookups for the evaluable species of one pool."""

    def __init__(self, species, lineage):
        """species: ordered list; lineage: species -> {genus, family, order}"""
        self.species = list(species)
        self.lin = {s: {"species": s, **lineage[s]} for s in self.species}
        self.idx = {s: i for i, s in enumerate(self.species)}
        # children per parent, counted over evaluable species only
        self.kids = {}
        for lv, parent in LEVELS:
            k = defaultdict(set)
            for s in self.species:
                if self.lin[s][lv] and self.lin[s][parent]:
                    k[self.lin[s][parent]].add(self.lin[s][lv])
            self.kids[lv] = k

    def label_array(self, level):
        return np.array([self.lin[s][level] for s in self.species],
                        dtype=object)


def macro(true, pred):
    per = defaultdict(list)
    for t, p in zip(true, pred):
        per[t].append(t == p)
    if not per:
        return float("nan"), 0
    return float(np.mean([np.mean(v) for v in per.values()])), len(per)


def score_all(scores, query_species, tax):
    """scores [nq, n_species]; query_species: list of species names.
    -> list of dicts: level, mode, macro_recall, n_classes, n_queries"""
    out = []
    best_all = scores.argmax(1)
    for lv, parent in LEVELS:
        lab = tax.label_array(lv)
        par = tax.label_array(parent)
        q_lab = np.array([tax.lin[s][lv] for s in query_species], dtype=object)
        q_par = np.array([tax.lin[s][parent] for s in query_species],
                         dtype=object)
        # unconstrained: best over all species, read at this level
        ok = q_lab != ""
        r, n = macro(q_lab[ok], lab[best_all][ok])
        out.append({"level": lv, "mode": "unconstrained", "macro_recall": r,
                    "n_classes": n, "n_queries": int(ok.sum())})
        # constrained: candidates share the query's parent
        elig = np.array([bool(a) and bool(p) and len(tax.kids[lv][p]) >= 2
                         for a, p in zip(q_lab, q_par)])
        if elig.any():
            s = scores[elig].copy()
            s[par[None, :] != q_par[elig][:, None]] = -np.inf
            r, n = macro(q_lab[elig], lab[s.argmax(1)])
        else:
            r, n = float("nan"), 0
        out.append({"level": lv, "mode": "constrained", "macro_recall": r,
                    "n_classes": n, "n_queries": int(elig.sum())})
    return out
