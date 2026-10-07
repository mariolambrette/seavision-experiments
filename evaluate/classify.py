"""Classifiers. Both return a score matrix [queries, species]; everything
about taxonomic levels happens afterwards, in metrics.py, from that matrix.

Nearest prototype (the primary rule): features L2-normalised, a species'
prototype is the normalised mean of its normalised references, score =
cosine. At k = 1 the prototype is the single reference.

Linear probe (the decision-rule stability check, plan section 5.4: "must be
implemented properly"): multinomial logistic regression on L2-normalised
features, initialised from the prototypes, L2 penalty fixed in advance (not
tuned on the queries), full-batch L-BFGS. It answers one question in WP8 --
does the readout ranking survive a change of decision rule -- so it is run at
k = 5 and 20 only (at k = 1 it collapses to the prototype).
"""
from __future__ import annotations

import numpy as np
import torch

PROBE_L2 = 1e-3          # fixed before any score was seen
PROBE_SCALE = 10.0       # logit scale for the prototype initialisation
PROBE_ITERS = 100


def _t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32,
                           device=dev)


def normalise(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def prototypes(ref, ref_y, n_classes):
    """ref [n, d] (normalised), ref_y [n] ints -> [n_classes, d] normalised"""
    p = torch.zeros(n_classes, ref.shape[1], device=ref.device)
    p.index_add_(0, ref_y, ref)
    return normalise(p)


def prototype_scores(ref, ref_y, query, n_classes, dev):
    r, q = normalise(_t(ref, dev)), normalise(_t(query, dev))
    y = torch.as_tensor(ref_y, device=dev)
    return (q @ prototypes(r, y, n_classes).T).cpu().numpy()


def probe_scores(ref, ref_y, query, n_classes, dev):
    r, q = normalise(_t(ref, dev)), normalise(_t(query, dev))
    y = torch.as_tensor(ref_y, device=dev)
    w = (PROBE_SCALE * prototypes(r, y, n_classes)).clone().requires_grad_()
    b = torch.zeros(n_classes, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=PROBE_ITERS,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(r @ w.T + b, y) + \
            PROBE_L2 * (w * w).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        return (q @ w.T + b).cpu().numpy()
