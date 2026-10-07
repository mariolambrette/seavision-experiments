"""SeaVision evaluation: one shared implementation of labels, reference/query
draws, classifiers and metrics.

Written for the WP8 readout sweep and meant to be reused unchanged by later
packages (WP13 filter rule, WP14 decision rules, WP17 thresholds), so the
methods cannot drift between them. Each module does one thing:

  labels    key -> source, deployment, species and its lineage (built once)
  pools     which species are evaluable, per pool of sources
  draws     reference/query splits by deployment, fixed by seed, saved once
            and shared by every backbone and readout (paired comparisons)
  readouts  turn an extraction run's arrays into feature matrices
  classify  nearest prototype, and a linear probe
  metrics   macro recall per taxonomic level, constrained and unconstrained
  leakage   the deployment-leakage diagnostic
  run_eval  the driver: one long results table per extraction run
"""
