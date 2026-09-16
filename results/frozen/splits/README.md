# results/frozen/splits — the locked image splits

Built once by `tools/build_frozen_splits.py`, committed, never rebuilt. Each
file carries a `sha256` over its own canonical serialization; every record
written by `saga/frozen/runner.py` cites that sha, so a split that changed
would invalidate every number keyed to it — which is why the builder refuses
to overwrite one.

| file | n | drawn from | used for |
|---|---|---|---|
| (discovery) `results/diagsplit/val_diag_split.json` | 10,000 | val | the EXISTING diagnostic split; unchanged, its sha recorded here |
| `calibration.json` | 2,000 (2/class) | val minus discovery | thresholds, empirical frequencies, label-independent setup |
| `evaluation.json` | 10,000 (10/class) | val minus discovery minus calibration | every new frozen-model intervention |
| `sub1k.json` | 1,000 | evaluation | SVD / full attention extraction |
| `sub2k.json` | 2,000 | evaluation | correspondence |

`calibration` and `evaluation` are disjoint from the discovery split and from
each other BY CONSTRUCTION — an image claimed by an earlier split is never a
candidate for a later one. `sub1k` and `sub2k` are nested INSIDE `evaluation`
on purpose, and are checked to be subsets of it.

## What this protocol is, and is not

The original full-validation accuracies have already been seen, so this is a LOCKED EVALUATION PROTOCOL FOR THE NEW ANALYSES — not a completely untouched test set, and not a retrospectively preregistered study.

The discovery split and the current maps have already informed the
hypotheses, and the 50k accuracies have already been seen. What the lock
buys is that the split IDs and the analysis configuration
(`docs/LOCKED_ANALYSIS.md`) were fixed before the new interventions' outcomes
were inspected — not that the data is untouched.
