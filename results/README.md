# results/ — the results contract

Every experimental number in the paper is read from a file in this tree.
Nothing in here is ever edited by hand. Large binaries (checkpoints, raw
attention dumps, norm dumps) are git-ignored; small artifacts (JSON, CSV,
gate files `phi_e*.npz`, `figures_data/`) are committed.

```
results/
  runs/<run_id>/            # run_id = <exp>_<arch>_<recipe>_<variant>_s<seed>
    meta.json               # written by saga/run_registry.py
    config.resolved.yaml
    log.csv                 # one row per epoch (training tasks; later)
    ckpt/                   # *.pth — git-ignored
    gates/phi_e###.npz      # SAGA gate params per epoch (small; committed)
    diag/diag_e###.json     # periodic diagnostics (later tasks)
    eval/*.json             # full-val evaluations
    attn/*.npz              # probe attention dumps — git-ignored
  legacy/                   # everything derived from pre-fix checkpoints
    checkpoint_manifest.csv
    eval/   diag/   attn/
  diagsplit/val_diag_split.json   # frozen 10k diagnostic images (committed)
  probe/probe_set.json            # frozen probe images (committed, later task)
  tables/*.csv              # generated tables — the paper reads only these
  figures_data/*            # one file per paper figure
```
