# results/frozen — the frozen-model analyses (I0-I7)

Everything the ICLR-2027 inference-only work packages produce lives here,
kept SEPARATE from the historical results so no new number can be mistaken
for one that was already reported, and no historical file is ever rewritten.

```
results/frozen/
  README.md                 <- this file
  I0_manifest/              manifest.csv, manifest.json, eligibility.md,
                            ckpt_hashes.json, smoke_<run_id>.json
  splits/                   calibration.json, evaluation.json, sub1k.json,
                            sub2k.json, README.md
  I1_spatial/  I2_terminal/  I3_gate_edits/  I4_perturbation/
  I5_readout/  I6_external/  I7_attention/
      <run_id>/records.{parquet|csv}, diag.{parquet|csv},
                run_meta.json, records.done.json
figures_data/frozen/        compact npz for plotting
```

## The stage names

Four, and no fifth anywhere in the project (`saga/frozen/stages.py`):

| stage | what it is |
|---|---|
| `s11_out` | output of the second-to-last block (block index 10 on the 12-block production models) |
| `s12_pre_norm` | output of the LAST block, **before** the final LayerNorm |
| `s12_post_norm` | after the final LayerNorm (`model.norm`) |
| `hist` | alias for whichever of the above the HISTORICAL diagnostics used |

`hist` currently resolves to **`s12_pre_norm`**. That was read out of the
code, not assumed: `saga/metrics.py:287-291` hooks each `model.blocks[i]` and
stores the block's own output, `saga/metrics.py:324-325` takes `feats[L-1]`
and drops the prefix rows, and `saga/vit.py:238-241` applies `self.norm`
only after the block loop — so nothing a block hook sees has passed through
the final norm. `tests/test_I0_frozen.py` pins the alias against the real
`compute_diagnostics` hook.

Patch rows are always `x[:, P:, :]` with `P = infer_num_prefix_tokens(model)`
— 1 for baseline/SAGA, 5 for the timm 4-register models. Never a hard-coded
1 (audit bug B2).

## The split protocol

The original full-validation accuracies have already been seen, so the new
splits are a **locked evaluation protocol for the new analyses** — not a
completely untouched test set, and not a retrospectively preregistered study.
See `splits/README.md` for the four files and how they are kept disjoint by
construction.

## What is committed here

Committed: `*.json`, `*.csv`, `*.md`, and the compact `figures_data/frozen/*.npz`.

Git-ignored and left on the HPC beside `_norms.npz`: per-condition logit
dumps, descriptors, raw attention — anything over ~100 MB. The full contract
is in `results/README.md`.

## Rules that apply to every file under this directory

- Nothing here overwrites a historical result, note, table or threshold. New
  keys only.
- Every record row carries its checkpoint sha256, split sha256, stage,
  prefix count, precision and git sha. A number that cannot be traced back to
  a checkpoint and a split is not a number this project reports.
- `MISSING` is a value. It is never averaged and never replaced by a guess.
- Writers are append-safe: a resubmitted job adds what is missing and
  rewrites nothing, and the completion marker (`*.done.json`) is written last.
