# TASK 10 handoff — Test-time registers as a baseline

**Status: COMPLETE (Phases A / B / C). Nothing pending from the HPC.**
Written 2026-09-15. Every number here is read from a committed file under
`results/`; none is typed from memory. The file that holds each number is
named in §8.

---

## 1. Why this task existed

Jiang, Dravid, Efros & Gandelsman, *Vision Transformers Don't Need Trained
Registers* (NeurIPS 2025 Spotlight, arXiv:2506.08010) showed that high-norm
outlier tokens are produced by a sparse set of **register neurons** in
mid-layer MLPs, and that at inference their contribution can be redirected
into one extra untrained token — giving register-like clean attention with
**no training at all**.

Every ICLR 2027 reviewer of a sink-suppression paper will ask *"why not just
do this?"* The project had no answer. TASK 10 produces one, honestly: a
negative result was an acceptable outcome **provided the implementation was
validated first**.

---

## 2. Bottom line

> **TTR works on ViT-S/mixup** — two independent checkpoints, ~90% of sinks
> removed for ~0.2 top-1 — **but it is below SAGA on top-1 in every cell**,
> by 0.22 to 1.23 points. It **fails** on ViT-B/mixup by a margin that sits
> inside measurement uncertainty, and on ViT-S/nomix because that cell has
> almost no sinks to remove. Where TTR passes it **relocates** the sink
> address like trained registers; where it fails it **preserves** it like
> SAGA.

That is a defensible answer to the reviewer question. It is not a tie, and
the one methodological choice made after seeing data is disclosed in §7
rather than buried.

---

## 3. What was built

### Implementation — NOT vendored, and that is the headline provenance fact

The authors' repository is reachable but **carries no license of its own**.
The GitHub API reports `license: null`, and the only LICENSE in a recursive
tree listing at the pinned commit is `dinov2/LICENSE` — Meta's, covering
*their* vendored DINOv2 subtree, not the authors' `shared/`,
`custom_model/`, `clip/clip_*` or `register_neurons.ipynb`. TASK-10 A1's
instruction to "vendor it **with its LICENSE file intact**" was therefore
unsatisfiable.

**Reconciliation taken:** their code was READ at pinned commit
`860df43515c8d8e9e90952af25a46c26e4469570` (2025-09-19) so the method is
exact — no guessing which variant was built — and **no line of it was
copied**. `third_party/ttr/PROVENANCE.md` records the licensing finding, the
method as their code defines it (with their file and function names), and a
table of **all eight deviations**.

The two most consequential deviations:

| | theirs | ours | why |
|---|---|---|---|
| token placement | appended, `[CLS][PATCH][REG]` | **prefix**, `[CLS][EXTRA][PATCH]` | A2.2 requires patch count and raster order untouched so `metrics.py` / `diagnose.py` / `sink_address.py` work unchanged. Equivalent by permutation-equivariance; a test measures it at max abs diff ~2e-6 on tokens of norm ~16 |
| outlier detection | whole token sequence, incl. CLS | **patch tokens only** | every τ in this repo is calibrated on `last_block_patch_norms`; thresholding CLS with a patch-calibrated τ is a category error |

### Code

| file | role |
|---|---|
| `saga/ttr.py` | `find_register_neurons` (authors' criterion default; the task's contrast variant selectable) and `apply_ttr` |
| `tools/ttr_validate.py` | the A3 gate — sweep, sink + top-1, PASS/FAIL, exit 0 / 3 |
| `tools/ttr_prepare_run.py` | creates `results/runs/ttr_<id>/` carrying the **base** cell's identity, which is what lets the existing canon-τ backfill resolve it |
| `tools/ttr_derive.py` | full 50k eval + canonical diagnostics of the patched model |
| `tools/ttr_paired_ci.py` | discordant counts, paired bootstrap, McNemar exact + mid-p |
| `analysis/build_ttr_tables.py` | → `results/tables/T_ttr.csv`, `T_ttr_sweep.csv` |
| `analysis/build_ttr_note.py` | → `results/notes/ttr_baseline.md` (generated) |
| `scripts/jobs/ttr_*.sbatch` | four launchers: gate, matrix, paired CI, paired CI for the PASS cells |
| `tests/test_task10_ttr.py`, `tests/test_task10_phasec.py` | the task's tests |

Two engineering points a successor should know:

- **`apply_ttr` yields a proxy, not a mutated model.** `num_prefix_tokens`
  has two readers with incompatible needs during a TTR forward:
  `SAGAViT._interpolate_pos_embed` requires 1 (it runs before the extra
  tokens exist, and its B6 tripwire correctly rejects anything else), while
  `saga.metrics.infer_num_prefix_tokens` must see `1 + n_extra`. The proxy
  satisfies both and leaves the wrapped model bit-identical.
- **`tools/eval.py` and `tools/diagnose.py` have a ZERO diff.**
  `ttr_derive.py` imports their exact pieces (`shard_counts`,
  `counts_to_metrics`, `build_val_transform`, the 50 000-image assert,
  `compute_diagnostics`), and a test pins that its unpatched path reproduces
  `tools/eval.py` to the last digit.

---

## 4. What was run, and the four failures that were not TTR

Phase B took **four attempts** to produce a verdict. **Three of the failures
were mine**, one was cluster-side. All are fixed and pinned by tests; both
landmines are documented in `docs/HPC_WORKFLOW.md`.

| # | symptom | cause |
|---|---|---|
| 1 | job never queued | a copied file named `ttr_validate copy.sbatch` — the space splits it into two arguments and sbatch opens neither |
| 2 | died during env setup, `.err` had one line | **mine**: `set -u` placed ABOVE the env sourcing. `env_alex.sh` sources `/etc/profile`, whose site scripts dereference unset variables (`debuginfod.sh` line 8, `DEBUGINFOD_URLS`), and under `set -u` bash aborts the *calling* script |
| 3 | `ModuleNotFoundError: No module named 'torch'` | cluster renamed `python/3.12-conda` → `python`, which had broken **every job in the repo**. **Plus two of mine**: `PY=$(which python)` fell back to `/usr/bin/python`, and my `import torch` preflight ran, failed, and its exit status was never tested — so the job staged 50 000 images for a run that could not work |
| 4 | — | first verdict |

For #2 the real mistake was the **choice of reference**: I copied the header
from `probe_attention.sbatch`, the only committed a40 example, which had
never completed a run. Every job file that *had* completed one (TASK-08
`ft_*`, TASK-09 `det_*`/`seg_*`/`dense_smoke_*`) already had the safe
ordering. A repo-wide test over `scripts/**/*.sbatch` now pins it.

Fixes that went beyond TASK 10 (deliberate cross-task scope, flagged at the
time, one line each to revert):

- all four LIVE `env_alex.sh` files (`scripts/`, `detection/`,
  `segmentation/`, `classification/`) now try the pinned module name, fall
  back to `module load python`, and warn loudly if neither loads;
- `scripts/jobs/probe_attention.sbatch` (TASK-11) had the identical fatal
  `set -u` line and a job submitted against it;
- `scripts/sync_results.sh` was printing an "activate the env first" hint
  naming the dead module.

Legacy launchers that also name the dead module
(`detection/scripts/e3_eval_tinyx.sh`, `evaluation/e5_lost/…`,
`evaluation/e6_finegrained/e6_*.sh`) were **deliberately left alone** —
superseded provenance that must not become accidentally runnable. A test
records that exemption.

### The scientific turning point: one neuron

The gate **FAILED at every n** over all 12 blocks. Merging the two committed
full-depth sweeps localised the entire accuracy cost to a **single neuron**:
rank 9 of the ranking is in **layer 0**, and adding it moved top-1 from
78.86 to 76.14 on the gate's 5 000-image subset — **2.72 points in one
step** — after which top-1 *plateaued* at 76.1–76.3 through n=16 while sink
removal kept improving to 78.8%.

The authors' own register neurons are mid-layer (their published DINOv2 list
is layers 12–17 of 24), so scanning from layer 0 was **this project's**
deviation. Restricting the scan to layers 3–12 is a faithfulness correction.
**No threshold was moved** — see §7 for the honest caveat about the ordering.

A bug was also caught **before** the matrix ran: `ttr_matrix.sbatch` would
have re-scanned all 12 blocks and written results labelled as the validated
configuration when they were the one that failed. `ttr_prepare_run.py` now
refuses a neurons file scanned over a different range
(`--expect-layer-range`).

---

## 5. Results

All at **n = 24 neurons, layers 3–12, 1 extra zero-initialised token, scale
1.0, patch activations zeroed**, full **50 000**-image ImageNet val, fp32,
per-cell canon τ never recalibrated on a patched model.

### 5.1 Gate verdicts

Criterion, frozen before any cell ran and never revisited:
**PASS iff some n reduces `sink_fixed_canon` by ≥ 50% AND the top-1 drop is
≤ 1.00 points.**

| cell | verdict | sweep values passing | gate best n | τ |
|---|---|---|---|---|
| ViT-S/mixup (e2r s1) | **PASS** | 3 of 5 | 24 | 20.8516 |
| ViT-S/mixup (legacy) | **PASS** | 5 of 5 | 24 | 20.8516 |
| ViT-B/mixup (e2r s1) | **FAIL** | 0 of 5 | — | 127.3125 |
| ViT-S/nomix (e2r s1) | **FAIL** | 0 of 5 | — | 22.8594 |

**The two failures fail for opposite reasons.** ViT-B clears the sink bar and
misses the accuracy bar; nomix leaves accuracy alone and cannot move sinks.

### 5.2 Headline — top-1

| cell | verdict | TTR | baseline | Δ vs baseline | 95% CI on the drop | SAGA | **Δ vs SAGA** |
|---|---|---|---|---|---|---|---|
| ViT-S/mixup (e2r s1) | PASS | 78.650 | 78.862 | −0.212 | [0.0480, 0.3760] | 79.176 (n=4) | **−0.526** |
| ViT-S/mixup (legacy) | PASS | 78.684 | 78.876 | −0.192 | [0.0300, 0.3520] | 79.176 (n=4) | **−0.492** |
| ViT-B/mixup (e2r s1) | FAIL | 76.138 | 77.114 | −0.976 | [0.8120, 1.1400] | 77.363 (n=2) | **−1.225** |
| ViT-S/nomix (e2r s1) | FAIL | 73.138 | 73.196 | −0.058 | MISSING | 73.362 (n=2) | **−0.224** |

**TTR is below SAGA in every cell.** That is the answer to the reviewer
question.

### 5.3 Sinks — absolute counts beside every percentage

Percentages across cells are **not comparable**: the cells start from very
different sink counts, so the same percentage removes a different number of
tokens. Counts are mean sink tokens per image out of 196 patch tokens.

| cell | baseline | TTR | removed | reduction | SAGA |
|---|---|---|---|---|---|
| ViT-S/mixup (e2r s1) | 19.68 | 1.91 | **17.76** | 90.3% | 8.40 |
| ViT-S/mixup (legacy) | 15.39 | 0.97 | **14.42** | 93.7% | 8.40 |
| ViT-B/mixup (e2r s1) | 11.29 | 5.05 | **6.23** | 55.2% | 4.53 |
| ViT-S/nomix (e2r s1) | 3.67 | 2.15 | **1.52** | 41.5% | 1.43 |

Read the nomix row against the top row: 41.5% of 3.67 removes **1.52
tokens**; 90.3% of 19.68 removes **17.76**.

### 5.4 The other diagnostic axes

| cell | oversmooth base → TTR (SAGA) | eff_rank base → TTR (SAGA) | CLS-norm ratio base → TTR |
|---|---|---|---|
| ViT-S/mixup (e2r s1) | 0.4729 → 0.4492 (0.3018) | 89.29 → 110.28 (103.86) | 0.684 → 0.759 |
| ViT-S/mixup (legacy) | 0.4022 → 0.3424 (0.3018) | 90.68 → 114.12 (103.86) | 0.630 → 0.769 |
| ViT-B/mixup (e2r s1) | 0.7332 → 0.7517 (0.3123) | 57.08 → 69.94 (80.94) | 0.653 → 0.660 |
| ViT-S/nomix (e2r s1) | 0.1388 → 0.1499 (0.1309) | 130.99 → 131.48 (133.23) | 1.001 → 0.985 |

TTR raises effective rank in every cell. Oversmoothing improves on the two
ViT-S/mixup cells and worsens slightly on the other two. SAGA is better on
oversmoothing everywhere.

### 5.5 Paired uncertainty on the drops

Computed from the discordant table over all 50 000 images; the marginal
top-1 values **cannot** give this (from two marginals only `b − c` is
recoverable, never `b` and `c` separately).

| cell | b (TTR broke) | c (TTR fixed) | discordant | drop | 95% CI | SE | 1.00 inside CI | McNemar p |
|---|---|---|---|---|---|---|---|---|
| ViT-S/mixup (e2r s1) | 932 | 826 | 1758 | 0.2120 | [0.0480, 0.3760] | 0.0840 | no | 1.22e-2 |
| ViT-S/mixup (legacy) | 891 | 795 | 1686 | 0.1920 | [0.0300, 0.3520] | 0.0820 | no | 2.07e-2 |
| ViT-B/mixup (e2r s1) | 1111 | 623 | 1734 | 0.9760 | [0.8120, 1.1400] | 0.0833 | **yes** | 5.33e-32 |

**Both PASS cells exclude zero and sit entirely below the 1.00 bar** — the
small cost is real rather than noise, and the pass is not itself a knife
edge. **ViT-B is the only cell whose CI straddles the bar**, so its FAIL is
a knife-edge miss inside measurement uncertainty.

**The threshold was NOT moved and the verdict stands at FAIL.** McNemar's p
tests the drop against **zero**, not against the bar; it is overwhelming for
ViT-B and says only that the accuracy cost is real, which was never in
doubt.

A second precision note: the gate decides on a 5 000-image class-balanced
subset while the headline is the full 50 000. For ViT-B they differ —
**1.0200 on the subset, 0.9760 on the full set**. The verdict stands on the
subset, which is what the frozen criterion was defined over.

### 5.6 The address result — the most interesting finding

Spearman ρ between the TTR map and that cell's OWN baseline map, under
**TASK-07's exact permutation null** (8 dihedral × 196 torus rolls, 1568
transforms, deterministic). The iid null is far too generous for these
spatially smooth maps.

| cell | verdict | ρ (canon) | p | null sd | ρ (MAD) | layers touched |
|---|---|---|---|---|---|---|
| ViT-S/mixup (e2r s1) | PASS | **−0.0930** | 0.7417 | 0.1952 | +0.1443 | 3,4,5,6,7,8,10 |
| ViT-S/mixup (legacy) | PASS | **−0.2915** | 0.1448 | 0.2035 | −0.3247 | 3,4,5,6,8,9,10,11 |
| ViT-B/mixup (e2r s1) | FAIL | **+0.9371** | 0.0006 | 0.2420 | +0.9807 | 3,4,5,6,9 |
| ViT-S/nomix (e2r s1) | FAIL | +0.7749 | 0.0006 | 0.3750 | +0.7913 | 4,10,11 |

Against TASK-07's reference values for the *trained* variants (read from
`results/tables/sink_address.csv`, where **registers relocate** and **SAGA
preserves**):

| cell | TTR | trained registers | SAGA | TTR behaves like |
|---|---|---|---|---|
| ViT-S/mixup | −0.0930 / −0.2915 | +0.4773 (n=2) | +0.8726 (n=4) | **registers — relocates** |
| ViT-B/mixup | +0.9371 | +0.0063 (n=1) | +0.7929 (n=2) | **SAGA — preserves** |

**The comparison inverts between architectures**, and **the one cell where
TTR does not clear the accuracy bar is the one where it does not move the
address**. Recorded as an observed association across two architectures, not
a demonstrated mechanism.

**No pooled address number is reported** — the cells disagree, and nomix's
map is flat by construction (TASK-07: the sink address is a border ring for
every mixup member and flat for true-nomix), so a correlation there is
between two near-uniform maps and means something different.

### 5.7 Why ViT-S/nomix fails, and what that failure is not

It leaves accuracy essentially untouched (−0.058) and still FAILS, because
sink reduction stops at 41.5%. The reason is the **denominator**: that cell
holds only **3.67** sinks per image unpatched against ViT-S/mixup's 19.68,
so there is little concentrated outlier structure for a register-neuron
intervention to move. This is consistent with TASK-07's flat-address
finding.

**This does not make it a pass.** The criterion is a relative sink
reduction, it was frozen before the runs, and this cell does not meet it.

---

## 6. What we did NOT do, and why

| not done | why |
|---|---|
| **Vendor the authors' code** | no license in their repo; vendoring all-rights-reserved code into a repo headed for publication is a different class of problem from a missing attribution. Read, not copied |
| **A denser ViT-B n-sweep to find a passing point** | explicitly rejected by the human: that is hunting for a pass, not a measurement. If ViT-B coverage is ever wanted it must be a tradeoff **curve** with the selection rule and multiplicity handling frozen in advance, and labelled as such |
| **Move the 1.00 threshold** | it was set before any cell ran, is recorded in every `validate.json`, and cannot be set from any launcher (test-asserted). ViT-B misses by 0.02 on the gate's subset; the response was to measure the uncertainty, not to move the bar |
| **A paired CI for ViT-S/nomix** | that cell fails on the **sink** bar, not the accuracy bar, so an accuracy CI would not bear on its verdict. `MISSING` in the tables with the reason stated inline |
| **Exclude the failing cells from the headline** | that would be selection on outcome. All four are reported with their own verdicts |
| **Apply TTR to SAGA-gated models** | `SpatialGate.forward` hard-codes `n_patches = N − 1`, so extra prefix tokens would be gated as patches. `apply_ttr` refuses loudly. TTR here is a baseline-model intervention |
| **Dense (detection/segmentation) TTR rows** | out of scope for TASK 10; classification only |
| **Repeats / seeds for ViT-B and nomix** | one checkpoint each; see §7 |
| **Fix the legacy launchers naming the dead conda module** | superseded provenance; making them accidentally runnable would be worse |

---

## 7. Limitations — read before quoting any number

**The scan depth was chosen after seeing the full-depth failure.** The
sequence was: gate FAILED at every n over all 12 blocks → sweeps inspected →
scan restricted to layers 3–12 → ViT-S/mixup PASSED. That ordering is stated
because a reader cannot see it in the numbers, and choosing an analysis after
seeing the outcome is what inflates apparent effects.

Three things bear on how much weight it should carry:

- **The mid-layer range is not derived from our data.** It comes from the
  authors' paper; the justification would stand if our full-depth run had
  never happened.
- **No threshold was moved.** Only which neurons were candidates changed.
- **The replication is partial, not held out.** Both ViT-S/mixup checkpoints
  pass, which is some evidence the choice is not fitted to one checkpoint —
  but the restriction was chosen while looking at one of them. A clean test
  would need a cell that played no part in selecting the range. **No such
  cell exists here, and none is claimed.**

Also:

- **n=1 cells.** ViT-B/mixup and ViT-S/nomix have a single TTR measurement
  each, no repeat, so no standard error and no significance claim.
- **The replication margin varies even where the verdict replicates** — the
  two ViT-S/mixup checkpoints qualify at 3/5 and 5/5 sweep values.
- **One address pair per cell.** The permutation null handles the spatial
  smoothness of a single pair; it says nothing about seed-to-seed variation.
- **Gate and headline measure different samples** (5 000 vs 50 000), which
  matters for ViT-B specifically.
- **Scope:** four checkpoints, two architectures, two recipes, one operating
  point per cell, classification only.

---

## 8. File map — which file holds which result

### Committed results

| path | contents |
|---|---|
| `results/tables/T_ttr.csv` | **the headline table** — 4 cells × 64 columns: top-1 and Δ vs baseline and vs SAGA, sink counts + reduction, both oversmoothing variants, eff_rank, CLS-norm ratio, address ρ with p and null sd, paired CI columns, gate verdict, τ and the never-recalibrated flag |
| `results/tables/T_ttr_sweep.csv` | the n-neurons sweep, 24 rows; `n_neurons = 0` is the unpatched reference measured the same way |
| `results/notes/ttr_baseline.md` | **the readable report**, 9 sections, fully generated — provenance, gate, headline, per-cell comparison, address analysis, the nomix explanation, the ViT-B CI, the sweep, limitations |
| `results/ttr_midlayer/<cell>/neurons.json` | the register-neuron ranking (top 256) + criterion string, τ, layer range, scan stats, checkpoint sha |
| `results/ttr_midlayer/<cell>/sweep.csv` | that cell's own gate sweep |
| `results/ttr_midlayer/<cell>/validate.json` | that cell's **gate verdict**, the frozen thresholds it was judged on, and the full per-n record |
| `results/ttr_midlayer/<cell>/paired_ci.json` | discordant counts, bootstrap CI, McNemar (3 cells; nomix by design has none) |
| `results/ttr_midlayer/<cell>/paired_ci.npz` | packed per-image correctness for both passes (~6 KB) — **lets the CI be recomputed by any method without a GPU** |
| `results/runs/ttr_<cell>/eval/eval_last.json` | full 50 000-image eval of the patched model |
| `results/runs/ttr_<cell>/diag/diag_last.json` | canonical diagnostics incl. `sink_fixed_canon` |
| `results/runs/ttr_<cell>/diag/diag_last_addr.json` | the sink-address frequency maps |
| `results/runs/ttr_<cell>/config.resolved.yaml` | the **base** cell's identity + the full TTR configuration |
| `results/ttr/e2r_vits_mixup_baseline_s1/*` | the **superseded full-depth** gate (the FAIL), kept for the record |

The four cells are `e2r_vits_mixup_baseline_s1`, `e2r_vitb_mixup_baseline_s1`,
`e2r_vits_nomix_baseline_s1`, `legacy_vits_baseline`.

> `results/ttr/` (full-depth, FAILED) and `results/ttr_midlayer/`
> (layers 3–12, the reported configuration) are **different experiments**.
> Everything in §5 comes from `ttr_midlayer/`.

### Where the comparison values come from

| quantity | source |
|---|---|
| unpatched baseline top-1 / diagnostics / address | paired by **checkpoint sha256**, never by filename — `results/runs/<cell>/{eval,diag}/` for e2r cells, `results/legacy/{eval,diag}/` for the legacy one |
| SAGA column | `results/tables/e2_pooled.csv`, rows with `kind == mean` |
| canon τ | `results/diagsplit/fixed_thresholds_canon.json` |
| registers / SAGA address reference | `results/tables/sink_address.csv`, `Q4_relocation` / `spearman_mean` / `canon` |

---

## 9. Reproducing

**LOCAL (Windows PowerShell)** — regenerate both tables and the note from
the committed results. One command per line; no `&&`.

```powershell
cd "F:\FAU\PhD\Side Quest\SAGA\SAGA_Code\SAGA"
..\.venv\Scripts\python.exe analysis\build_ttr_tables.py
..\.venv\Scripts\python.exe analysis\build_ttr_note.py
..\.venv\Scripts\python.exe -m pytest -q
```

The note regenerates identically (a test asserts it), so re-running is safe
and changes only the `*Generated ...*` timestamp line.

**HPC (bash)** — the launchers, for reference. None needs to be re-run.

```bash
sbatch --partition=a100 --gres=gpu:a100:1 scripts/jobs/ttr_validate.sbatch
N_BEST=24 sbatch --partition=a100 --gres=gpu:a100:1 scripts/jobs/ttr_matrix.sbatch
sbatch --partition=a100 --gres=gpu:a100:1 scripts/jobs/ttr_paired_ci_pass_cells.sbatch
```

`ttr_validate.sbatch` honours `N_NEURONS`, `LAYER_RANGE` and `TTR_OUT_ROOT`;
the PASS thresholds are deliberately **not** settable from any launcher.

Test suite at completion: **438 passed, 22 skipped**.

---

## 10. Commits

28 commits tagged `[TASK-10]`, all on `main`. The ones that carry content
rather than process:

| commit | what |
|---|---|
| `17ce433` | implementation + validation tool (Phase A) |
| `c707021` | repo-wide conda module rename fix |
| `1a2b1da` | full-depth gate: FAIL at every n (HPC) |
| `8b70f18` | expose `LAYER_RANGE` / `TTR_OUT_ROOT` |
| `63f34a9` | mid-layer gate: PASS (HPC) |
| `e6a2e69` | matrix runs the configuration the gate validated |
| `65d972e` | the four-cell matrix results (HPC) |
| `ba927ed` | Phase C tables + note |
| `76f896a` | paired CIs for the two PASS cells (HPC) |
| `864fcf9` | all three CIs in; headline carries them |

`docs/TASK_LOG.md` holds the full dated narrative including every failure.

---

## 11. Open items for whoever picks this up

Nothing is required. Options, in rough order of value:

1. **A held-out cell for the layer-range choice** — the one thing that would
   close §7's caveat. Any ViT-S/mixup checkpoint that played no part in
   selecting layers 3–12 would do.
2. **Seeds for ViT-B/mixup and ViT-S/nomix** — would give those deltas an
   SE and turn two n=1 rows into cells.
3. **A ViT-B tradeoff curve** — only with the selection rule and
   multiplicity handling frozen in advance, and labelled as a curve. Never
   as a search for a passing n.
4. **Dense TTR rows** (detection/segmentation) if the paper wants TTR in the
   Gate-2 comparison. Note from TASK-09: the dense rewrite saves no best-AP
   detector weights, only the JSON pair, so re-inference from the best model
   is not possible unless `best_epoch == 24`.
5. **`results/ttr/` (full-depth) could be pruned** if the superseded FAIL is
   no longer wanted on disk — but it is the evidence for §7's ordering, so
   keeping it is the safer choice.

Still outstanding from earlier tasks, unrelated: the optional ViT-B
true-nomix pair, registers seeded reruns, and the `PROJECT.md` milestone
rewrite queued by the 2026-09-05 erratum.
