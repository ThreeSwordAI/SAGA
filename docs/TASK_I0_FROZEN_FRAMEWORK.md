# TASK I0 — Cohort manifest, locked parameters, evaluation splits, and the frozen‑intervention framework

**Intended repo path:** `docs/TASK_I0_FROZEN_FRAMEWORK.md`
**Written:** 2026‑09‑16 by Planning Claude. **Executor:** Claude Code, one session, worktree `../SAGA-I0`, branch `task/I0`. Commit tag `[I0]`.
**Phases:** A — local code + tests (this session). B — HPC: checkpoint hashes, split build, 32‑image smoke (the human). C — local: manifest tables, note, handoff (next session).
**Hard constraints:** no training, no fine‑tuning, no optimizer anywhere in this task. No historical result file is edited or overwritten. No `_norms.npz`, checkpoint, or attention dump is deleted. Claude Code never pushes.

---

## 0. Why this task exists

Every later work package (I1–I7) edits a frozen forward pass and records per‑image responses, or extracts stage‑tagged patch statistics. If each package writes its own hooks, we will have three definitions of "feature stage," two of "prefix tokens," and merge conflicts in `saga/metrics.py`. I0 builds the shared layer once, pins the identity of every checkpoint the paper may use, fixes the image splits, and drafts the locked analysis parameters that every later file must reference instead of restating. Nothing scientific is concluded in I0; everything scientific downstream depends on it.

## 1. Deliverables

| ID | Deliverable | Location | Phase |
|---|---|---|---|
| D1 | Cohort manifest (machine‑readable + generated eligibility note) | `results/frozen/I0_manifest/manifest.csv`, `manifest.json`, `eligibility.md` | A (from committed JSON/YAML), completed in B (hashes) and C (tables) |
| D2 | Image splits: calibration (2,000) and locked evaluation (10,000), disjoint from the discovery split; fixed 1,000 and 2,000 sub‑subsets | `results/frozen/splits/` | A builder + tests; B build on login node |
| D3 | `docs/LOCKED_ANALYSIS.md` — DRAFT of every parameter I3/I4/I5 will use; frozen by the human before I3/I4 Phase B | `docs/` | A |
| D4 | Framework package `saga/frozen/` + driver `tools/frozen_eval.py` | `saga/frozen/{stages,edits,records,runner}.py`, `tools/frozen_eval.py` | A |
| D5 | Smoke checker + job file (nine checks, 32 images, exit 0/3) | `tools/frozen_smoke.py`, `scripts/jobs/frozen_smoke.sbatch` | A; runs in B |
| D6 | Hash tool for Phase B | `tools/frozen_manifest_hashes.py` | A; runs in B |
| D7 | Tests (CPU, fake models, fake data) | `tests/test_I0_manifest.py`, `tests/test_I0_splits.py`, `tests/test_I0_frozen.py` | A |
| D8 | `docs/TASK_LOG.md` entry; `analysis/build_I0_handoff.py` → `docs/I0_HANDOFF.md` | `docs/` | A (log), C (handoff) |

## 2. Ground rules that every file in this task obeys

1. **Evidence hierarchy.** Resolved training config and checkpoint hash > raw arrays with provenance > generated per‑run tables > aggregated tables > notes/handoffs > framing documents. A note never overrides a resolved config.
2. **`recipe_actual` comes from the resolved config, never from a directory name.** Import cell membership and the recipe erratum from `analysis/build_pooled_tables.py` (as TASK 07 did); do not redefine it. Legacy `*_nomix` directories are `mixup`. The legacy ViT‑B mixup‑dir trio is VOID (incomplete training) and must appear in the manifest with `status=invalid`, never in an eligible cohort.
3. **Seeds come from training metadata** (`meta.json` / resolved config), never from a diagnostics JSON, whose `seed` field may be an evaluation default of 0.
4. **Prefix tokens come from the model** (`saga.metrics.infer_num_prefix_tokens` or the model's own attribute): 1 for baseline/SAGA, 5 for the timm 4‑register models. Every patch‑row operation in the framework uses that count. The SAGA `SpatialGate.forward` assumption `n_patches = N − 1` is documented, not changed (SAGA checkpoints all have one prefix token).
5. **Stages are named, and the historical stage is identified, not assumed.** Read `tools/diagnose.py` and `saga/metrics.py` and record whether historical patch diagnostics were computed before or after the final LayerNorm. Label that stage `hist` in the manifest and in the framework's stage list.
6. **New metric keys are added; historical keys are never overwritten.** Anything computed at a new stage or with a new calibration gets its own key and its own provenance line.
7. **Every record row carries** checkpoint sha256, run_id, arch, recipe_actual, variant, n_prefix, stage, split sha, image_id, condition_id, precision, git sha (+ dirty flag and patch file if dirty).
8. **Edits are temporary by construction.** Every intervention runs inside a context manager that hashes the model state before and after and asserts equality on exit.
9. **Diagnostics come from the LAST checkpoint** of each run, as everywhere in the project. `*_best*` artifacts are recorded in the manifest as existing but are never eligible.

## 3. D1 — the cohort manifest

**One row per saved checkpoint or checkpoint‑derived condition.** Sources: `results/runs/*/meta.json`, `config.resolved.yaml`, `eval/*.json`, `diag/*.json`; `results/legacy/**` and its manifest with `recipe_actual`; `configs/{abl_matrix,ft_matrix,dense_matrix}.yaml`; `results/runs/ttr_*/config.resolved.yaml`; `third_party/ttr/PROVENANCE.md`.

| Column | Content |
|---|---|
| `run_id`, `family` | family ∈ {e2r_300ep, legacy_300ep, ablation_100ep, finetune, dense_det, dense_seg, ttr_edit} |
| `arch`, `variant`, `recipe_actual`, `recipe_source` | variant ∈ {baseline, saga, registers}; recipe_source = path of the resolved config used |
| `seed`, `seed_source`, `seed_controlled` | controlled = trained under the seeded e2r trainer |
| `epochs_completed`, `ckpt_kind` (`last`/`best`), `ckpt_path`, `ckpt_sha256` (filled in Phase B), `ckpt_dir_source` (`meta.json["ckpt_dir"]` when present) |
| `n_prefix`, `gate_mode`, `gate_init_logit`, `gate_params_registered`, `gate_params_trainable` |
| `hist_stage`, `hist_eval_precision`, `hist_split_sha` | what the historical numbers were computed on |
| `status` ∈ {eligible, eligible_legacy, invalid, superseded, exploratory, derived} | with `status_reason` (one line) |
| `git_sha`, `git_dirty`, `patch_file` | for dirty trees, the saved diff path if one exists; else `MISSING` |

**Facts the manifest must reproduce, pinned by tests:**
- The 300‑epoch classification cohort has 19 completed conditions: 8 baselines (4 ViT‑S mixup, 2 ViT‑S true‑nomix, 2 ViT‑B mixup), 8 matching SAGA, 3 registers (2 ViT‑S mixup, 1 ViT‑B mixup, all legacy). Fresh seeded pairs (s1/s2) are flagged `seed_controlled=1`; legacy repeats `0`.
- The 6 ablation arms (100 ep, seed 0; checkpoints under the woody `abl_ckpt` root recorded in each `meta.json`) are a separate family, never mixed with 300‑epoch rows.
- 24 fine‑tuning runs (`ft_*`), with backbone sha and `best.pth` (this family selects on val, so `best` is its canonical checkpoint — record it as such).
- Dense runs: record what weight files actually exist per run (the detection rewrite may have saved only JSON pairs); `MISSING` is a value.
- TTR rows are `derived` edits of a baseline checkpoint (base run_id, neurons file sha, layer range, n_neurons), not models.
- Prefix counts: 1 everywhere except registers (5).

**Eligibility note (generated, `eligibility.md`):** the eligible cohort per family, the excluded rows with reasons, and the historical stage. Rendered by `analysis/build_I0_manifest.py`; a test re‑renders and asserts byte identity (project convention).

## 4. D2 — image splits

- **Discovery** = the existing frozen 10,000‑image split `results/diagsplit/val_diag_split.json` (unchanged; its sha recorded).
- **Calibration** = 2,000 val images, class‑stratified (2/class), seeded, **disjoint from discovery**.
- **Evaluation (locked)** = 10,000 val images, stratified (10/class), seeded, **disjoint from both**.
- **Sub‑subsets** = fixed 1,000 (SVD / attention) and 2,000 (correspondence) drawn from evaluation.
- Builder `tools/build_frozen_splits.py`: stdlib only (runs on a login node), reads the val listing the existing split builder used, writes `results/frozen/splits/{calibration,evaluation,sub1k,sub2k}.json` with relative paths, labels, seed, counts, and a sha256 over canonical serialization. **Write‑once**: refuses to overwrite an existing file. Follow the pattern of `evaluation/e6_finegrained/tools/build_ft_split.py` / `tools/build_probe_set.py`.
- Tests: disjointness (all pairs), stratification, determinism from seed, write‑once refusal, sha stability.
- The 50k accuracies have already been seen, so the paper calls this a *locked evaluation protocol for the new analyses*, not an untouched test set. Put that sentence in the split README.

## 5. D3 — `docs/LOCKED_ANALYSIS.md` (DRAFT until the human signs and dates it)

List, with one line of justification each, the parameters below. Values in brackets are the plan's defaults; Claude Code fills them in from `docs/SAGA_ICLR2027_FINAL_PLAN.md` §6 and §5.5 and marks anything the plan leaves open as `DECISION NEEDED`.

| Item | Draft value |
|---|---|
| Stage for all new patch diagnostics | decided by I2; candidates `s11_out`, `s12_pre_norm`, `s12_post_norm`, `hist` |
| I3 layers | paper blocks 8 and 9 (0‑based indices 7, 8); selection disclosed as based on prior exploratory correlations |
| I3 variants | original; per‑head mean μ; μ + 0.5δ; 10 fixed position permutations; 10 fixed within‑ring permutations; dihedral set; energy‑matched edit (plan §5.5) |
| Permutation seeds | fixed list, shared across checkpoints on the same grid |
| I4 mask | 16 highest‑prevalence coordinates from the discovery map at the block input; 10 ring‑matched random control masks |
| I4 ε | 0.10 primary, 0.25 sensitivity; per‑image energy matching within checkpoint |
| Primary endpoint | paired per‑image NLL difference; top‑1, logit shift, norms, maps secondary |
| Units | images for frozen‑edit effects (paired; image‑level bootstrap, 10,000 resamples); checkpoints for training effects; permutation draws are repeated interventions, never seeds |
| Multiplicity | declared primary contrasts only; everything else labelled exploratory |
| Decision rule | direction consistent in both fresh ViT‑S/mixup pairs, image‑conditional CI excludes zero, magnitude interpretable |
| Splits | shas of `calibration`, `evaluation`, `sub1k`, `sub2k` |

## 6. D4 — the framework

### `saga/frozen/stages.py`
Forward hooks that return patch tokens (prefix rows excluded, using the model's prefix count) at named stages: `s11_out` (output of block index 10), `s12_pre_norm` (output of the last block, before the final norm), `s12_post_norm` (after the final norm), and `hist` (alias for whichever of these the historical diagnostics used — set once the reading of `tools/diagnose.py` is done, with a test that pins it). Also returns CLS logits from the native head. Works for `SAGAViT`, plain baseline, and the timm register model (do not wrap register models in `SAGAViT`; it refuses `num_prefix_tokens != 1` by design).

### `saga/frozen/edits.py`
Each edit is a context manager; on exit it restores state and asserts `state_hash_before == state_hash_after`.
- `terminal_gate_override(model, value)` — sets the last block's patch gate to a constant in {0.25, 0.5, 0.75, 1.0}; `1.0` is bypass. SAGA models only. (I2)
- `gate_edit(model, layer, mode, *, alpha=None, perm=None)` — `mode ∈ {original, mean, mean_plus_alpha_delta, permute, permute_within_ring, dihedral}`; μ and δ per head; permutations shared across heads; ring indices imported from `analysis/address_analysis.py` (`border_distance_map`, `ring_indices`), never re‑derived. (I3)
- `receiver_perturbation(model, layer, mask, epsilon, *, energy_target=None)` — multiplies the patch attention output at masked coordinates by `(1 − ε)` after any native gate and before the residual addition; excludes prefix rows; when `energy_target` is given, rescales so the per‑image injected update norm equals the target (plan §5.5); logs the measured norm. Works for baseline, SAGA, and register models. (I4)
- All edits refuse an unknown layer, a mask touching prefix rows, or a model whose prefix count cannot be inferred.

### `saga/frozen/records.py`
Long‑format record schema (one row per image × condition): the §2.7 provenance columns plus `condition_id`, `edit_type`, `edit_params` (JSON), `layer`, `epsilon`, `measured_perturbation_norm`, `nll`, `correct`, `top1`, `max_abs_logit_diff_vs_native`. Per‑image diagnostic summaries (patch‑norm quantiles, fixed and MAD counts under declared thresholds, cosine both variants, effective rank on `sub1k`) go to a second file keyed the same way. Writers are append‑safe and atomic; files are Parquet if available, else CSV + npz.

### `saga/frozen/runner.py` and `tools/frozen_eval.py`
Shared loop: load a manifest row by `run_id`, verify `ckpt_sha256`, build the model through `tools/model_factory.py` with the recorded `gate_mode`, load the split by sha, run the declared conditions on the declared split, write records under `results/frozen/<WP>/<run_id>/`. Deterministic inference, fp32 for logit comparisons, one image order keyed by image_id. `--conditions` are read from a small YAML per work package (I0 ships `configs/frozen/smoke.yaml` only). No condition, layer, ε, or mask is selectable from the command line except through that YAML.

## 7. D5 — smoke checks (from the plan §13.3), 32‑image fixed subset, exit 0 on all PASS, 3 otherwise

1. Native loader reproduces the recorded eval logits/accuracy on the shared subset within documented tolerance.
2. Identity edit reproduces native output bit‑for‑bit.
3. Terminal patch‑gate override leaves CLS logits unchanged (fp32 tolerance documented) for a SAGA checkpoint.
4. Prefix rows are untouched by every patch edit, on the register model in particular.
5. Gate permutations preserve per‑head means and value histograms.
6. Within‑ring permutations preserve ring counts.
7. Energy‑matched edits achieve the declared injected norm within tolerance.
8. Stage hooks return tensors of shape `[B, N_patches, d]` with `N_patches = N − n_prefix` at every stage.
9. State hash restored after every edit.

Each check writes PASS/FAIL with the measured quantity to `results/frozen/I0_manifest/smoke_<run_id>.json`. Run on three checkpoints: one ViT‑S/mixup baseline s1, its SAGA s1, and the legacy ViT‑S registers checkpoint.

## 8. Phase plan and acceptance

**Phase A (this session, local, CPU).** Implement D1–D7. Manifest built from committed files with `ckpt_sha256 = MISSING`. Tests green (`pytest -q`); do not weaken or delete existing tests. Append the TASK_LOG entry. Commit on `task/I0`. Print the Phase B block below. Stop.

Acceptance for A: manifest reproduces the pinned facts of §3; splits builder refuses overwrite and passes disjointness tests on fake listings; every edit passes its CPU test on a tiny fake ViT with a SAGA gate and on a tiny fake 4‑register model; the `hist` stage is identified with a citation of the code lines; `LOCKED_ANALYSIS.md` exists with every open item marked.

**Phase B (the human, HPC).** Merge `task/I0` into `main`, push, pull on the HPC, then run §9. Push back the small outputs (`manifest.csv` with hashes, split JSONs, smoke JSONs).

**Phase C (next session, local).** Regenerate `eligibility.md` with hashes; update tests to the final row counts; `analysis/build_I0_handoff.py` → `docs/I0_HANDOFF.md`; log entry.

## 9. HPC block (for the human; commands only, nothing scientific)

```bash
# login node, after: git pull on main
source classification/scripts/env_alex.sh        # source BEFORE any set -u (TASK 10 lesson)
python tools/build_frozen_splits.py               # stdlib; write-once; prints shas
python tools/frozen_manifest_hashes.py            # sha256 of every ckpt_path in the manifest; CPU
sbatch --partition=a100 --gres=gpu:a100:1 --exclude=a0801 scripts/jobs/frozen_smoke.sbatch
# then: bash scripts/sync_results.sh  (or the usual small-results commit) and push
```

`frozen_smoke.sbatch` header must follow a job file that has completed a run (`ft_finegrained_array.sbatch`), not `probe_attention.sbatch`'s original header. Checkpoint roots come from each run's `meta.json["ckpt_dir"]` via `tools/derive_runs.ckpt_dir_for()`; never hard‑code `<run_dir>/ckpt`.

## 10. Results layout (new, keep separate from historical results)

```
results/frozen/
  README.md                 # what lives here, the stage names, the split protocol sentence
  I0_manifest/              # manifest.csv, manifest.json, eligibility.md, smoke_*.json
  splits/                   # calibration.json, evaluation.json, sub1k.json, sub2k.json
  I1_spatial/  I2_terminal/  I3_gate_edits/  I4_perturbation/  I5_readout/  I6_external/  I7_attention/
      <run_id>/records.parquet, diag.parquet, conditions.yaml, run_meta.json
figures_data/frozen/        # compact npz for plotting (committed with -f like Faddr.npz)
```
Small derived files are committed. Anything over ~100 MB (per‑condition logits, descriptors, attention) is git‑ignored and stays on the HPC beside `_norms.npz`.

## 11. What not to do

- No training, fine‑tuning, probe fitting, or optimizer import in any new file.
- No edit to `saga/gate.py`, `saga/vit.py`, `saga/metrics.py`, `tools/eval.py`, `tools/diagnose.py` beyond adding a stage hook if unavoidable; if a change there is unavoidable, it is additive, test‑pinned to reproduce every historical number, and called out in the log.
- No renaming or rewriting of historical result files, notes, or thresholds; new keys only.
- No selection of layers, masks, ε, or permutations by looking at evaluation losses — I0 produces no evaluation losses at all beyond the smoke subset.
- No shared checkout with another session; worktree `../SAGA-I0` only.
- No push.

## 12. Log and handoff

TASK_LOG entry format (append‑only, newest last): `## 2026‑09‑DD — TASK I0, PHASE A (manifest, splits, framework, smoke)` with Done / Commits / Pending‑from‑HPC / Open decisions. `docs/I0_HANDOFF.md` is generated from the manifest and smoke JSONs by `analysis/build_I0_handoff.py` in Phase C; no number in it is typed by hand.
