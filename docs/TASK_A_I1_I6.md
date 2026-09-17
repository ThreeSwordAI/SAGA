# TASK A — I1 (spatial structure, scale null, recipe contrast, D5 block‑input maps) then I6 (external public checkpoints)

**Intended repo path:** `docs/TASK_A_I1_I6.md`
**Written:** 2026‑09‑17 by Planning Claude. **Executor:** Claude Code, worktree `../SAGA-A`, branch `task/A`, commit tags `[I1]` and `[I6]`. One task file, two work packages, executed in order with a **checkpoint** between them (commit, log entry, report); the human may split them across two sessions.
**Depends on:** I0 and I2 merged to `main`. D1 = `s11_out` (signed). D2, D3, D4, D6, D7, D8 closed. **D5 is open and this task closes it.**
**Phases:** A — I1 code + CPU analyses on I2's existing maps + tests → checkpoint → A′ — I6 registry, adapter, download tool, job → stop. B — HPC (the human): I1 extraction on discovery and evaluation; I6 download + runs. C — local: tables, D5 masks and note, figure data, handoff; then the human closes D5 and freezes `LOCKED_ANALYSIS.md`.
**Hard constraints:** inherit I0 handoff §8 and the I2 rules. No training, no optimizer, no probe fitting. Nothing is selected on the evaluation split. No historical file is edited. Claude Code never pushes.

---

## 0. What Track A has to settle

I1: *Does lower occurrence mean weaker spatial structure, or just a sparser estimate of the same structure?* And, now that D1 is `s11_out`: *does the spatial structure measured for four years at the last block exist one block earlier, and at the inputs to blocks 7 and 8 where I3 and I4 act?* I1 also closes D5, the last open blocker in the programme: the prevalence masks I4 perturbs.

I6: *Do public models we did not train show the same structure, and does its presence follow the training recipe as our controlled cell suggests?* The prediction is registered in §9 before any run.

**Consume, do not recompute.** I2 already produced, on the evaluation split, exceedance frequency maps at `s11_out`, `hist` (native) and `hist/term_1.00` for all 19 runs under both `tau_cal` and MAD, in `results/frozen/I2_terminal/evaluation/<run_id>/maps.npz`, with `thresholds_cal.json` beside them. Seed stability, ring profiles, recipe contrast, concentration excess, ring‑adjusted residuals and the SAGA‑vs‑baseline comparison at the new stage are CPU work on those files and start in Phase A. New inference in I1 is only what I2 did not store: **per‑token norms** (needed for the scale‑only null and the image‑level statistics) and the **block‑input stages** (needed for D5).

## 1. Deliverables

| ID | Deliverable | Location | Phase |
|---|---|---|---|
| A1 | Prevalence‑map contract module: schema, loader, validation, top‑k mask, ring‑matched control masks, conversion to/from the `_addr.json` schema so `analysis/address_analysis.py` functions run unchanged | `saga/frozen/prevalence.py` | A |
| A2 | Two new stage hooks, additive: `in_b07`, `in_b08` = residual stream entering `blocks[7]` / `blocks[8]` (0‑based; = outputs of `blocks[6]` / `blocks[7]`; paper blocks 8 and 9 under the LOCKED numbering) | `saga/frozen/stages.py` | A |
| A3 | Norm extraction condition + writer: per‑image per‑token L2 norms at `s11_out`, `in_b07`, `in_b08`, `hist` (native; plus `hist/term_1.00` for SAGA) → `norms_<stage>.npz` (HPC‑only, git‑ignored) and `maps_<stage>.npz` (committed: frequency map + `np.packbits` per‑image indicators under `tau_cal` and MAD) | `configs/frozen/I1_spatial.yaml`, `saga/frozen/norms.py` | A code, B run |
| A4 | Per‑stage thresholds for the new stages, canon recipe, new keys, per split | `results/frozen/I1_spatial/<split>/thresholds_cal.json` | B |
| A5 | CPU analyses on I2's evaluation maps (start now) and on I1's maps (after B) | `analysis/frozen_I1_spatial.py` → `results/frozen/I1_spatial/tables/T_I1a–f*.csv` | A (I2 maps), C (all) |
| A6 | D5: primary masks at `in_b07` and `in_b08` from the **discovery** split, 10 ring‑matched control masks each, sha256, generated note | `configs/frozen/I4_masks.json`, `results/frozen/I1_spatial/D5_note.md` | C |
| A7 | Figure data: F3 (spatial structure) final; F1 panels A/B draft | `figures_data/frozen/F3_spatial.npz`, `F1_prevalence_draft.npz` | C |
| A8 | I6 model registry with the pre‑registered prediction, timm adapter, download tool, conditions, job | `configs/frozen/I6_models.yaml`, `saga/frozen/external.py`, `tools/frozen_I6_download.py`, `configs/frozen/I6_external.yaml`, `scripts/jobs/frozen_I6.sbatch` | A′ |
| A9 | I6 tables and figure data | `results/frozen/I6_external/tables/T_I6a*.csv`, `figures_data/frozen/F3_external.npz` | C |
| A10 | Job file for I1 (split is an argument) | `scripts/jobs/frozen_I1.sbatch` | A |
| A11 | Tests | `tests/test_I1_spatial.py`, `tests/test_I6_external.py` | A, A′ |
| A12 | Log entries; handoff generator | `docs/TASK_LOG.md`; `analysis/build_A_handoff.py` → `docs/A_HANDOFF.md` | each phase |

## 2. Cohort and splits

- **I1 runs:** the 19 eligible 300‑epoch conditions (8 baseline, 8 SAGA, 3 registers), from the manifest, `ckpt_kind = last`, sha verified before load.
- **Selection data (D5 only):** the **discovery** split `results/diagsplit/val_diag_split.json` (10,000 images; its sha recorded). Masks are built from it and from nothing else.
- **Reporting data:** the **evaluation** split (10,000). Every table the paper shows comes from it. The word "discovery" never appears in a reporting table except as the source of a mask.
- The two splits are disjoint by construction (I0 tests). A test in I1 asserts that no mask‑building function is ever called on an evaluation‑split map.

## 3. Stages

| stage | definition | why I1 needs it |
|---|---|---|
| `s11_out` | output of `blocks[10]` | D1: the reporting stage for every new patch diagnostic |
| `hist` (= `s12_pre_norm`) | last block output, before final norm | control, always beside `s11_out`; for SAGA also under `term_1.00` |
| `in_b07`, `in_b08` | residual stream entering `blocks[7]`, `blocks[8]` | D5: the maps I4 perturbs are defined at the block input, where the perturbation acts |

`s12_post_norm` is not extracted (I2 §5: zero exceedance mass). A test pins `in_b07 == output of blocks[6]` and `in_b08 == output of blocks[7]` on a fake model, so the off‑by‑one that plagues block numbering cannot happen silently; the docstring states both conventions.

## 4. I1 conditions (`configs/frozen/I1_spatial.yaml`)

| condition_id | applies to | stages captured | writes |
|---|---|---|---|
| `native` | all 19 | `s11_out`, `in_b07`, `in_b08`, `hist` | norms + maps |
| `term_1.00` | SAGA (8) | `hist` only | norms + maps (the bypass control) |

Precision fp32, deterministic, image order by `image_id`, batch size from the job file only. `norms_<stage>.npz` per run: `float32 [n_images, N_patches]`, image ids, stage, split sha, run_id, ckpt sha, git sha — **written to the results tree on the cluster and git‑ignored** (10,000 × 196 × 4 B ≈ 8 MB per stage per run; ~1 GB total across both splits; it stays beside `_norms.npz`). `maps_<stage>.npz` per run: `freq_fixed_cal [N]`, `freq_mad [N]`, `ind_fixed_cal` and `ind_mad` as `packbits` over `[n_images, N]` (≈ 245 kB each), counts, thresholds used, provenance — **committed**.

## 5. Thresholds

`tau_cal[cell][stage]` for `in_b07`, `in_b08` (and, for completeness, `s11_out`, `hist`) is the canon recipe (`tools/compute_fixed_thr.per_image_mad_thresholds`, median over images of the per‑image MAD threshold) run on each cell's canon‑designated baseline **on the split being processed**. Discovery thresholds are used only for D5; evaluation thresholds for reporting. Written to `results/frozen/I1_spatial/<split>/thresholds_cal.json` with baseline run_ids, shas, split sha. `fixed_thresholds_canon.json` is never opened for writing (sha pinned, as in I2). At `s11_out`/`hist` on evaluation the values must match I2's `thresholds_cal.json` to numerical tolerance — a test compares them once both exist.

## 6. The contract module (`saga/frozen/prevalence.py`)

- `PrevalenceMap` dataclass: `freq [N]`, `n_images`, `n_exceed_total`, `basis ∈ {fixed_cal, mad}`, `threshold`, `stage`, `split_sha`, `run_id`, `ckpt_sha256`, `grid_side`, `n_prefix`, `git_sha`. `save/load` to npz; `validate()` refuses NaN, wrong length, `freq` outside [0,1], missing provenance.
- `to_addr_json(map) / from_addr_json(path)`: round trip to the `_addr.json` schema so `analysis/address_analysis.py` (ring profile, concentration excess with its finite‑sample reference, spatial Spearman with the 1568‑transform permutation reference) is **imported and reused**, never re‑derived. If `ring_indices`/`border_distance_map` hard‑code side 14, add an optional `side` argument additively (needed for I6's 16×16 grids) with a test that side 14 is unchanged.
- `topk_mask(map, k=16) -> sorted coordinate list`; `ring_matched_controls(mask, n=10, seeds=200..209)`: random masks with identical per‑ring counts, drawn from all positions in each ring; overlap with the primary mask is allowed, recorded per control, and never used to reject a draw; a test asserts ring composition equality and determinism.
  > **Amended 2026‑09‑17 by Mahfuzur Rahman Chowdhury.** As written, this clause said "drawn from positions outside the primary mask", which contradicts `docs/LOCKED_ANALYSIS.md` §5 — the signed document — and would have enlarged the primary-vs-control contrast by construction, which is exactly what §5 forbids. LOCKED §5 stands; this task file is wrong and is corrected here. `configs/frozen/I4_masks.json` and `results/frozen/I1_spatial/D5_note.md` record the observed overlap per control and, per mask, the analytic expectation Σ_r n_r²/N_r. No primary-excluded set is built.
- `cell_mean_map(maps)`: mean of `freq` over a cell's baseline repeats (equal weight per checkpoint).

## 7. I1 analyses (`analysis/frozen_I1_spatial.py`) and tables

All spatial correlations carry the permutation reference p (assumptions stated: dihedral × torus‑roll exchangeability, which bordered maps only approximately satisfy — say so in the note). All concentration statistics carry the finite‑sample reference at the map's own mass and report the **excess**. Image‑level statistics use `ind_*` packbits and image‑level bootstrap (10,000 resamples, seed 0 = D6).

| table | content | data |
|---|---|---|
| `T_I1a_prevalence` | per run × stage × basis: mean count/image, entropy ratio, excess Gini, top‑5 share × uniform, 7‑ring profile, ring‑1 share, ring‑0/ring‑1/centre frequencies; **absolute** map and **conditional** map `P(p exceeds | image has ≥1)` and **share** map `freq/Σfreq`; split‑half reliability ρ (fixed halves of the evaluation images, seed 0) | I2 maps (`s11_out`, `hist`, `hist/term_1.00`) now; I1 maps (`in_b07`, `in_b08`) after B |
| `T_I1b_scale_null` | per SAGA–baseline pair, stage `s11_out` and `hist`: the scalar `c` that rescales the baseline norm field so its fixed‑τ count equals SAGA's; ring‑1 share and full ring profile of the rescaled baseline map; ρ(rescaled baseline, SAGA) vs ρ(baseline, SAGA) vs ρ(baseline, other baseline seed), each with permutation p; MAD count unchanged under `c` (invariance check, must be exact) | I1 norms (after B) |
| `T_I1c_recipe` | at `s11_out` and `hist`: within‑recipe vs cross‑recipe baseline map ρ with p; ring profiles mixup vs true‑nomix with the finite‑sample reference band; excess concentration per cell; the ring‑0 < ring‑1 ordering with its bootstrap CI — the direct test of the "object‑centric framing" explanation on identical images | I2 maps now; I1 after B |
| `T_I1d_seed_stability` | all baseline pairs per cell at `s11_out`, `in_b07`, `in_b08`, `hist`: ρ, p; **ring‑adjusted residual** ρ (subtract each map's per‑ring mean, correlate residuals, same reference) | I2 maps now; I1 after B |
| `T_I1e_eta2` | per baseline map × stage: η²_pos = Var_P f(P) / (f̄(1−f̄)) with image‑level bootstrap CI; per‑cell mean | I2 maps if `ind_*` exist there, else I1 after B |
| `T_I1f_blockinput` | `in_b07`/`in_b08`: per cell mean map on **evaluation** (reporting), ρ with the same run's `s11_out` map; SAGA vs baseline at the block input; registers; and, separately labelled **discovery/selection**, the cell mean baseline maps the masks come from | I1 after B |
| `T_I1g_variant_maps` | SAGA and registers vs baseline at `s11_out` (native) and `hist` (native and, for SAGA, `term_1.00`): ρ with p, Δ excess concentration, Δ ring‑1 share, Δ count — descriptive wording only ("residual map correlation"), no functional language | I2 maps now |

Every table carries the I0 provenance columns; every generated note is pinned by a byte‑identity test. `MISSING` is never averaged.

## 8. D5 closure (Phase C)

From the **discovery** norms and thresholds: `cell_mean_map` over the four ViT‑S/mixup baselines at `in_b07` and at `in_b08`, `fixed_cal` basis (D4 = canon basis). `topk_mask(k=16)` for each → the two primary masks. `ring_matched_controls(n=10)` for each. Write `configs/frozen/I4_masks.json` with: coordinates (row, col and flat index), the cell mean frequency at each coordinate, ring composition, control masks, source run_ids and shas, discovery split sha, thresholds used, git sha, sha256 of the file itself. `D5_note.md` (generated) lists the masks, their ring composition, how concentrated the mean map is at the block input (excess vs reference), and the ρ between the `in_b07` mask source map and the `s11_out` reporting map on **evaluation** — so the reader knows the perturbed coordinates are the same address the paper reports, or knows by how much they are not. The human then closes D5 in `LOCKED_ANALYSIS.md` with the file sha and freezes the document.

## 9. I6 — external public checkpoints

**Pre‑registered prediction (written before any external forward pass; goes into the note verbatim):**
> Supervised ImageNet‑1k models whose recipe includes MixUp and CutMix (DeiT‑S, DeiT3‑S, DeiT3‑B, AugReg ViT‑B/16) will show, at the second‑to‑last block output and at the last block output, a prevalence map with positive concentration excess and a ring profile peaking at ring 1 rather than ring 0. Models trained without mixing (DINOv2‑S/B, OpenCLIP ViT‑B/16, MAE ViT‑B/16 encoder) will not show a ring‑1 peak, whatever other structure they show. Register variants (DINOv2 reg4) are exploratory and make no prediction. Either outcome is reported.

**Registry (`configs/frozen/I6_models.yaml`)**, one entry per model: timm name, hub id, expected weight sha256 (filled by the download tool), input size 224, patch size, grid side (16 for patch‑14 models at 224, 14 for patch‑16), prefix count (read from the model at load and cross‑checked), recipe flag `mixing ∈ {yes, no}` with the paper citation, `group ∈ {supervised_mixing, no_mixing, registers_exploratory}`:
`deit_small_patch16_224.fb_in1k`, `deit3_small_patch16_224.fb_in1k`, `deit3_base_patch16_224.fb_in1k`, `vit_base_patch16_224.augreg_in1k` · `vit_small_patch14_dinov2.lvd142m`, `vit_base_patch14_dinov2.lvd142m`, `vit_base_patch16_clip_224.openai`, `vit_base_patch16_224.mae` · `vit_small_patch14_reg4_dinov2.lvd142m` (exploratory). If a name does not resolve in the installed timm, record it as `UNAVAILABLE` with the timm version; do not substitute silently.

**Adapter (`saga/frozen/external.py`):** builds the timm model, applies its own pretrained normalization (from timm's data config, recorded), exposes the analog stages `ext_s11_out` (output of `blocks[-2]`) and `ext_hist` (last block output before final norm) through the same `capture_stages` mechanism, patch rows by the model's prefix count. No SAGA wrapper, no gate. Refuses models whose token grid is not square or whose depth is not 12 unless the registry says so.

**Runs:** `tools/frozen_I6_download.py` (login node; sets `HF_HOME`/`TORCH_HOME` to a woody path recorded in the registry; writes shas). Then per model: thresholds by the canon recipe on the **calibration** split (`tau_cal[model][stage]`, new file), maps + packbits indicators on **evaluation** at both analog stages, under `fixed_cal` and MAD. Cost: one forward per image per model.

**Table `T_I6a_external`:** per model × stage × basis: count/image, entropy ratio, excess Gini, top‑5 × uniform, 7‑ring profile, ring‑0/ring‑1/centre frequencies with bootstrap CIs, ring‑1‑peak indicator, prediction group, **prediction outcome** (met / not met / not applicable), split‑half ρ. One row per model; nothing pooled across groups.

## 10. Phases and acceptance

**Phase A (I1).** A1, A2, A3 code, A5 on I2's evaluation maps (tables `T_I1a`, `T_I1c`, `T_I1d`, `T_I1g`, and `T_I1e` if `ind_*` are recoverable from I2 — if not, `T_I1e` is `PENDING B`), A10, A11 (I1 tests), log. Tests (CPU, fake models/data): stage hooks equal to block outputs; norms writer shapes and packbits round trip; `PrevalenceMap` round trip + validation + `_addr.json` round trip reproduces an existing `_addr.json` byte‑for‑byte; `ring_indices(side=16)` correct and `side=14` unchanged; `topk_mask` and ring‑matched controls; scale null: MAD count invariant under scaling (exact), fixed count monotone in `c`, matched `c` found and unique; ring‑adjusted residual is zero for a pure ring‑profile map; η² known values; split‑half determinism; guard: mask builders refuse evaluation‑split maps; table byte identity. `pytest -q` green; nothing weakened. **Checkpoint:** commit, log entry `TASK A — I1 PHASE A`, report, then continue.

**Phase A′ (I6).** A8, A11 (I6 tests: registry parse; adapter prefix/grid inference on fake models incl. 5 prefix tokens and a 16×16 grid; refusal of unknown names; download tool writes shas and refuses to run inference), log entry `TASK A — I6 PHASE A`. Print the Phase B block. **Stop.**

**Phase B (human, HPC).** Merge `task/A` → `main`, push, pull. Run §11. Push back: `maps_*.npz`, `thresholds_cal.json` (both splits), I6 maps and thresholds; norms stay on the cluster.

**Phase C (next session).** All tables from evaluation; D5 masks from discovery; `D5_note.md`; F3 + F1 draft data; `T_I6a`; `build_A_handoff.py` → `docs/A_HANDOFF.md`; log. Report the D5 sha to the human. **Stop.** The human closes D5 and freezes `LOCKED_ANALYSIS.md`.

## 11. HPC block (for the human)

```bash
source classification/scripts/env_alex.sh          # before any set -u
python tools/frozen_manifest_hashes.py --verify
# I1: norms + maps at s11_out, in_b07, in_b08, hist (+ hist/term_1.00 for SAGA); 19 runs; split is the argument
sbatch --partition=a100 --gres=gpu:a100:1 --exclude=a0801 --array=0-18 scripts/jobs/frozen_I1.sbatch \
       results/diagsplit/val_diag_split.json          # discovery — D5 selection only
sbatch --partition=a100 --gres=gpu:a100:1 --exclude=a0801 --array=0-18 scripts/jobs/frozen_I1.sbatch \
       results/frozen/splits/evaluation.json          # reporting
# I6: download on the login node (network), then run
python tools/frozen_I6_download.py                    # writes weight shas into configs/frozen/I6_models.yaml
sbatch --partition=a100 --gres=gpu:a100:1 --exclude=a0801 --array=0-8 scripts/jobs/frozen_I6.sbatch
bash scripts/sync_results.sh
```

Job files read run_ids/roots from the manifest via `tools/derive_runs.ckpt_dir_for()`, verify shas before load, write split shas into every artifact, write the completion marker last; resubmission adds what is missing and rewrites nothing. Add `results/frozen/**/norms_*.npz` to `.gitignore` in Phase A.

## 12. What not to do

- Do not recompute maps that I2 already produced on evaluation; load them.
- Do not build, inspect, or rank any mask on evaluation‑split data; masks come from discovery only.
- Do not describe residual‑map correlations in functional language ("empties", "relocates", "sink function"); descriptive only.
- Do not pool across cells or across I6 groups.
- Do not touch `analysis/address_analysis.py` except to add the optional `side` argument; do not re‑derive its references.
- Do not substitute an unavailable public model with a different one.
- One worktree; new files only, WP‑prefixed; additive edits to shared files; no push.

## 13. Log and handoff

Log entries: `TASK A — I1 PHASE A`, `TASK A — I6 PHASE A`, `TASK A — PHASE C`. `docs/A_HANDOFF.md` is generated: the I6 prediction copied verbatim from the registry with its outcome per model; D5 masks with sha; every table's headline row; what Track A does not settle (I1 is descriptive; the scale null tests one alternative explanation, not all; external models differ in data, resolution and depth, so I6 tests presence of the pattern, not its cause).
