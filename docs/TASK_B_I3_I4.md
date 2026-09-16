# TASK B — I3 (does the learned gate arrangement matter to the frozen model?) then I4 (do high‑prevalence coordinates respond differently to a matched perturbation?)

**Intended repo path:** `docs/TASK_B_I3_I4.md`
**Written:** 2026‑09‑17 by Planning Claude. **Executor:** Claude Code, worktree `../SAGA-B`, branch `task/B`, commit tags `[I3]` and `[I4]`. One task file, two work packages, executed in order with a **checkpoint** between them; the human may split them across sessions.
**Depends on:** I0 and I2 merged. D1 = `s11_out`, D2 (8 dihedral transforms), D3 (`configs/frozen/permutations_14x14.json`), D4 (`fixed_cal` basis), D6 (bootstrap seed 0), D7 (three named contrasts, no correction), D8 closed. **I4 additionally depends on Track A:** `saga/frozen/prevalence.py` on `main` (I1 Phase A) for its Phase A, and D5 (`configs/frozen/I4_masks.json`) plus a **frozen** `docs/LOCKED_ANALYSIS.md` for its Phase B.
**Phases:** A — I3 code, YAML, analysis, tests → checkpoint → A′ — I4 code, YAML, analysis, tests → stop. B — HPC (the human): I3 immediately after merge; **I4 only after D5 is closed and the LOCKED file is frozen.** C — local: tables, F4 data, handoff.
**Hard constraints:** inherit I0 handoff §8 and the I2/A rules. No training, no optimizer, no probe fitting. Nothing is selected on the evaluation split (masks and permutations are already fixed elsewhere). No historical file is edited. Claude Code never pushes.

---

## 0. The two questions, and the three contrasts named in advance

I3 asks whether the *spatial arrangement* of the learned gate matters to the model that learned it. A trained SAGA checkpoint is edited at inference — the gate at one block replaced by its per‑head mean, by a partially flattened version, by a permutation with the same histogram, by a ring‑preserving symmetry — and the paired per‑image NLL is measured against the unedited forward. No retraining, so the answer is about what the frozen model *uses*, not about what a scalar gate trained from scratch would achieve (that question needs Track T and is not claimed here).

I4 asks whether the coordinates where high‑norm tokens form respond differently to a perturbation than other coordinates of the same ring composition and the same injected energy — in the baseline, in SAGA, and in the register model, on the same mask. It uses the masks D5 fixed on the discovery split.

**D7 — the three primary contrasts (everything else in Track B is exploratory and labelled so):**

| id | contrast | unit | direction that would support "arrangement / coordinates matter" |
|---|---|---|---|
| **C1** | I3: `mean` collapse − `original`, per‑image NLL | images, paired; per checkpoint | ΔNLL > 0 |
| **C2** | I3: mean over the 10 `permute` conditions − mean over the 7 non‑identity `dihedral` conditions, per‑image NLL | images, paired; per checkpoint | > 0 (arrangement beyond ring/symmetry structure) |
| **C3** | I4: θ(primary mask) − mean θ(10 ring‑matched, energy‑matched controls), ε = 0.10, per‑image NLL | images, paired; per checkpoint | ≠ 0 with the same sign across checkpoints of a method |

Decision rule (from `LOCKED_ANALYSIS.md`, applied per contrast): direction consistent across the fresh ViT‑S/mixup checkpoints (`s1`, `s2`), image‑level bootstrap CI (10,000 resamples, seed 0) excludes zero in each, effect size stated in nats and in top‑1 points. Legacy checkpoints are reported alongside and do not decide. A contrast that fails is reported as a null with its CI; nothing is re‑framed.

## 1. Deliverables

| ID | Deliverable | Location | Phase |
|---|---|---|---|
| B1 | I3 conditions (the only place they are defined) | `configs/frozen/I3_gate_edits.yaml` | A |
| B2 | Runner additions, additive: native‑caching per batch; per‑image `delta_update_norm` at the edited block; per‑image `energy_target` tensor support in `receiver_perturbation` if I0 shipped a scalar only | `saga/frozen/runner.py`, `saga/frozen/edits.py` | A (B2a), A′ (B2b) |
| B3 | I3 analysis + tables | `analysis/frozen_I3_analysis.py` → `results/frozen/I3_gate_edits/tables/T_I3a–d*.csv` | A code, C run |
| B4 | Contrast + decision module for C1–C3 | `analysis/i34_contrasts.py` | A |
| B5 | I3 job file | `scripts/jobs/frozen_I3.sbatch` | A |
| B6 | I4 conditions | `configs/frozen/I4_perturbation.yaml` | A′ |
| B7 | I4 analysis + tables | `analysis/frozen_I4_analysis.py` → `results/frozen/I4_perturbation/tables/T_I4a–d*.csv` | A′ code, C run |
| B8 | I4 job file with the **freeze guard** | `scripts/jobs/frozen_I4.sbatch`, guard in `tools/frozen_eval.py` | A′ |
| B9 | Figure data F4 | `figures_data/frozen/F4_interventions.npz` | C |
| B10 | Tests | `tests/test_I3_gate_edits.py`, `tests/test_I4_perturbation.py` | A, A′ |
| B11 | Log entries; handoff generator | `docs/TASK_LOG.md`; `analysis/build_B_handoff.py` → `docs/B_HANDOFF.md` | each phase |

## 2. Cohort and split

- **I3:** the 4 eligible ViT‑S/mixup SAGA checkpoints (`ckpt_kind = last`, sha verified). Fresh pair `s1`, `s2` decide; the two legacy repeats are reported. The 2 ViT‑S/nomix and 2 ViT‑B/mixup SAGA checkpoints run the same YAML as **exploratory scope** (their gates have different spatial structure; nomix has ~1.5× the spatial std), labelled so in every table.
- **I4:** the ViT‑S/mixup cell only — 4 baseline, 4 SAGA, 2 registers = 10 checkpoints. D5's masks are defined for this cell; no other cell has masks, so no other cell is in I4.
- **Split:** `results/frozen/splits/evaluation.json` (10,000) for everything reported. Nothing runs on discovery or calibration in Track B.
- **Layers:** 0‑based `blocks[7]` and `blocks[8]` (paper blocks 8 and 9). Each condition edits **one** layer; joint edits are not in scope.

## 3. I3 conditions (`configs/frozen/I3_gate_edits.yaml`)

Per layer ℓ ∈ {7, 8}, applied through `gate_edit`, all with state restore:

| condition_id | edit | count | records `diag` at `s11_out`? |
|---|---|---|---|
| `original` | none (native forward; also the reference for every Δ) | 1 | yes |
| `mean_L{ℓ}` | per‑head spatial mean μ_h replaces σ(φ_{ℓ,h,·}) | 1 | yes |
| `mean_half_L{ℓ}` | μ_h + 0.5·δ_h | 1 | no |
| `perm{k}_L{ℓ}`, k = 0..9 | position permutation k from `permutations_14x14.json`, shared across heads | 10 | no |
| `ringperm{k}_L{ℓ}`, k = 0..9 | within‑ring permutation k from the same file | 10 | no |
| `dihedral{t}_L{ℓ}`, t = 0..7 | the 8 symmetries of the 14×14 grid applied to the gate map; t = 0 is the identity and **must be bit‑identical to `original`** (a check, recorded) | 8 | no |

Total per checkpoint: 1 + 2 × 30 = **61 forwards** of 10,000 images. All permutations and transforms are loaded from the committed file/definitions; `gate_edit` must refuse to sample (D3 test exists in `[I3-prep]`; reuse it).

**Recorded per image, per condition:** I0 record schema (`nll`, `correct`, `top1`, `max_abs_logit_diff_vs_native`) plus `delta_update_norm` = ‖u_edit − u_native‖₂ where u is the block‑ℓ attention‑branch residual update for that image's patch rows (prefix rows excluded), computed against the cached native forward of the same batch. For conditions marked `diag`, the `s11_out` per‑image summaries (`count_fixed_cal`, `count_mad`, `cos_all`, `cos_nosink_mad`, `eff_rank` on `sub1k`) using I2's evaluation `tau_cal[s11_out]`.

**Energy matching in I3 is by stratification, not by an extra forward:** ΔNLL is reported within deciles of `delta_update_norm` (10 quantile bins per checkpoint, computed on the pooled edited conditions), so `mean`, `permute`, `ringperm` and `dihedral` are compared at equal injected energy. This is a clarification of the LOCKED I3 variant list and goes into the LOCKED file with the freeze; the plan's §5.5 per‑image matching is used in I4, where a scalar ε makes it natural.

## 4. I3 tables (`analysis/frozen_I3_analysis.py`)

| table | content |
|---|---|
| `T_I3a_conditions` | per checkpoint × layer × condition: mean ΔNLL vs `original`, bootstrap CI, Δtop‑1 (points), fraction of images with ΔNLL > 0, mean/median `delta_update_norm`, `max_abs_logit_diff_vs_native` |
| `T_I3b_contrasts` | C1 and C2 per checkpoint with CI; decision‑rule outcome for the fresh pair; legacy and exploratory rows labelled |
| `T_I3c_energy_strata` | ΔNLL by `delta_update_norm` decile, per condition family (`mean`, `mean_half`, `permute`, `ringperm`, `dihedral`), per checkpoint |
| `T_I3d_diag_s11` | `s11_out` diagnostics under `original` vs `mean_L7`, `mean_L8`: paired Δ with CI (does removing the arrangement move the exceedance count or cosine one block from the readout?) |

Interpretation guide, fixed now: C1 ≈ 0 and C2 ≈ 0 → the frozen model does not use its gate arrangement beyond the per‑head mean; C1 > 0 but C2 ≈ 0 → it uses the ring/symmetry structure but not the within‑ring arrangement; C1 > 0 and C2 > 0 → it uses the specific arrangement. Each is reportable; none is a failure.

## 5. I4 conditions (`configs/frozen/I4_perturbation.yaml`)

Masks from `configs/frozen/I4_masks.json` (D5): primary mask `P_L7` at `in_b07`, `P_L8` at `in_b08` (16 coordinates each), controls `K{j}_L7`, `K{j}_L8`, j = 0..9 (ring‑matched). The runner verifies the file's sha256 against the LOCKED D5 line before loading.

Per layer ℓ ∈ {7, 8}, applied through `receiver_perturbation` (attention output at masked coordinates × (1 − ε), after any native gate, before the residual add, prefix rows excluded):

| condition_id | mask | ε | energy | count | `diag` |
|---|---|---|---|---|---|
| `native` | — | — | — | 1 | yes |
| `prim_e10_L{ℓ}` | P_Lℓ | 0.10 | fixed ε; injected norm measured per image | 1 | yes |
| `prim_e25_L{ℓ}` | P_Lℓ | 0.25 | fixed ε | 1 | no |
| `ctrl{j}_e10m_L{ℓ}` | K_j,Lℓ | — | **per‑image `energy_target` = the primary’s measured injected norm** at ε = 0.10 on the same image (plan §5.5) | 10 | no |
| `ctrl{j}_e25m_L{ℓ}` | K_j,Lℓ | — | matched to primary at 0.25 | 10 | no |
| `ctrl{j}_e10f_L{ℓ}` | K_j,Lℓ | 0.10 | fixed ε, **unmatched** (shows the size of the energy confound) | 10 | no |

Total per checkpoint: 1 + 2 × (2 + 30) = **65 forwards** of 10,000 images. Within each batch the runner executes the primary condition first, reads the per‑image injected norm, then the matched controls with that target. The achieved norm is recorded for every condition (`measured_perturbation_norm`) and a tolerance check (relative error < 1%) is written per image.

**Recorded per image:** I0 schema + `measured_perturbation_norm`, `energy_target`, `energy_rel_error`; for `diag` conditions the `s11_out` summaries as in I3.

## 6. I4 tables (`analysis/frozen_I4_analysis.py`)

| table | content |
|---|---|
| `T_I4a_theta` | per checkpoint × layer × ε × mask: θ = mean paired ΔNLL vs `native`, CI, Δtop‑1, mean injected norm, mean `energy_rel_error` |
| `T_I4b_contrast` | C3 per checkpoint: θ(primary) − mean θ(matched controls), CI; the control band (min, max of the 10 θ_control); decision‑rule outcome for the fresh baselines and the fresh SAGA checkpoints separately; unmatched controls beside them |
| `T_I4c_cross_method` | on the common mask at matched energy: θ(primary) for baseline (4), SAGA (4), registers (2); paired‑by‑seed baseline − SAGA difference where seeds match; registers n = 2, labelled |
| `T_I4d_diag_s11` | `s11_out` diagnostics under `prim_e10_L7/L8` vs `native`: does attenuating the high‑prevalence coordinates at the block input change exceedance counts or cosine one block from the readout? |

Interpretation guide, fixed now: θ(primary) inside the control band for every method → high‑prevalence coordinates are not special beyond their ring composition and injected energy; outside the band with consistent sign in baseline but not in SAGA (or the reverse) → the methods differ in how those coordinates are used, which is the cross‑method row; outside the band with the same sign in all methods → coordinate specificity is a property of the data position, not of the intervention. Each is reportable.

## 7. Guards

- **Freeze guard (I4 only):** `tools/frozen_eval.py` refuses to run any `configs/frozen/I4_*.yaml` unless `docs/LOCKED_ANALYSIS.md` contains a header line `STATUS: FROZEN` with a date and git sha, and D5 is closed with a sha256 that matches `configs/frozen/I4_masks.json`. The refusal message names what is missing. Tested with a fake LOCKED file in both states.
- **No sampling:** permutations, ring permutations, dihedral transforms and masks are loaded from committed files/definitions; any code path that draws a random spatial object at run time is a test failure (extend the I0 AST test to `numpy.random`/`torch.rand*` in `saga/frozen/` and `tools/frozen_eval.py`).
- **State restore:** every edit context manager's before/after hash equality, already in I0, is exercised in the runner integration test on a fake model for both I3 and I4 condition types.
- **Reference condition identity:** `original` (I3) and `native` (I4) are bit‑identical to a plain forward on the same batch; `dihedral0` is bit‑identical to `original`.

## 8. Phases and acceptance

**Phase A (I3).** B1, B2a, B3, B4, B5, I3 tests, log. Tests (CPU, fake models/data): YAML parse and refusal of undeclared conditions; 61 conditions per checkpoint; permutations from the committed file only; `mean` leaves per‑head mean exactly; `dihedral` transforms are bijections that preserve ring membership (test on `ring_indices`) and `dihedral0 ≡ original`; `delta_update_norm` equals a hand computation on a fake model; decile stratification determinism; C1/C2 on synthetic records for all interpretation branches; byte identity of generated notes; `pytest -q` green, nothing weakened. **Checkpoint:** commit, log `TASK B — I3 PHASE A`, report, wait for "continue with I4".

**Phase A′ (I4).** Precondition: `saga/frozen/prevalence.py` exists on `main` (Track A, I1 Phase A). If it does not, **stop and say so**; do not write a substitute. Then B2b, B6, B7, B8, I4 tests, log `TASK B — I4 PHASE A`. Tests: YAML parse; 65 conditions; masks loaded through the contract with sha verification; prefix rows untouched on a fake 5‑prefix model; per‑image `energy_target` achieved within 1% on a fake model; unmatched vs matched controls differ only in energy; freeze guard both states; C3 on synthetic records; byte identity. Print the Phase B block (with the I4 embargo line). **Stop.**

**Phase B (human, HPC).** Merge `task/B` → `main` (after Track A's Phase A if both touched `runner.py`), push, pull. Run I3 now. **Run I4 only after D5 is closed and the LOCKED file is frozen** — the guard will refuse otherwise. Push back records‑derived small files; raw parquet stays on the cluster (I2 policy) with shas in `run_meta.json`, plus a compact per‑image primary pack `figures_data/frozen/I3_primary.npz` / `I4_primary.npz` (per‑image ΔNLL for every condition in float32: I3 ≈ 61 × 10,000 × 8 checkpoints × 4 B ≈ 20 MB; I4 ≈ 65 × 10,000 × 10 × 4 B ≈ 26 MB — committed with `-f`, as `Faddr.npz` was) so every reported number regenerates from the repository.

**Phase C (next session).** `T_I3a–d`, `T_I4a–d`, C1–C3 outcomes from `i34_contrasts.py`, `F4_interventions.npz`, `build_B_handoff.py` → `docs/B_HANDOFF.md`, log `TASK B — PHASE C`. The contrasts module decides; nothing is re‑framed by inspection.

## 9. HPC block (for the human)

```bash
source classification/scripts/env_alex.sh          # before any set -u
python tools/frozen_manifest_hashes.py --verify
# I3: 4 ViT-S/mixup SAGA checkpoints (array 0-3) + 4 exploratory SAGA (array 4-7); split fixed to evaluation in the job
sbatch --partition=a100 --gres=gpu:a100:1 --exclude=a0801 --array=0-7 scripts/jobs/frozen_I3.sbatch
# I4: EMBARGOED until D5 is closed and docs/LOCKED_ANALYSIS.md is FROZEN; the guard refuses otherwise
sbatch --partition=a100 --gres=gpu:a100:1 --exclude=a0801 --array=0-9 scripts/jobs/frozen_I4.sbatch
bash scripts/sync_results.sh
```

Job files read run_ids and checkpoint roots from the manifest via `tools/derive_runs.ckpt_dir_for()`, verify shas before load, write the split sha and the masks/permutations file shas into every artifact, write the completion marker last; resubmission adds what is missing and rewrites nothing. Expected wall time: I3 ≈ 61 forwards × 10k images per checkpoint; I4 ≈ 65; ViT‑S fp32 on one A100 is well under two hours per checkpoint.

## 10. What not to do

- Do not run I4 on the cluster before the freeze; do not weaken the guard.
- Do not edit jointly at two layers, add layers, change ε, or add masks; do not draw any random spatial object at run time.
- Do not compare I3's edits to what a scalar gate would achieve if trained — that is Track T and is not claimed.
- Do not use functional language in tables or notes ("the model needs", "sink function"); report ΔNLL, top‑1, energy, and the contrasts.
- Do not run any Track B condition on discovery or calibration.
- One worktree; new files only, WP‑prefixed; additive edits to `runner.py`/`edits.py` only; no push.

## 11. Log and handoff

Log entries: `TASK B — I3 PHASE A`, `TASK B — I4 PHASE A`, `TASK B — PHASE C`. `docs/B_HANDOFF.md` is generated: the three contrasts with their outcomes copied from `i34_contrasts.py` output, the interpretation branch each landed in (from §4 and §6, verbatim), the energy‑stratified view, the cross‑method row, and what Track B does not settle (frozen‑model edits show what the trained model uses, not what a differently trained model would achieve; I4's masks come from one cell; registers n = 2).
