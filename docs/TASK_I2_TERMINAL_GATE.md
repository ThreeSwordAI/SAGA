# TASK I2 — Terminal patch‑gate sweep and the feature‑stage decision (D1)

**Intended repo path:** `docs/TASK_I2_TERMINAL_GATE.md`
**Written:** 2026‑09‑17 by Planning Claude. **Executor:** Claude Code, one session for Phase A, worktree `../SAGA-I2`, branch `task/I2`, commit tag `[I2]`.
**Depends on:** TASK I0 (merged to `main`): `saga.frozen` API, `results/frozen/I0_manifest/manifest.csv`, `results/frozen/splits/`, `docs/LOCKED_ANALYSIS.md` (DRAFT, D1 pending I2).
**Phases:** A — code, YAML, tests (this session). B1 — HPC sweep on the **calibration** split (the human). C1 — local tables + a D1 proposal; the human signs D1. B2 — the same sweep on the **evaluation** split (the human). C2 — final tables, figure data, handoff.
**Hard constraints:** inherit every rule in I0 handoff §8. No training, no optimizer, no probe fitting. Nothing selects a layer, mask, ε, or permutation. No historical file is edited. Claude Code never pushes.

---

## 0. The question and why it comes second

Under a CLS‑only loss, the last block's patch gate receives no gradient and sits at σ(0) = 0.5 in every SAGA checkpoint. I0 confirmed the architectural half on the real model: overriding that gate changes the CLS logits by exactly 0. The empirical half is open: **how much of the SAGA‑vs‑baseline gap on every patch diagnostic — cosine, effective rank, exceedance counts, the residual sink‑frequency map — is produced by that untrained constant, rather than by anything training did?** Both models have final‑block patch rows that no loss ever supervised; SAGA has an additional untrained factor on top. Until this is measured, no patch diagnostic computed at the historical stage (`hist` = `s12_pre_norm`) is safe to typeset, and I1, I3, I4, I5 do not know which stage to report. I2 answers that with one sweep and one pre‑declared rule, and closes D1.

## 1. Deliverables

| ID | Deliverable | Location | Phase |
|---|---|---|---|
| D1 | Conditions file — the only place conditions are defined | `configs/frozen/I2_terminal.yaml` | A |
| D2 | Per‑stage, per‑cell calibrated thresholds (new keys; `fixed_thresholds_canon.json` untouched) | `results/frozen/I2_terminal/thresholds_cal.json` | B1 (written by the runner on the baseline rows), tests in A |
| D3 | Records, per‑image diagnostics, exceedance maps per run | `results/frozen/I2_terminal/<run_id>/{records,diag}.parquet`, `maps.npz`, `conditions.yaml`, `run_meta.json` | B1, B2 |
| D4 | Table builder + decision‑rule module | `analysis/build_I2_tables.py`, `analysis/i2_decision.py` → `results/frozen/I2_terminal/tables/` | A code, C1/C2 run |
| D5 | D1 proposal note (generated) and, after signature, the frozen line in `docs/LOCKED_ANALYSIS.md` | `results/frozen/I2_terminal/D1_proposal.md` | C1 (proposal), human (signature) |
| D6 | Figure‑5A data | `figures_data/frozen/F5A_terminal.npz` | C2 |
| D7 | Job file (array over eligible run_ids) | `scripts/jobs/frozen_I2.sbatch` | A |
| D8 | Tests | `tests/test_I2_terminal.py` | A |
| D9 | Log entries; handoff generator | `docs/TASK_LOG.md`; `analysis/build_I2_handoff.py` → `docs/I2_HANDOFF.md` | A (log), C2 |

## 2. Cohort

All **19 eligible 300‑epoch conditions** from the manifest (`family ∈ {e2r_300ep, legacy_300ep}`, `status ∈ {eligible, eligible_legacy}`, `ckpt_kind = last`): 8 baseline, 8 SAGA, 3 registers. Cost is trivial (≤ 5 forward passes of 2,000 images per checkpoint), so there is no reason to subsample. The ablation arms are **not** in I2; they belong to the ablation appendix and would blur the stage question.

Pairing for gaps follows `analysis/build_pooled_tables.py` (seeded pairs s1–s1, s2–s2; legacy pairs as that module pairs them). Import it; do not re‑pair.

## 3. Conditions (`configs/frozen/I2_terminal.yaml`)

| condition_id | applies to | edit | stages captured in the same forward |
|---|---|---|---|
| `native` | all 19 | none | `s11_out`, `s12_pre_norm`, `s12_post_norm` |
| `term_0.50` | SAGA | `terminal_gate_override(model, 0.50)` | `s12_pre_norm`, `s12_post_norm` |
| `term_0.25` | SAGA | `terminal_gate_override(model, 0.25)` | `s12_pre_norm`, `s12_post_norm` |
| `term_0.75` | SAGA | `terminal_gate_override(model, 0.75)` | `s12_pre_norm`, `s12_post_norm` |
| `term_1.00` | SAGA | `terminal_gate_override(model, 1.00)` (bypass) | `s12_pre_norm`, `s12_post_norm` |

`s11_out` cannot depend on the terminal gate; it is captured once under `native` and a test asserts it is bit‑identical if captured under any `term_*` condition on a fake model. `native` and `term_0.50` must be bit‑identical on every real SAGA checkpoint (φ_L = 0 is a weight‑decay fixed point); the table reports the measured max |Δ| and flags any checkpoint where it is not 0.

Precision fp32; deterministic; image order by `image_id`; batch size from the job file only.

## 4. What is recorded per image, per condition, per stage

Into `records.parquet` (I0 schema): `nll`, `correct`, `top1`, `max_abs_logit_diff_vs_native`.

Into `diag.parquet` (one row per image × condition × stage), all on patch rows only, prefix excluded by the model's count:

| key | definition |
|---|---|
| `norm_p50, norm_p90, norm_p99, norm_p999, norm_max` | quantiles of per‑token L2 norm |
| `count_fixed_canon` | exceedances of the cell's historical τ from `fixed_thresholds_canon.json` — **defined only at `hist`**; `MISSING` elsewhere |
| `count_fixed_cal` | exceedances of `tau_cal[cell][stage]` (see §5) |
| `count_mad` | exceedances of the per‑image median + 5·MAD threshold |
| `cos_all` | mean pairwise cosine, closed form, all patch tokens |
| `cos_nosink_mad` | same among tokens below the per‑image MAD threshold |
| `eff_rank` | entropy‑based effective rank of the per‑image patch matrix (the project's existing definition in `saga/metrics.py`; import, do not reimplement) |
| `exceed_fixed_cal` (in `maps.npz`) | boolean `[N_patches]` indicator per image under `tau_cal` — aggregated to a per‑condition, per‑stage frequency map; also under `count_mad` |

Reuse `saga/metrics.py` functions wherever a definition already exists (closed‑form cosine, MAD threshold, effective rank, `infer_num_prefix_tokens`). New keys only; no historical key is written.

## 5. Per‑stage calibrated thresholds (new keys)

The historical τ is defined at `hist` only. For stage comparisons the runner computes, from each cell's **baseline** rows on the split being run, `tau_cal[cell][stage] = median over images of the per‑image MAD threshold` — the same recipe that produced the canon thresholds, applied per stage. Written to `results/frozen/I2_terminal/thresholds_cal.json` with the baseline run_ids, checkpoint shas, split sha and stage. At `hist` this reproduces the canon recipe on a different split; the table reports both `tau_canon` and `tau_cal[hist]` side by side so the split effect is visible. **`results/diagsplit/fixed_thresholds_canon.json` is never opened for writing; a test pins its sha.**

## 6. Tables (`analysis/build_I2_tables.py`)

| table | content |
|---|---|
| `T_I2a_invariance.csv` | per SAGA checkpoint × constant: max abs logit diff vs native, top‑1 agreement, mean |ΔNLL| |
| `T_I2b_sweep.csv` | per SAGA checkpoint × constant × stage: every diagnostic mean over images |
| `T_I2c_gaps.csv` | per cell × stage × diagnostic: paired SAGA−baseline gap (mean, per‑pair values, image‑level bootstrap CI over paired image means, 10,000 resamples, seed from `LOCKED_ANALYSIS` D6) at three settings: `hist/native`, `hist/term_1.00`, `s11_out`; plus `survival = gap(term_1.00)/gap(native)` and `survival_s11 = gap(s11_out)/gap(native)` |
| `T_I2d_maps.csv` | per pair × stage × condition: spatial Spearman between SAGA and baseline exceedance maps with the existing permutation reference (`analysis/address_analysis.py`; import, do not re‑derive), ring‑1 share, excess concentration vs the finite‑sample reference |
| `T_I2e_registers.csv` | registers vs baseline at `s11_out` and `hist` (native only), same diagnostics, n stated |

All tables carry the I0 provenance columns. Every generated note is pinned by a byte‑identity test.

## 7. The pre‑declared decision rule for D1 (`analysis/i2_decision.py`)

Computed on **calibration** only, for the ViT‑S/mixup cell (4 pairs), on the primary diagnostics `cos_all`, `cos_nosink_mad`, `eff_rank`, `count_fixed_cal`, `count_mad`:

1. **Terminal gate is a minor contributor** if, for every primary diagnostic, `sign(gap_term_1.00) == sign(gap_native)` and `survival ≥ 0.70` (paired means). → **D1 = `hist`** (native gate). The paper reports `term_1.00` and `s11_out` numbers as controls in the appendix.
2. **Otherwise, if** for every primary diagnostic `sign(gap_s11) == sign(gap_native)` and `survival_s11 ≥ 0.70` → **D1 = `s11_out`** for all new diagnostics; final‑block diagnostics are reported only as a readout‑contaminated control, and the paper says so.
3. **Otherwise** → **D1 = `s11_out`** for reporting, and I2 is promoted to a main finding: the historical SAGA‑vs‑baseline diagnostic distinction was substantially a terminal‑gate effect. The claim ladder narrows accordingly.

The cutoff 0.70 is conventional and declared as such; the complete tables are reported whatever the branch. The ViT‑S/nomix and ViT‑B cells are reported alongside but do not vote (n = 2). The module returns the branch, the per‑diagnostic survival values, and the sentence to paste into `LOCKED_ANALYSIS.md`; **the human signs D1 (name, date, git sha) before any B2 or any I1/I3/I4/I5 Phase B runs**.

## 8. Phases and acceptance

**Phase A (this session).** D1, D4 (code), D7, D8, log. Tests (CPU, fake models, fake data): YAML parses and the runner refuses any condition/layer/ε not in it; `s11_out` identical across constants; `native ≡ term_0.50` on a fake SAGA model with φ_L = 0; register model (5 prefix tokens) runs `native` with correct patch counts; diag keys present at the right stages and `count_fixed_canon` is `MISSING` off‑`hist`; `thresholds_cal.json` writer refuses to touch the canon file (sha pinned); `maps.npz` shapes; table builder byte‑identity; decision rule exercised on synthetic gaps for all three branches; provenance columns present. `pytest -q` green; no existing test weakened.

**Phase B1 (human, HPC).** Merge `task/I2` → `main`, push, pull; run §9 on `calibration.json`. Push back `results/frozen/I2_terminal/` (small) — records and diag parquet for 19 × 2,000 images are a few MB; `maps.npz` small.

**Phase C1 (next session).** Run the table builder and the decision module; write `D1_proposal.md`; draft Figure 5A from calibration data (labelled draft); log. **Stop and hand D1 to the human.**

**Phase B2 (human).** After D1 is signed: same array on `evaluation.json`.

**Phase C2.** Final tables from evaluation; `F5A_terminal.npz`; `build_I2_handoff.py`; log.

## 9. HPC block (for the human)

```bash
source classification/scripts/env_alex.sh          # before any set -u
python tools/frozen_manifest_hashes.py --verify     # refuse if any eligible checkpoint changed
# calibration sweep (B1); the array indexes the 19 eligible run_ids from the manifest
sbatch --partition=a100 --gres=gpu:a100:1 --exclude=a0801 --array=0-18 scripts/jobs/frozen_I2.sbatch \
       results/frozen/splits/calibration.json
# after D1 is signed (B2)
sbatch --partition=a100 --gres=gpu:a100:1 --exclude=a0801 --array=0-18 scripts/jobs/frozen_I2.sbatch \
       results/frozen/splits/evaluation.json
bash scripts/sync_results.sh
```

The job file reads run_ids and checkpoint roots from the manifest (via `tools/derive_runs.ckpt_dir_for()`), verifies `ckpt_sha256` before loading, writes the split sha into every row, and writes the completion marker last. Resubmission adds what is missing and rewrites nothing.

## 10. What not to do

- Do not run on `evaluation.json` before D1 is signed.
- Do not include ablation arms, fine‑tuned, dense, or TTR rows.
- Do not add stages beyond the four; do not change `saga/frozen/` except additively (a diagnostic helper in `saga/frozen/diag.py` is fine; the edit context managers are not touched).
- Do not report `count_fixed_canon` at any stage but `hist`, and do not recalibrate the canon thresholds.
- Do not decide D1 by inspection; the module decides, the human signs.
- One worktree; no push.

## 11. Log and handoff

`docs/TASK_LOG.md`: `## 2026‑09‑DD — TASK I2, PHASE A (terminal sweep code, YAML, decision rule)`; later entries for C1 (with the proposed D1 branch and survival values) and C2. `docs/I2_HANDOFF.md` is generated; the D1 sentence in it is the signed line copied from `LOCKED_ANALYSIS.md`.
