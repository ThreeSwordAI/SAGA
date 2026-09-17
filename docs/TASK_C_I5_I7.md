# TASK C — I5 (does the distinction matter for a useful readout?) then I7 (are high‑norm tokens attention sinks here? and the TTR operating curve)

**Intended repo path:** `docs/TASK_C_I5_I7.md`
**Written:** 2026‑09‑17 by Planning Claude. **Executor:** Claude Code, worktree `../SAGA-C`, branch `task/C`, commit tags `[I5]` and `[I7]`. One task file, two work packages, executed in order with a **checkpoint** between them; sessions may be split.
**Depends on:** I0 and I2 merged (framework, manifest, splits, D1 = `s11_out`). **Nothing from Tracks A or B.** Tracks A and B are running in `../SAGA-A` and `../SAGA-B`; this track never touches them.
**Before the freeze:** §9 contains the D9 (I5) and D10 (I7) parameter lines for the human to sign into `docs/LOCKED_ANALYSIS.md` **before** D5 closes and the document is frozen. Track C's Phase B does not run until they are signed.
**Phases:** A — I5 code + tests → checkpoint → A′ — I7 code + tests + TTR curve from existing files → stop. B — HPC (the human): I5a, I7 (and I5b if weights exist). C — local: tables, figure data, handoff.
**Hard constraints:** inherit I0 handoff §8 and the I2/A/B rules. No training, no fine‑tuning, no head fitting, no optimizer. Nothing selected on the evaluation split. No historical file edited. Claude Code never pushes.

---

## 0. The two questions

**I5.** The paper's thesis is that outlier counts and cosine do not tell you what changed for a consumer of the features. That needs a consumer. I5 provides two that require no training: (a) **training‑free geometric correspondence** — a patch's feature should identify the same patch after a known image transform; accuracy under nearest‑neighbour matching is a direct, per‑image, chance‑calibrated readout of local feature usefulness; (b) **the existing ADE20K heads, frozen**, re‑evaluated for paired per‑image records, only if the manifest shows matched‑epoch weights. Both are measured for baseline, SAGA and registers, at `s11_out` (D1), at `hist`, and at `s12_post_norm` (what a dense head consumes) — and, for SAGA, under the terminal‑gate bypass, which turns I2's "the classifier cannot see it" into "can a feature consumer see it?"

**I7.** The project measures **norm** outliers. Calling them "attention sinks" is a claim about incoming attention that has not been measured at the positions the paper reports. I7 measures it, with a wording rule declared in advance, and draws the test‑time‑registers operating curve from existing sweep files so the chosen TTR point is shown on its trade‑off rather than alone.

I5 and I7 are **secondary endpoints** under D7: the three primary contrasts of the paper are C1–C3 in Track B. Every I5/I7 comparison is reported with CIs and labelled secondary or descriptive; none decides a primary claim.

## 1. Deliverables

| ID | Deliverable | Location | Phase |
|---|---|---|---|
| C1 | Feature capture at declared stages + transform pipeline (new module, one‑line runner hook) | `saga/frozen/features.py`, `saga/frozen/transforms.py` | A |
| C2 | I5a conditions | `configs/frozen/I5_readout.yaml` | A |
| C3 | Correspondence matcher, per‑image records, on‑the‑fly MAD exceedance flags | `saga/frozen/correspondence.py` | A |
| C4 | I5b frozen‑head evaluator (eval‑only wrapper around the existing seg pipeline), conditional on the manifest | `tools/frozen_I5b_seg_eval.py` | A |
| C5 | I5 analysis + tables | `analysis/frozen_I5_analysis.py` → `results/frozen/I5_readout/tables/T_I5a–e*.csv` | A code, C run |
| C6 | I5 job files | `scripts/jobs/frozen_I5a.sbatch`, `scripts/jobs/frozen_I5b.sbatch` | A |
| C7 | Attention capture (fused attention off for the dump; new module) + incoming‑mass summaries + value‑norm ratio | `saga/frozen/attention.py`, `configs/frozen/I7_attention.yaml` | A′ |
| C8 | Wording‑rule module | `analysis/i7_wording.py` | A′ |
| C9 | TTR operating‑curve builder from existing files | `analysis/frozen_I7_ttr_curve.py` | A′ |
| C10 | I7 analysis + tables; job file | `analysis/frozen_I7_analysis.py` → `results/frozen/I7_attention/tables/T_I7a–c*.csv`; `scripts/jobs/frozen_I7.sbatch` | A′ code, C run |
| C11 | Figure data | `figures_data/frozen/F5B_readout.npz`, `F1C_readout_draft.npz`, `F7_attention.npz`, `F_ttr_curve.npz` | C |
| C12 | Tests | `tests/test_I5_readout.py`, `tests/test_I7_attention.py` | A, A′ |
| C13 | Log entries; handoff generator | `docs/TASK_LOG.md`; `analysis/build_C_handoff.py` → `docs/C_HANDOFF.md` | each phase |

## 2. Cohort and splits

- **I5a:** all 19 eligible 300‑epoch conditions; plus, for the 8 SAGA checkpoints, the `term_1.00` bypass. Images: `results/frozen/splits/sub2k.json` (2,000 evaluation images). One forward per (image, transform) captures all three stages; ≈ 8,000 forwards per checkpoint (+8,000 for SAGA bypass). Trivial.
- **I5b:** `dense_seg` rows of the manifest for ViT‑B/mixup baseline and SAGA at a **matched epoch** (`epochs_completed` equal, weights present). The registers seg row is not backbone‑matched (unseeded legacy backbone) and is reported only as flagged context if run at all. If matched weights are `MISSING` in the manifest, I5b is **DROPPED** and the handoff says so; nothing is retrained or re‑fetched.
- **I7:** the ViT‑S/mixup cell — 4 baseline, 4 SAGA, 2 registers — on `results/frozen/splits/sub1k.json` (1,000 evaluation images). Attention maps are never stored; only per‑image, per‑position summaries.
- Nothing in Track C runs on discovery or calibration.

## 3. I5a — the correspondence readout

**Transforms (`saga/frozen/transforms.py`), all with exactly known patch correspondence on the 14×14 grid:**

| id | transform | known correspondence |
|---|---|---|
| `T0` | identity (the reference forward) | — |
| `T1` | horizontal flip | patch (r, c) ↔ (r, 13 − c), all 196 |
| `T2` | grid‑aligned translation by 1 patch: resize the source to 240×240, take the two 224×224 crops offset by 16 px horizontally; the "original" is the left crop, the "transformed" the right crop | 14 × 13 shared positions |
| `T3` | translation by 2 patches: 256×256 resize, crops offset by 32 px | 14 × 12 shared positions |

Non‑grid‑aligned transforms (scale, rotation, sub‑patch shifts) are excluded because their correspondence is not exact; the docstring says so. Preprocessing follows the model's evaluation pipeline (same normalization); for T2/T3 the "original" is the left crop, not the standard centre crop, so both members of a pair pass through identical processing.

**Descriptors:** patch tokens at `s11_out`, `hist`, `s12_post_norm`; **primary** = raw tokens L2‑normalized per token; **secondary** = image‑centred then L2‑normalized (subtract the image's mean patch token first). Prefix rows excluded by the model's count.

**Matching:** for each shared position in the transformed image, cosine nearest neighbour among all 196 original‑image patches (secondary: the transformed image's own patches excluded — not needed since images differ). Correct‑exact = NN is the known position; correct‑1 = NN within Chebyshev distance 1. Chance = 1/196 (exact) and ≤ 9/196 (1‑tolerance), computed and printed on every table.

**Per‑image record (`corr_records.parquet`):** I0 provenance + `transform`, `stage`, `descriptor`, `acc_exact`, `acc_1`, `n_shared`, `mean_nn_sim`, and, from the same forward, `count_mad_s11` and the per‑position MAD exceedance flag of the original image at the matched stage (computed on the fly with `saga.metrics`, no cross‑track dependency), so accuracy can be split into exceedance positions vs others.

**Tables (`analysis/frozen_I5_analysis.py`):**

| table | content |
|---|---|
| `T_I5a_readout` | per checkpoint × stage × transform × descriptor: mean `acc_exact`, `acc_1`, image‑level bootstrap CI (10,000, seed 0), chance |
| `T_I5b_methods` | at `s11_out` (primary), `hist`, `s12_post_norm`: SAGA − baseline paired by seed (`s1`–`s1`, `s2`–`s2`; legacy pairs labelled), registers − baseline (n = 2, labelled); per transform; CIs; **secondary endpoint** stamped on the table |
| `T_I5c_terminal` | 8 SAGA checkpoints: `acc` native vs `term_1.00` at `hist` and `s12_post_norm` (the bypass cannot change `s11_out`; a test asserts identity there) — can a feature consumer see the terminal constant? |
| `T_I5d_positions` | accuracy at MAD‑exceedance positions vs non‑exceedance positions, per checkpoint and stage, with CIs |
| `T_I5e_diag_vs_readout` | within checkpoint: Spearman over images between `acc_exact` and `count_mad_s11` / `cos_all` (from I2's evaluation primary pack where the image sets overlap, else recomputed on the fly); across checkpoints (n = 19): descriptive scatter data only, no test |

Interpretation guide, fixed now: if method ordering on `acc` matches the ordering on counts/cosine, diagnostics predicted this utility here; if it does not, or if `T_I5c` shows the terminal constant moving `acc` at the head‑facing stage, the paper's thesis has a functional instance. Either is reportable.

## 4. I5b — the existing ADE20K heads, frozen (conditional)

Eval‑only wrapper around the repository's existing segmentation evaluation: load the matched‑epoch baseline and SAGA seg checkpoints, run ADE20K validation once each, write **per‑image** IoU/accuracy records (the historical numbers are aggregate mIoU without pairing), and, for the SAGA seg model, a second pass under `term_1.00` on the backbone (the head was trained on gated features; this measures sensitivity, labelled exploratory). Tables: `T_I5f_seg` — mIoU_ss, paired per‑image Δ with CI, per‑class IoU deltas (all 150), background rows; `T_I5g_seg_terminal`. No re‑training, no re‑tuning, no change to the seg training code.

## 5. I7 — incoming attention at the exceedance positions

**Capture (`saga/frozen/attention.py`):** with fused attention disabled for the dump (as `tools/dump_attention.py` does), obtain the post‑softmax attention `A[b, h, q, k]` of `blocks[10]` and `blocks[11]`. Compute on the fly and store per image, per key position: `in_mass_patchq[k]` = mean over heads and **patch** queries of `A[..., q, k]`; `in_mass_clsq[k]` = mean over heads of `A[..., cls, k]`; for register models also the mass landing on each prefix token. Also `value_norm[k]` = ‖v_k‖ averaged over heads (the Fesser et al. NOP/broadcast diagnostic; exploratory). Stored: `incoming_mass.npz` per run (1,000 × 197/201 × 2 blocks × 3 quantities, float32 ≈ 5 MB), committed.

**Alignment:** attention inside `blocks[ℓ]` is computed from that block's **input** tokens, so it is compared with exceedance of those tokens: `blocks[10]` ↔ exceedance at the output of `blocks[9]` (capture it in the same forward as stage `in_b10`, defined here additively — only for I7's own use); `blocks[11]` ↔ exceedance at `s11_out`. Exceedance flags per image by MAD (primary, threshold‑free) and by I2's `tau_cal[s11_out]` where defined.

**Tables:**

| table | content |
|---|---|
| `T_I7a_incoming` | per checkpoint × block × query group: mean incoming mass per exceedance token vs per non‑exceedance token; ratio with image‑level bootstrap CI; share of total mass received by exceedance tokens vs their share of positions; AUC of `in_mass` for the exceedance flag |
| `T_I7b_registers` | registers checkpoints: mass landing on the 4 register tokens vs CLS vs patch exceedances (descriptive) |
| `T_I7c_value_norm` | `value_norm` at exceedance vs non‑exceedance positions, per checkpoint (exploratory) |

**Wording rule (`analysis/i7_wording.py`, pre‑declared — D10):** the paper uses "attention sink" for the exceedance tokens **only if**, in every fresh ViT‑S/mixup baseline checkpoint (`s1`, `s2`), at both blocks, for patch queries, the ratio of mean incoming mass per exceedance token to mean per non‑exceedance token is ≥ 3 with the bootstrap CI excluding 3. Otherwise the paper says "high‑norm outlier tokens" throughout and reports the measured ratio. The 3× is conventional and declared as such; the module returns the verdict and the sentence.

## 6. I7 — the TTR operating curve (CPU, existing files)

From `results/tables/T_ttr.csv`, `results/notes/ttr_baseline.md`, and whatever sweep JSONs exist under the TTR run directories (per layer range × neuron count: sink reduction, top‑1 Δ), build `F_ttr_curve.npz`: sink‑reduction vs top‑1 Δ per cell with the validation‑gate thresholds drawn and the chosen operating point marked. Record which grid points exist in committed files and which are `MISSING` (on the cluster or never run); no new TTR runs. Table `T_I7d_ttr_curve` lists every point with its provenance.

## 7. Guards and rules

- **New module + one‑line hook.** Feature capture (`features.py`) and attention capture (`attention.py`) are new modules; each touches `runner.py` with a single hook line. Tracks A and B are adding their own hooks; keep yours separable.
- **Identity checks:** `T0` reproduces the plain forward bit‑for‑bit; `term_1.00` cannot change `s11_out` (test on a fake model and assert on real records in Phase C).
- **No stored attention maps; no stored raw features** beyond what the matcher needs in memory. Records only.
- **No selection:** transforms, stages, descriptors, blocks and the 3× rule are fixed in this file and in D9/D10; the runner refuses anything not in the YAML.
- **Descriptive wording** in all outputs; "sink" appears in code only inside `i7_wording.py`.
- **AST guard** from I0/B extended to this track's modules.

## 8. Phases and acceptance

**Phase A (I5).** C1–C6, I5 tests, log `TASK C — I5 PHASE A`. Tests (CPU, fake models/data): every transform's correspondence table is a bijection on the shared set and matches a synthetic image with unique patches at 100%; chance values; NN matcher on synthetic descriptors; L2 and centred descriptors; `T0` identity; `term_1.00` leaves `s11_out` unchanged; on‑the‑fly MAD flags equal `saga.metrics`; prefix exclusion on a fake 5‑prefix model; I5b wrapper refuses to run without matched weights in the manifest and refuses any training‑mode call; bootstrap determinism; table byte identity. `pytest -q` green. **Checkpoint:** commit, report, wait for "continue with I7".

**Phase A′ (I7).** C7–C10, C12 (I7 tests), log `TASK C — I7 PHASE A`. Tests: fused‑attention‑off capture yields rows summing to 1 on a fake model; incoming‑mass and value‑norm computations against hand values; prefix separation for 5 prefix tokens; `in_b10` equals the output of `blocks[9]`; wording rule on synthetic data (both verdicts, CI edge); TTR curve builder on a fake sweep file with `MISSING` points; byte identity. Print the Phase B block. **Stop.**

**Phase B (human, HPC).** Only after D9 and D10 are signed. Merge `task/C` → `main` (after A and B if all touched `runner.py`), push, pull, run §10. Push back records‑derived small files; `corr_records.parquet` is small (≈ 2,000 × transforms × stages × descriptors rows per run) and is committed; `incoming_mass.npz` committed.

**Phase C (next session).** All tables; `i7_wording.py` verdict copied into the handoff; F5B, F1C draft, F7, TTR curve data; `build_C_handoff.py` → `docs/C_HANDOFF.md`; log `TASK C — PHASE C`.

## 9. Lines for `docs/LOCKED_ANALYSIS.md` — to be signed by the human before the freeze

> **D9 (I5, secondary endpoints).** Correspondence readout on `sub2k` (2,000 evaluation images; sha recorded). Transforms T1 flip, T2 one‑patch translation via 240→224 offset crops, T3 two‑patch translation via 256→224; exact grid correspondence only. Descriptors: patch tokens at `s11_out` (primary), `hist`, `s12_post_norm`; L2‑normalized (primary) and image‑centred L2 (secondary). Cosine nearest neighbour; `acc_exact` primary, `acc_1` secondary; chance printed. Image‑level bootstrap 10,000, seed 0. Paired by seed where seeds match. I5b runs only on matched‑epoch weights present in the manifest. All I5 comparisons are secondary under D7.
>
> **D10 (I7, wording rule).** Incoming attention captured at `blocks[10]` and `blocks[11]` on `sub1k`, fused attention off for the dump, compared with MAD exceedance of the same block's input tokens. The paper uses "attention sink" only if, in both fresh ViT‑S/mixup baselines, at both blocks, for patch queries, mean incoming mass per exceedance token ≥ 3× that per non‑exceedance token with the bootstrap CI excluding 3; otherwise "high‑norm outlier tokens" throughout. The 3× is conventional and declared. Value‑norm ratio and register‑token mass are exploratory.

## 10. HPC block (for the human — after D9/D10 are signed)

```bash
source classification/scripts/env_alex.sh          # before any set -u
python tools/frozen_manifest_hashes.py --verify
# I5a: 19 runs; SAGA runs also execute term_1.00; split fixed to sub2k in the job
sbatch --partition=a100 --gres=gpu:a100:1 --exclude=a0801 --array=0-18 scripts/jobs/frozen_I5a.sbatch
# I7: ViT-S/mixup cell, 10 runs, sub1k
sbatch --partition=a100 --gres=gpu:a100:1 --exclude=a0801 --array=0-9 scripts/jobs/frozen_I7.sbatch
# I5b: only if the manifest shows matched-epoch seg weights (the job refuses otherwise); ADE20K val on the cluster
sbatch --partition=a100 --gres=gpu:a100:1 --exclude=a0801 scripts/jobs/frozen_I5b.sbatch
bash scripts/sync_results.sh
```

## 11. What not to do

- No training, fine‑tuning, head fitting, or probe fitting; I5b is evaluation of existing heads only.
- No transforms without exact grid correspondence; no extra stages, blocks, or descriptors.
- No stored attention maps or raw feature dumps.
- No "sink" wording outside `i7_wording.py`; no functional language elsewhere.
- No comparison in I5/I7 presented as a primary claim; D7 fixed those in Track B.
- One worktree; new files only, WP‑prefixed; single‑line hooks in shared files; no push.

## 12. Log and handoff

Log entries: `TASK C — I5 PHASE A`, `TASK C — I7 PHASE A`, `TASK C — PHASE C`. `docs/C_HANDOFF.md` is generated: the D9/D10 lines copied from `LOCKED_ANALYSIS.md`, the I7 wording verdict and the sentence it returns, each table's headline row, the I5b status (run / DROPPED with reason), the TTR points available vs `MISSING`, and what Track C does not settle (correspondence under exact transforms is one utility, not utility; the 3× rule is a convention; registers n = 2).
