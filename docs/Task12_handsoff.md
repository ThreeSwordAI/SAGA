# TASK 12 — matched-init ablation: handoff

**Written 2026-09-15. Status: Phases A, B and C COMPLETE. Phase D triggered but
NOT started. Nothing is pending from the HPC.**

Everything below is read from a committed file; each section names its source.
No number in this document was typed from memory, and where evidence is
missing it says MISSING rather than guessing.

---

## 1. TL;DR — what the experiment answered, and what it did not

**The question:** SAGA's gate starts at sigmoid(0) = 0.5, so every patch token
is halved at initialisation. Is the gain the *learned spatial structure*, or
would any per-layer/per-head rescaling at the same insertion point do as well?
Nothing in `results/` distinguished the two before this task.

| | verdict |
|---|---|
| **Does the spatial arm beat the non-spatial controls?** | **Yes**, on a single seed. E beats the head-scalar control by +0.284, LayerScale by +0.480, the frozen floor by +0.224. LayerScale is the *worst* of the six arms. |
| **Does the spatial gate beat no gate at all?** | **No** — E is 0.064 *below* the ungated baseline here, against +0.456 at 300 epochs. One seed; see §5.3 before reading anything into it. |
| **Does the initialisation matter?** | F (init +4) is the only arm above baseline (+0.176) and beats E by +0.240 — **but arm F is not a clean init test** (§5.6). |
| **Does the gate target the sink address?** | **Yes, and it replicates TASK-07 independently**: ρ = −0.7461 at layer 7, Bonferroni p = 0.0140. This is the strongest result in the task. |
| **Is the sink *reduction* a learned effect or an init-scale effect?** | **UNANSWERED.** The primary sink metric saturates in this cell and orders nothing (§5.4). This is the one question the ablation was expected to settle and did not. |

---

## 2. What was built and run

| phase | what | where |
|---|---|---|
| **A** | five gate modes, the six-arm matrix, launchers, 51 tests | local, commits `94a85cc` → `d5aeece` |
| **B** | 2-epoch six-arm smoke (job 4228405), then six 100-epoch runs on 4×A100 | HPC, ~6.5 h per arm |
| **B** | read-only completeness gate over the six runs: **160 checks, 0 failed** | local, commit `d04b871` |
| **C** | derivation job → exact fp32 eval + canonical diagnostics + address maps; **48/48 integrity checks** | HPC (`abl_derive.sbatch`, run on a100) |
| **C** | tables, address correlations, figure, generated note | local, commits `eafbe5f` → `a98261b` |
| **D** | **not started** — condition met, decision is the human's | — |

Twelve `[TASK-12]` commits, all merged to `main` (currently `4d5d095`).
`pytest -q`: **586 passed, 30 skipped**.

---

## 3. The design — six arms, differing only in the gate

All six: **ViT-S/16, mixup recipe, seed 0, 100 epochs, 4×A100**, resolved onto
`classification/configs/base.yaml` exactly as `e2r_vits_mixup_*` is. They
differ **only** in `gate_mode` (and `gate_init_logit` for arm F).

| arm | run_id | gate_mode | gate params (registered / trainable) |
|---|---|---|---|
| A | `abl_vits_mixup_baseline_s0` | `none` — no gate module at all | 0 / 0 |
| B | `abl_vits_mixup_const05_s0` | `const` — φ registered, **frozen at 0** (G = 0.5) | 14,112 / **0** |
| C | `abl_vits_mixup_headscalar_s0` | `headscalar` — φ[L,H,1], broadcast over positions | 72 / 72 |
| D | `abl_vits_mixup_layerscale_s0` | `layerscale` — per-channel γ at the same insertion point | 4,608 / 4,608 |
| E | `abl_vits_mixup_spatial_i0_s0` | `spatial` — **SAGA as shipped**, φ[L,H,196], init 0 | 14,112 / 14,112 |
| F | `abl_vits_mixup_spatial_i4_s0` | `spatial`, init +4 → sigmoid = 0.982 | 14,112 / 14,112 |

The counts are *counted from a model built at each run's own recorded
`gate_mode`*, never tabulated by hand (`analysis/build_ablation_tables.py`).

### How "differ only in gate_mode" was enforced, not asserted

1. The six resolved configs are diffed key by key; anything outside
   {`run_id`, `variant`, `model.gate`, `model.gate_mode`,
   `knobs.gate_init_logit`, `instrumentation.log_grad_phi`} differing is a
   FAIL. Checked in the smoke, in the Phase-B gate, and in the test suite.
2. The 100-epoch schedule is written **once** as a matrix-wide `overrides`
   block, so no arm can drift onto its own schedule.
3. A test asserts the six arms share every non-gate weight at init and every
   activation upstream of the gate, and that the real trainer + real loader
   deliver the same data in the same order.
4. **Structural check that each arm is what it claims:** the final
   within-layer spatial std of the gate is **0 by construction** for B and C,
   and > 0 only for E and F. Confirmed on the real runs: B 0.000000,
   C 0.000000, E 0.022243, F 0.020688.

---

## 4. Where every result lives

| file | what it holds |
|---|---|
| `results/tables/T2_ablation.csv` | **the headline table.** One row per arm: fp32 top-1/top-5, Δ vs A, gate parameter counts, canon-τ sinks + saturation flag, MAD sinks, both oversmoothing variants, eff. rank, final gate mean and spatial std, and the bf16/in-training columns kept separate |
| `results/tables/T2_ablation_address.csv` | 103 rows: per-layer gate-vs-address Spearman ρ with the exact permutation null, per basis, plus the extremal-layer rows with Bonferroni p |
| `results/notes/ablation.md` | **the readable answer** (210 lines, generated — a test re-renders it from the tables and requires byte-identity). Answers the four questions, flags the inversion, documents the saturation, states the caveats |
| `results/figures/F_ablation_draft.pdf` (+ `.png`) | top-1 by arm; final gate maps for B/C/E/F on a shared colorbar; spatial std vs depth |
| `results/runs/abl_*/log.csv` | per-epoch training history: 100 rows, full-val top-1 every epoch (bf16) |
| `results/runs/abl_*/eval/imagenet_val_{best,last}.json` | **the canonical fp32 top-1**, full 50 000 images, asserted before writing |
| `results/runs/abl_*/diag/diag_final_{best,last}.json` | canonical diagnostics on the frozen 10k split, incl. the backfilled `sink_fixed_canon` |
| `results/runs/abl_*/diag/diag_final_{best,last}_addr.json` | the 196-position sink-frequency maps (the "address") |
| `results/runs/abl_*/diag/diag_e0[0-9]9.json` | the trainer's own periodic diagnostics, every 10 epochs |
| `results/runs/abl_*/gates/phi_e###.npz` | **φ at every one of the 100 epochs** for B, C, E, F — this is how the arm-F decay trajectory in §5.6 was measured |
| `results/runs/abl_*/grads/grad_phi.csv` | per-layer ‖∂L/∂φ‖ for E and F, epochs 0–30 |
| `results/runs/abl_*/meta.json` | provenance incl. `ckpt_dir` (the checkpoints are on **woody**, not in the repo) |
| `docs/TASK_LOG.md` | the chronological record, five TASK-12 entries |

Checkpoints are **not** in the repo: `/home/woody/iwi5/iwi5359h/SAGA/abl_ckpt/<run_id>/ckpt/`
on the HPC (~1.0 GB per run). `*_norms.npz` are git-ignored and must not be
deleted — the address maps and any re-thresholding read them.

---

## 5. The results

### 5.1 Accuracy — `results/tables/T2_ablation.csv`

Exact fp32, full 50 000-image val, `last` checkpoint, one seed per arm:

| arm | gate_mode | trainable | top-1 | Δ vs A | top-5 |
|---|---|---|---|---|---|
| A | none | 0 | 76.526 | — | 93.232 |
| B | const 0.5 | 0 | 76.238 | −0.288 | 93.010 |
| C | headscalar | 72 | 76.178 | −0.348 | 92.842 |
| D | layerscale | 4,608 | 75.982 | −0.544 | 92.836 |
| E | spatial (init 0) | 14,112 | 76.462 | −0.064 | 92.986 |
| F | spatial (init +4) | 14,112 | **76.702** | **+0.176** | 93.094 |

### 5.2 Q1 — the spatial arm beats every non-spatial control

- E − B (frozen floor) = **+0.224**
- E − C (head scalar) = **+0.284**
- E − D (LayerScale) = **+0.480**

**This is the comparison the task existed for, and it goes SAGA's way.** The
reviewer's hypothesis — that a per-layer/per-head rescaling would do as well —
is not what happened: the head-scalar arm is 0.284 below the spatial one, and
standard LayerScale is the worst arm in the experiment, 0.544 below the
ungated baseline.

**The margin is smaller than the spread between arms** (0.720 from D to F),
which is exactly TASK-12's Phase-D trigger: with n = 1 this ordering has no
standard error.

### 5.3 Q3 — E is below the ungated baseline here, against +0.456 at 300 epochs

`results/tables/e2_pooled.csv` (ViT-S/mixup, 300 epochs, n = 4): SAGA −
baseline = **+0.456**, SE 0.212, from per-repeat deltas −0.126, +0.570,
+0.490, +0.890.

Here: E − A = **−0.064**.

The note flags this as a sign inversion, loudly, as the task requires — **and
then calibrates it**: one of those four 300-epoch repeats was itself −0.126.
A single 100-epoch draw of this size sits inside the spread the headline cell
already shows. It is **not** evidence of an inversion and **not** evidence of
replication. It is one seed on a shorter schedule.

### 5.4 The primary sink metric SATURATES — the one question left unanswered

Under the committed `vit_small|mixup` canon τ = **20.8515625**, every arm
counts essentially all 196 patch tokens as sinks:

| arm | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| `sink_fixed_canon` | 196.0000 | 195.9180 | 195.9790 | 195.9999 | 195.9632 | 195.9988 |

That is above TASK-02B's own saturation flag (≥ 95 % of the patch tokens), and
a threshold every token clears cannot rank anything. The differences are the
last decimal of a saturated count.

**Cause — a schedule mismatch, not a fault in the runs.** τ was calibrated on
the *300-epoch* `e2r_vits_mixup_baseline_s1`. These are 100-epoch models with a
different norm scale: arm A's own last-block patch norms have a median of
**37.28**, and its own per-image MAD threshold — the quantity the canon
definition is built from — is **52.95**
(`results/runs/abl_vits_mixup_baseline_s0/diag/diag_final_last_normstats.json`,
fields `p50` and `mean_threshold_mad_k5`). A τ calibrated on this cell would
sit near the latter, roughly **2.5×** the one applied.

**Nothing was recalibrated.** TASK-12's acceptance list requires the canon τ be
taken from the ViT-S/mixup cell and never recalibrated. So the saturation is
reported, a `sink_canon_saturated` column carries it, and **the mechanism half
of this ablation is recorded as unanswered rather than answered either way.**

**The MAD metric cannot stand in for it**, and this is the important part:

| arm | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| `sink_mad_k5` | 11.7130 | **4.4334** | 6.6557 | 17.4579 | **4.6851** | 10.0260 |

On MAD the frozen arm B (zero trainable parameters) removes as many sinks as
the learned spatial arm E. That *looks* like "the init-scale does all the
work" — but PROJECT.md §3.2 records that per-image median+5·MAD falls with the
norm bulk, and a uniform 0.5 gate compresses exactly that bulk. So arm B
scoring well here is the **predicted behaviour of the metric**, not evidence
about the mechanism. It is equally not evidence that the two differ. The
question is open.

**To close it** would need a τ calibrated on *this* cell's own arm-A baseline,
declared under its own key with its own provenance (e.g.
`vit_small|mixup@100ep`) — a decision for the human, never a silent reuse of
the canon name. The norms are already on the HPC, so this is a CPU-only
re-thresholding, not a retrain.

### 5.5 Other diagnostics

| arm | oversmoothing | oversmoothing (no-sink) | eff. rank |
|---|---|---|---|
| A | 0.2680 | 0.2919 | 115.53 |
| B | 0.2921 | 0.2993 | 125.18 |
| C | 0.2745 | 0.2842 | 122.39 |
| D | 0.3461 | 0.4022 | 100.60 |
| E | 0.2767 | 0.2835 | 124.67 |
| F | 0.3043 | 0.3263 | 118.23 |

Context, and it matters: the **300-epoch** baseline of this same cell sits at
oversmoothing 0.43144 and eff. rank 89.75. A 100-epoch model is a far less
pathological object than a 300-epoch one, so there is much less for any gate to
recover here. SAGA's 300-epoch oversmoothing win (−0.13 paired) does **not**
reproduce at 100 epochs (E is +0.009 *worse* than A) — consistent with there
being no headroom, but that is an interpretation, not a measurement.

Note LayerScale (D) is worst on every axis: lowest top-1, most sinks, worst
oversmoothing, lowest effective rank.

### 5.6 Arm F — the init does not survive its own schedule

Measured from the per-epoch φ dumps
(`results/runs/abl_vits_mixup_spatial_i4_s0/gates/phi_e###.npz`):

| epoch | 0 | 24 | 49 | 74 | 99 |
|---|---|---|---|---|---|
| mean φ | 3.9926 | 1.5686 | 0.6737 | 0.5059 | 0.4808 |
| mean gate | 0.9819 | 0.8243 | 0.6598 | 0.6221 | 0.6163 |

φ = 0 is a fixed point of decoupled weight decay; φ = +4 is not. Before launch
I computed that weight decay **alone** would carry φ = +4.0 to +0.31 over this
schedule (∏(1 − lr·wd) = 0.0784); the measured endpoint is 0.4808. Arm F's
median ‖∂L/∂φ‖ is 9.96e-05 against arm E's 6.59e-04 — **6.6× smaller**,
consistent with sigmoid saturation at φ = 4.

So **arm F is not a clean test of initialisation**: it is a weaker-gradient arm
that decays toward arm E's regime. Its +0.176 must not be read as "initialise
at +4". Both arms share the weight-decay treatment deliberately (they may
differ only in `gate_mode` and `gate_init_logit`).

### 5.7 The gate targets the sink address — an independent replication

Source: `results/tables/T2_ablation_address.csv`. Correlations use TASK-07's
**exact permutation null** (8 dihedral × 196 torus rolls, 1568 transforms),
imported from `analysis/address_analysis.py` — never the iid Spearman
reference, which is far too generous for maps this smooth. The extremal layer
is an argmax over 12, so its p is Bonferroni-corrected.

| arm | basis | layer | ρ | p (exact) | p (Bonferroni) | null sd |
|---|---|---|---|---|---|---|
| E | mad | **7** | **−0.7461** | 0.00128 | **0.01403** | 0.2582 |
| E | mad | 2 | +0.4910 | 0.00064 | 0.00702 | 0.2015 |
| F | mad | 8 | −0.5901 | 0.00702 | 0.07717 | 0.2798 |

**TASK-07 found this same relationship on the 300-epoch runs at layers 7–8,
negative, ρ = −0.518…−0.594, Bonferroni-significant in 4/4 repeats**
(`results/notes/sink_address.md`). Arm E here: same sign, same depth, larger
magnitude, on an independent schedule. **This is the strongest result in the
task** — the mechanism claim holds up where the accuracy claim could not be
resolved at n = 1.

Arm E also carries a *significant positive* correlation at layer 2 (+0.4910,
Bonferroni 0.0070) — reported because TASK-07's convention is to report the
opposite-sign extreme too, not only the headline one.

Three limits, all in the note:
- **one seed**;
- **one basis only.** The canon-basis address map is *flat* — freq exactly
  1.0 at all 196 positions, the same saturation as §5.4 — so TASK-07's
  requirement that a result hold in **both** bases cannot be checked here;
- a correlation is an alignment, not a demonstration that the alignment is
  what produces the accuracy.

Arms B and C get **no correlation value at all**, deliberately: their gate is
constant across positions by construction, so ρ is *undefined*, not zero.
Printing 0.0 would read as "the non-spatial gate ignores the address", which
would be a statement about arithmetic rather than about the model. Layer 11 of
E and F is likewise blank — that layer never moved from its init.

---

## 6. What we did NOT do

1. **Phase D — not started.** Its condition **is met** (E's margin over the
   better of {C, D} is +0.284 against an arm-to-arm spread of 0.720), and the
   note records that. No seeds were prepared: the task file puts that decision
   with the human after reading the note. Preparing it means 2 arms × 2 seeds
   (E+C for the margin; E+F if the init disagreement is also to be resolved)
   at ~6.5 h per run.
2. **No τ was recalibrated** for this cell (§5.4), so the sink question stands
   open by choice, not by oversight.
3. **No ViT-B, no true-nomix, no other recipe.** One cell only.
4. **No 300-epoch arms.** Cross-schedule comparisons of absolute top-1 are not
   meaningful and the note says so.
5. **`warmup_epochs` was not retuned** — it stays at 20, i.e. 20 % of this
   schedule against 6.7 % of the headline's. Retuning would have made the arms
   incomparable to `e2r_vits_mixup_*` in a second, uncontrolled way.
6. **Arm D deviates from the gate in two documented ways** (both deliberate,
   both in `configs/abl_matrix.yaml`): standard LayerScale scales *every*
   token including CLS, which the gate never does; and its γ carries the same
   weight decay φ does, where timm would exclude a 1-dim γ.
7. **Nothing was pushed by Claude Code.** Every push in this task was the
   human's.

---

## 7. Code: what was added or changed

| file | what |
|---|---|
| `saga/gate.py` | `gate_mode` ∈ {none, const, headscalar, layerscale, spatial}; `LayerScaleGate`; `build_gate()` — the single place a mode becomes a module. `spatial` is the shipped gate unchanged |
| `saga/vit.py` | `build_saga_vit(gate_mode=…, layerscale_init=…)`; `None` derives the mode from `gate`, so every pre-TASK-12 caller builds exactly what it built before; a gate/gate_mode contradiction is refused |
| `classification/tools/train.py` | gate wiring; matrix-wide `overrides`; gate parameter counts printed; `--ckpt_root`; `gate_init_logit` refuses to be a silent no-op or to thaw the frozen arm; a zero-duration epoch no longer crashes the logger |
| `tools/model_factory.py`, `tools/eval.py`, `tools/diagnose.py` | `--gate-mode` passthrough, without which a headscalar or layerscale checkpoint cannot be strict-loaded |
| `tools/derive_runs.py` | `ckpt_dir_for()` — follows each run's `meta.json["ckpt_dir"]`, since the checkpoints are off the repo filesystem |
| `scripts/gen_slurm_chain.py` | templates the matrix path and the optional `--ckpt_root`; the e2r job files stay byte-identical |
| `configs/abl_matrix.yaml` | the six arms, the shared schedule, `ckpt_root`, and the recorded design decisions |
| `scripts/jobs/abl_smoke.sbatch` | 2 epochs of all six arms in one job |
| `scripts/jobs/abl_derive.sbatch` | the Phase-C derivation |
| `tools/check_abl_smoke.py` | PASS/FAIL per acceptance item, read from the smoke's own files |
| `analysis/build_ablation_tables.py` | → `T2_ablation.csv` |
| `analysis/ablation_address.py` | → `T2_ablation_address.csv` |
| `analysis/build_ablation_note.py` | → `results/notes/ablation.md` |
| `plotting/plot_T2.py` | → `results/figures/F_ablation_draft.pdf` |
| `tests/test_task12_ablation.py` (51) · `tests/test_task12_phasec.py` (16) | the suite |

### Regenerating every artifact (local, CPU, seconds)

```bash
python analysis/build_ablation_tables.py
python analysis/ablation_address.py
python analysis/build_ablation_note.py
python plotting/plot_T2.py
```

They are pure functions of the committed run artifacts; re-running them
rewrites the same bytes.

---

## 8. Provenance

| arm | run_id | checkpoint sha256 | training wall | finished |
|---|---|---|---|---|
| A | `abl_vits_mixup_baseline_s0` | `db707f74a109deb6…` | 6.63 h | 2026-09-14 23:54 |
| B | `abl_vits_mixup_const05_s0` | `9b0fa7dc289288b1…` | 6.60 h | 2026-09-15 00:13 |
| C | `abl_vits_mixup_headscalar_s0` | `09d0fc8470db9f0c…` | 6.45 h | 2026-09-15 02:59 |
| D | `abl_vits_mixup_layerscale_s0` | `9b1c0ff0f5e66970…` | 6.47 h | 2026-09-15 03:01 |
| E | `abl_vits_mixup_spatial_i0_s0` | `d25db85e2576889f…` | 6.53 h | 2026-09-15 03:20 |
| F | `abl_vits_mixup_spatial_i4_s0` | `5ca0077425d1f4db…` | 6.49 h | 2026-09-15 07:40 |

Every run: seed 0, world_size 4, 100 contiguous epochs, **zero resumes**,
cosine reaching `min_lr` 1.00e-06. Verified in the Phase-B gate (160 checks)
and the Phase-C derivation gate (48 checks).

**Commits** (all on `main`, merged via `4d5d095`):

```
94a85cc  phase A: gate modes, matrix, launchers
d5aeece  log: phase A
9849135  fix the smoke checker's epoch-0 phi check; record the wd/init asymmetry
7a47d30  log: six-arm smoke result
161b130  six-arm smoke contract files          (HPC)
11d86ad  --ckpt_root for the classification trainer
08ffa24  log: --ckpt_root, the zero-duration-epoch crash
d04b871  phase B complete: six runs verified; derive_runs follows ckpt_dir
eafbe5f  phase C: tables, address, figure, note
764297f  log: phase C built ahead of the derivation
7bbf96e  phase C filled in: the derivation landed, the metric saturates
a98261b  log: phase C filled in
```

---

## 9. Things that bit us, so they don't bite the next task

1. **A τ calibrated on one schedule does not transfer to another.** This is the
   headline lesson: the canon τ silently saturated on 100-epoch models. Any
   future short-schedule work must check the saturation flag before quoting a
   canon sink count. TASK-02B hit the same class of failure on ViT-B/mixup.
2. **`phi_e000.npz` is dumped AFTER epoch 0 trains**, so it never equals the
   init exactly. My first smoke checker asserted equality and produced a false
   FAIL (the real value, 3.9926 vs 4.0, is one epoch of weight decay).
3. **Weight decay is not neutral between initialisations.** φ = 0 is a fixed
   point; φ = +4 is not. Any future "does init matter" arm needs this stated up
   front, or the answer is confounded.
4. **A smoke and a production run that share a run_id must not share a
   `--ckpt_root`**, or the smoke's 2-epoch `last.pth` is what `--resume auto`
   finds. Recorded as a **live latent hazard in TASK-09's files**:
   `dense_smoke_det.sbatch` and `det_vitb_saga_s1.sbatch` share both the run id
   and `--ckpt_root .../dense_ckpt`. It never bit (those smokes predate the
   flag and were deleted), but re-running that smoke today would seed the
   production run's resume. One line to fix if you want it.
5. **A ViT-S e2r checkpoint pair costs ~1.0 GB, not the 0.53 GB the legacy
   manifest implies.** Six runs = 6.0 GB. `/home/hpc` had 4.7 GB free; the
   checkpoints now go to woody via `--ckpt_root`.
6. **Moving the checkpoints broke `derive_runs`' assumption** that they live in
   the run dir. Fixed via `meta.json["ckpt_dir"]` — but any *other* tool that
   hard-codes `<run_dir>/ckpt` will need the same treatment.
7. **Switching partition is a command-line flag, never a copied job file**
   (`sbatch --partition=a100 --gres=gpu:a100:1 <file>`). The derivation was run
   this way.

---

## 10. Open decisions for the human

1. **Phase D?** Its condition is met. E+C would give the margin a standard
   error; E+F would resolve the init disagreement. ~6.5 h per run, launchers
   not yet prepared.
2. **Re-threshold this cell?** The sink question is answerable with a τ
   calibrated on arm A, as a new key with its own provenance. CPU-only, the
   norms are already on the HPC. Without it the mechanism half stays open.
3. **How to frame §5.2 vs §5.3 in the paper.** The controls lose to the
   spatial gate, but the spatial gate does not beat *no gate* in this cell at
   n = 1. Both facts are in the table and neither should travel without the
   other.
4. **Whether the layer-7 replication (§5.7) carries the mechanism section**
   given it rests on one seed and one metric basis.
5. The TASK-09 smoke/production `ckpt_root` collision above.
