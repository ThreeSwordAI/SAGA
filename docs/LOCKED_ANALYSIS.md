# LOCKED_ANALYSIS.md — the analysis parameters I3/I4/I5 will use

> ## STATUS: **DRAFT — NOT YET FROZEN**
>
> Signed and dated by: *(the human — nobody else)*
> Date frozen: `PENDING`
> Git sha at freeze: `PENDING`
>
> Until those three lines are filled in, nothing in this file is locked and
> **no I3/I4 Phase-B job may run**. Once they are filled in, every value below
> is fixed: a later file REFERENCES this document rather than restating a
> parameter, and a change is a new dated section, never an edit in place.

Drafted for TASK I0 D3 from `docs/SAGA_ICLR2027_FINAL_PLAN.md` §5.5, §6.2 and
§6.6-§6.8. Values the plan leaves open are marked **`DECISION NEEDED`** and
are listed together in §11 so the human can settle them in one pass.

Why this file exists: I3 and I4 are the only experiments in the project that
could be tuned into a result. Fixing the layers, the masks, the ε, the
permutation seeds, the endpoint and the decision rule BEFORE the evaluation
outcomes are inspected is what makes their numbers evidence. I0 computes
nothing on the evaluation split, so nothing below was chosen by looking at
one.

---

## 1. Feature stage for all new patch diagnostics

| | |
|---|---|
| Value | **`DECISION NEEDED` — decided by I2** |
| Candidates | `s11_out`, `s12_pre_norm`, `s12_post_norm`, `hist` |
| Default if I2 is inconclusive | `hist` |

`hist` is an **alias**, resolved once in `saga/frozen/stages.py`, for the
stage the historical diagnostics used: **`s12_pre_norm`**, the output of the
last transformer block *before* the final LayerNorm. That was read out of the
code, not assumed — `saga/metrics.py:287-291` hooks each `model.blocks[i]`
and stores the block's own output, `saga/metrics.py:324-325` takes
`feats[L-1][:, P:, :]`, and `saga/vit.py:238-241` applies `self.norm` only
after the block loop. `tests/test_I0_frozen.py` pins the alias against the
real `compute_diagnostics` hook.

Justification for defaulting to `hist`: every number already in `results/`
was computed there, so a new diagnostic at the same stage is directly
comparable and a stage change cannot be confused with an effect. I2 exists to
say whether a different stage is more informative; until it reports, the
comparable choice wins.

## 2. I3 — layers

| | |
|---|---|
| Value | **paper blocks 8 and 9 = 0-based block indices 7 and 8** |
| Sensitivity | one joint early-through-middle edit, only if needed |

Justification, and the disclosure that goes in the paper verbatim: these two
layers were selected because **prior exploratory correlations highlighted
them** — TASK-07 found the gate's suppression map significantly
anti-correlated with the sink address at layers 7-8 in all four ViT-S/mixup
repeats (`results/tables/sink_address.csv`, Q5). That is a selection made on
*discovery* material, and it is disclosed as such. No layer is selected after
inspecting an evaluation loss (plan §6.6).

## 3. I3 — variants at one layer at a time

1. `original` — the gate's own map, through the replacement module (also the
   bit-exactness control)
2. `mean` — per-head mean μ_h at every position
3. `mean_plus_alpha_delta` with **α = 0.5** — μ_h + 0.5·δ_h(p)
4. **10 fixed position permutations**, shared across heads
5. **10 fixed within-ring permutations**, shared across heads
6. **dihedral set** — `DECISION NEEDED`: all 8 elements of `DIHEDRAL_OPS`, or
   the 3 non-identity rotations plus the horizontal flip? The plan says "a
   small prespecified set of grid rotations/reflections" without fixing the
   size. Default if unanswered: **all 8** (it is 8 forward passes and avoids
   a second selection).
7. **energy-matched edit** (plan §5.5) — the spatial-component control:
   `Y^{π,energy} = Y^μ + (‖D^id‖_F / ‖D^π‖_F) · D^π`, single-layer so the
   incoming activations are identical. A zero denominator is handled
   explicitly and logged.

μ and δ are **per head**; permutations are **shared across heads**
(`saga/frozen/edits.py::build_edited_gate_map`).

## 4. Permutation seeds

| | |
|---|---|
| Value | **`DECISION NEEDED` — the fixed index lists are not yet generated** |
| Rule | one list per (grid, kind), generated once, committed, shared across every checkpoint on the same grid |
| Proposed generator | `numpy.random.RandomState(s).permutation(196)` for `s = 0..9`; within-ring: the same ten `s`, permuting inside each ring of `analysis.address_analysis.border_distance_map` |

Permutation draws are **repeated interventions on a checkpoint, never new
model seeds** (plan §6.6). They are an axis of the condition list, not an
axis of replication, and the units table in §8 says so.

`saga/frozen/edits.py` REFUSES to draw a permutation: `gate_edit` requires an
explicit `perm`, so no code path can quietly sample a fresh one per
checkpoint. The lists must be generated and committed by I3 Phase A, to
`configs/frozen/permutations_14x14.json`.

## 5. I4 — masks

| | |
|---|---|
| Primary mask | the **16 highest-prevalence patch coordinates** of the discovery map, taken **at the input to the edited block** (or the preceding block's output) — never a later final-layer norm |
| Controls | **10 fixed random masks**, ring-matched: the same number of selected coordinates in each boundary ring |
| Cross-method rule | the **baseline-derived** coordinate mask is the primary cross-method comparison on the same grid; each method's own map is a clearly labelled secondary question |
| Overlap | random masks MAY overlap the high-prevalence mask. The overlap is REPORTED; draws are never rejected to amplify contrast |
| Map basis | **`DECISION NEEDED`** — `canon` (fixed τ per `(arch, recipe_actual)`, `results/diagsplit/fixed_thresholds_canon.json`) or `mad` (per-image median+5·MAD)? Both exist in `results/tables/sink_address.csv`. Default if unanswered: **`canon`**, the project's primary basis since TASK-06B |
| Which discovery map | **`DECISION NEEDED`** — the existing `*_addr.json` maps are computed at the LAST block, but I4 needs prevalence at the INPUT TO BLOCK 7/8 (plan §6.7 "temporal alignment"). These maps do not exist yet and must be produced by I1 before I4 Phase B |

Rings come from `analysis.address_analysis.border_distance_map` — THE ring
definition, which TASK-07's answers and TASK-13's ring ablation were both
written against. It is imported, never re-derived
(`saga/frozen/edits.py::ring_of`).

## 6. I4 — ε and energy matching

| | |
|---|---|
| Primary | **ε = 0.10** |
| Sensitivity | **ε = 0.25**, one condition — a stability test, NOT an operating point to optimise |
| Energy matching | per-image, **within checkpoint**: κ_i = ε·min_M ‖M⊙U_i‖_F over the prespecified mask family, α_iM = κ_i/‖M⊙U_i‖_F (plan §5.5) |
| Zero-norm images | receive zero perturbation; their **frequency is reported**. Images are never discarded according to the eventual loss effect |
| Cross-checkpoint | energy is matched WITHIN a checkpoint only. The native update scale and the normalised perturbation size are reported when comparing checkpoints; equal absolute energy across differently scaled representations would introduce its own interpretive problem |

The intervention site is the patch attention output **after any native SAGA
gate and before the attention residual addition**, with the projection bias
excluded from U as §5.5 defines it: the module output is U + b, so the edit
is `(1−α)·U + b`. Prefix rows are excluded using the model's own prefix
count. `saga/frozen/edits.py::receiver_perturbation`; smoke checks 4 and 7.

## 7. Primary endpoint

| | |
|---|---|
| Primary | **paired per-image NLL difference** |
| Secondary | top-1 difference, logit-distribution change, patch norms, spatial maps |
| I4 primary contrast | θ_m = E_i[ Δ_i^{high,m} − (1/R)·Σ_r Δ_i^{ringcontrol(r),m} ] |

A positive θ means attenuation at high-prevalence coordinates worsens loss
more than matched controls. Either sign needs interpretation in the context
of the actual perturbation; neither establishes that high norms are the
mediator.

## 8. Units of analysis

| axis | unit | resampling |
|---|---|---|
| frozen-edit effects | **images** (paired) | image-level bootstrap, **10,000 resamples** |
| training effects | **checkpoints** | as the existing tables do (paired by provenance tag) |
| permutation draws | **repeated interventions on one checkpoint** | never treated as seeds, never as replication |
| ring controls (R = 10) | repeated interventions | averaged within image before the contrast |

Bootstrap seed: **`DECISION NEEDED`** — propose `0`, recorded in every output
JSON alongside the git sha and checkpoint sha256, as every other seeded tool
in this repo does.

## 9. Multiplicity

Declared **primary contrasts only**:

1. I3: `original` vs `permute`, and `permute` vs `permute_within_ring`, at
   each of the two declared layers, on the two FRESH ViT-S/mixup SAGA pairs.
2. I4: θ for the baseline-derived mask, per checkpoint, at ε = 0.10.
3. I4 cross-method: θ(SAGA) − θ(baseline) on matched checkpoints, same
   images, same coordinates.

Everything else — every other layer, ε, architecture, recipe, the legacy
repeats, the dihedral set, the energy-matched variants — is labelled
**exploratory** and reported as such. Correction across the primary set:
**`DECISION NEEDED`** (propose: none, with the three contrasts named in
advance and each reported with its interval, rather than a Bonferroni factor
over a set that would then be tempting to shrink).

## 10. Decision rule

A meaningful effect is claimed only when **all three** hold:

1. the direction is **consistent in both fresh ViT-S/mixup pairs**
   (`e2r_vits_mixup_{baseline,saga}_s1` and `_s2`);
2. the **image-conditional CI excludes zero**;
3. the magnitude is **practically interpretable** — reported in NLL units
   beside the native NLL, not as a significance statement alone.

Legacy and cross-recipe results are shown, never converted into extra
training replication by counting their images.

If the result is merely "any shuffled parameter hurts", it becomes supporting
evidence and I4/I5 carry the paper (plan §6.6).

## 11. Splits

| split | sha256 | n |
|---|---|---|
| discovery (`results/diagsplit/val_diag_split.json`) | `PENDING — Phase B` | 10,000 |
| `results/frozen/splits/calibration.json` | `PENDING — Phase B` | 2,000 |
| `results/frozen/splits/evaluation.json` | `PENDING — Phase B` | 10,000 |
| `results/frozen/splits/sub1k.json` | `PENDING — Phase B` | 1,000 |
| `results/frozen/splits/sub2k.json` | `PENDING — Phase B` | 2,000 |

The splits are built by `tools/build_frozen_splits.py` in Phase B (it needs
an extracted val tree, which on this cluster exists only inside a job). The
four shas it prints go in this table before the freeze.

**The protocol sentence, which must survive into the paper:** the original
full-validation accuracies have already been seen, so this is a *locked
evaluation protocol for the new analyses* — not a completely untouched test
set and not a retrospectively preregistered study.

---

## 12. Every `DECISION NEEDED`, in one list

| # | §  | Question | Default if unanswered |
|---|---|---|---|
| D1 | 1 | Which stage for all new patch diagnostics? | `hist` (= `s12_pre_norm`), pending I2 |
| D2 | 3 | Which dihedral subset — all 8, or a smaller prespecified set? | all 8 |
| D3 | 4 | The ten fixed permutation index lists are not yet generated or committed | `RandomState(0..9).permutation(196)`, committed by I3 Phase A to `configs/frozen/permutations_14x14.json` |
| D4 | 5 | Map basis for the I4 prevalence mask: `canon` or `mad`? | `canon` |
| D5 | 5 | The discovery map at the INPUT to block 7/8 does not exist — the committed `*_addr.json` maps are last-block. I1 must produce it before I4 Phase B | blocks I4; no default |
| D6 | 8 | Bootstrap seed | `0` |
| D7 | 9 | Multiplicity correction across the three primary contrasts | none, with all three named in advance |
| D8 | 11 | The five split shas | filled in Phase B |

D5 is the only one that BLOCKS a work package. The rest can be defaulted; the
defaults are stated above so that defaulting is a visible decision rather
than a silent one.
