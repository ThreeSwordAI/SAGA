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
§6.6-§6.8. Values the plan left open were marked `DECISION NEEDED` and listed
together in §12 so they could be settled in one pass. **Seven of the eight are
now closed** — each says so where it is defined, and §12 records how. **D5 is
the one that remains, and it blocks I4.**

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
| Value | **`s11_out`** — **CLOSED by I2** |
| Candidates | `s11_out`, `s12_pre_norm`, `s12_post_norm`, `hist` |
| Default if I2 is inconclusive | `hist` (not used; I2 reported) |

> D1 = `s11_out`: on the calibration split, in the `vit_small|mixup` cell (n = 4 pairs), the SAGA-vs-baseline gap at the historical stage survives a terminal-gate bypass at 0.68–0.91 of its native size with every sign preserved, so the untrained terminal constant is a contributor but not the main one; the cosine diagnostics, however, retain only 0.09–0.20 of that gap one block earlier. The final-block patch diagnostics are therefore a mixture of what training did and what a readout-invisible constant does, and they are reported with the `term_1.00` and `s11_out` controls beside them, never alone.
>
> Signed Mahfuzur Rahman Chowdhury, 2026-09-16, a120f4f. Decided by `analysis/i2_decision.decide()` on the calibration split;
> the rule was committed before any result existed.

The wording above is the alternative offered in §3(b) of
`results/frozen/I2_terminal/D1_proposal.md`, chosen over the rule's own
branch-3 boilerplate. The VERDICT follows the pre-declared rule exactly — 2
of the 5 primary diagnostics fall below the conventional 0.70 survival cutoff
under the terminal-gate bypass, and the cutoff was not moved. What was
rejected is only the boilerplate's explanatory clause, "substantially a
terminal-gate effect", which the same table contradicts: the bypass retains
0.68–0.91 of every gap with every sign preserved, while the one-block stage
change removes 80–91% of the cosine gaps and leaves rank and counts at
0.73–0.98.

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
6. **dihedral set** — **CLOSED: all 8 elements of `DIHEDRAL_OPS`.** The
   written default, ACCEPTED rather than defaulted into: it is 8 forward
   passes and it avoids a second selection. Signed Mahfuzur Rahman Chowdhury, 2026-09-16, a120f4f.
7. **energy-matched edit** (plan §5.5) — the spatial-component control:
   `Y^{π,energy} = Y^μ + (‖D^id‖_F / ‖D^π‖_F) · D^π`, single-layer so the
   incoming activations are identical. A zero denominator is handled
   explicitly and logged.

μ and δ are **per head**; permutations are **shared across heads**
(`saga/frozen/edits.py::build_edited_gate_map`).

## 4. Permutation seeds

| | |
|---|---|
| Value | **CLOSED — the lists are generated and committed** |
| File | `configs/frozen/permutations_14x14.json`, sha256 (LF-normalised) `75149f36329e480e192e23a25ac39c88b9a77d6b230ad40938d043525ddfa460` |
| Rule | one list per (grid, kind), generated once, committed, shared across every checkpoint on the same grid |
| Position family | `numpy.random.RandomState(s).permutation(196)` for `s = 0..9` |
| Within-ring family | `s = 100..109`; the identity, then the flat indices INSIDE each Chebyshev ring permuted among themselves, rings `k = 0..6` from `analysis.address_analysis.ring_indices` |

Signed Mahfuzur Rahman Chowdhury, 2026-09-16, a120f4f. Generated and pinned by `tests/test_I3_permutations.py`, which is also
the generator: it regenerates the document from the seeds and asserts byte
identity with what is committed, so neither can drift from the other.

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
| Map basis | **CLOSED: `canon`** (fixed τ per `(arch, recipe_actual)`, `results/diagsplit/fixed_thresholds_canon.json`) — the written default, ACCEPTED rather than defaulted into; it has been the project's primary basis since TASK-06B, and `mad` remains available in `results/tables/sink_address.csv` as the clearly labelled secondary. Signed Mahfuzur Rahman Chowdhury, 2026-09-16, a120f4f. |
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

Bootstrap seed: **CLOSED: `0`** — the written default, ACCEPTED rather than
defaulted into. Recorded in every output JSON alongside the git sha and
checkpoint sha256, as every other seeded tool in this repo does, and already
carried on every row of `T_I2c_gaps.csv` and `T_I2e_registers.csv`. Signed Mahfuzur Rahman Chowdhury, 2026-09-16, a120f4f.

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
**CLOSED: none** — the written default, ACCEPTED rather than defaulted into.
The three contrasts are named in advance, above, and each is reported with
its interval, rather than a Bonferroni factor over a set that would then be
tempting to shrink. Signed Mahfuzur Rahman Chowdhury, 2026-09-16, a120f4f.

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
| discovery (`results/diagsplit/val_diag_split.json`) | `0a686340c00846818a857cf4cbf472cbdc035e0a88c20d79483f7d9f463b69b1` | 10,000 |
| `results/frozen/splits/calibration.json` | `6b707eb39f934274a5ea753d613af54dc9aad01510808f246b70506a96003003` | 2,000 |
| `results/frozen/splits/evaluation.json` | `7fdf5f9f2ace98ef03a6267455daa1104b92b6689c510ab8b1e5420340f27014` | 10,000 |
| `results/frozen/splits/sub1k.json` | `6a38d1000b32f2ca0c5bcfd1187cde86258b2d45a4b1e4b47cfd5580fe088ef3` | 1,000 |
| `results/frozen/splits/sub2k.json` | `d2fc8b5b4c5ab40928c1ea861f21a3b561c0654a93479048773802568d338117` | 2,000 |

Built by `tools/build_frozen_splits.py` on the HPC, 2026-09-16 (it needs an
extracted val tree, which on this cluster exists only inside a job — the task
file put this step on a login node, which is not possible here). Every sha
above is read from the file it names; `discovery` is the sha256 of the file
itself, the other four are each split's own recorded `sha256` over its
canonical serialization, which `tools/build_frozen_splits.py` re-verifies on
every invocation. All 1,000 classes are present in each; calibration is
2/class and evaluation 10/class exactly; `discovery`, `calibration` and
`evaluation` are pairwise disjoint and `sub1k`/`sub2k` are subsets of
`evaluation`. **D8 is closed.**

**The protocol sentence, which must survive into the paper:** the original
full-validation accuracies have already been seen, so this is a *locked
evaluation protocol for the new analyses* — not a completely untouched test
set and not a retrospectively preregistered study.

---

## 12. Every decision, and where it stands

| # | §  | Question | Resolution |
|---|---|---|---|
| ~~D1~~ | 1 | ~~Which stage for all new patch diagnostics?~~ | **CLOSED — `s11_out`**, decided by I2's pre-declared rule on the calibration split. Signed Mahfuzur Rahman Chowdhury, 2026-09-16, a120f4f. |
| ~~D2~~ | 3 | ~~Which dihedral subset — all 8, or a smaller prespecified set?~~ | **CLOSED — all 8**, default accepted. Signed Mahfuzur Rahman Chowdhury, 2026-09-16, a120f4f. |
| ~~D3~~ | 4 | ~~The ten fixed permutation index lists are not yet generated or committed~~ | **CLOSED** — `configs/frozen/permutations_14x14.json`, sha256 `75149f36329e480e192e23a25ac39c88b9a77d6b230ad40938d043525ddfa460`. Signed Mahfuzur Rahman Chowdhury, 2026-09-16, a120f4f. |
| ~~D4~~ | 5 | ~~Map basis for the I4 prevalence mask: `canon` or `mad`?~~ | **CLOSED — `canon`**, default accepted. Signed Mahfuzur Rahman Chowdhury, 2026-09-16, a120f4f. |
| D5 | 5 | The discovery map at the INPUT to block 7/8 does not exist — the committed `*_addr.json` maps are last-block. I1 must produce it before I4 Phase B | **OPEN** — blocks I4; no default. I1 owns it |
| ~~D6~~ | 8 | ~~Bootstrap seed~~ | **CLOSED — `0`**, default accepted. Signed Mahfuzur Rahman Chowdhury, 2026-09-16, a120f4f. |
| ~~D7~~ | 9 | ~~Multiplicity correction across the three primary contrasts~~ | **CLOSED — none**, all three named in advance; default accepted. Signed Mahfuzur Rahman Chowdhury, 2026-09-16, a120f4f. |
| ~~D8~~ | 11 | ~~The five split shas~~ | **CLOSED** — filled in from the Phase-B build, 2026-09-16 |

**One remains open: D5**, and it is the one that BLOCKS a work package — I4
cannot run until I1 produces the prevalence map at the INPUT to block 7/8.
D2, D4, D6 and D7 were closed at their WRITTEN DEFAULTS, recorded above as
"default accepted" rather than allowed to pass silently; D1 was decided by
I2's pre-declared rule, and D3 by generating and committing the lists.

**This document is still a DRAFT.** Closing a decision and freezing the
document are separate acts: the three signature lines in the header stay
`PENDING` while D5 is open.
