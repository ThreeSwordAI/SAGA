# Framing memo — the ICLR 2027 paper, built on what actually survived

Written 2026-09-13, after Gate 2. This replaces Part III of the original master plan, which was written before the evidence existed. Everything here is keyed to files that exist today.

---

## 1. The story, in one paragraph

> Supervised ViTs place their high-norm attention sinks at a **specific, stable address**: a ring one patch inside the image border, reproducible across seeds (ρ = 0.91) and across models trained months apart with different code. We show this address can be targeted directly. SAGA — a 14K-parameter, input-independent, per-head positional gate on the attention output — learns a suppression map that is significantly anti-correlated with the sink address at mid depth (layers 7–8, every repeat), removing sinks **without relocating them**: unlike register tokens and test-time register surgery, which both move the sink pattern elsewhere, SAGA empties the address in place while the residual sink mass consolidates into the ungated CLS token. The result is a small but seeded and recipe-robust accuracy gain (+0.46 ± 0.21 on ViT-S/16, n = 4), a large and universal reduction in oversmoothing (25–40% in every cell and repeat), and a transfer profile that follows the mechanism rather than contradicting it: gains where the suppressed positions carry no class evidence, costs where they do.

## 2. Title

Primary: **Attention Sinks Have an Address: Positional Gating in Vision Transformers**
Alternates: *Where Sinks Live: A 14K-Parameter Positional Gate for Vision Transformers* · *Emptying the Address: Sink Suppression Without Relocation*

Avoid anything promising accuracy ("Better ViTs…") or recipe contrasts. The address is the asset.

## 3. Contributions (write these first; every experiment serves one)

1. **A measurement contribution.** Sink counting is threshold-dependent in ways that invert conclusions; we give a calibrated primary metric, a full robustness sweep, finite-sample nulls for concentration statistics, and a spatial permutation null for map correlations. (Nobody in this literature does this, and it is the part a reviewer can verify instantly.)
2. **A phenomenon.** The sink address: concentrated, seed-stable, recipe-dependent, locatable on ring 1.
3. **A method.** SAGA: 14K parameters, no extra tokens, targets the address (quantified, not asserted), improves both diagnostic axes, and beats both register-based alternatives on accuracy.
4. **An honest transfer characterization.** Where positional suppression helps and where it costs, with a mechanism-level explanation and a direct test.

## 4. Page budget (9 pages)

| §, pages | Content | Source |
|---|---|---|
| 1 Intro + teaser, 1.25 | The address in one figure; contributions | F1 |
| 2 Related work, 0.75 | Registers, test-time registers, LLM gated attention, sinks-are-necessary theory | — |
| 3 Method, 1.0 | Gate, placement, init, cost; explicitly *not* content-conditioned | F2 |
| 4 Measuring sinks, 1.0 | Two axes; threshold dependence; the nulls | T8, A1 |
| 5 The address, 1.5 | Existence, ring-1 location, seed stability, recipe dependence; SAGA preserves-and-empties vs registers/TTR relocate; gate–address correlation | **F3, F6, T_addr** |
| 6 Main results, 1.5 | Classification (n=4 significant), oversmoothing everywhere, consolidation at B; registers and TTR as baselines | T1, T_ttr, F4 |
| 7 Transfer, 1.25 | Fine-grained (both directions), COCO (AP up, AP_S down, with the per-category distribution), ADE20K; the ring-ablation test | T3, T4, T5, F7 |
| 8 Limitations + conclusion, 0.75 | n=1 dense cells; undecided ViT-B gate link; modest effect sizes; scope | — |

**Section 5 is the paper.** It is the only section no competing work can produce, and it is where the strongest statistics live. Do not let Section 6 become the center of gravity — a +0.46 headline cannot carry an ICLR paper, but it corroborates one nicely.

## 5. Tables

| ID | § | Content | Source file |
|---|---|---|---|
| T1 | 6 | Classification: cells × variants, repeats listed, paired Δ, SE, significance | `e2_pooled.csv` |
| T2 | 6 | **Matched-init ablation** (pending) | — |
| T_ttr | 6 | TTR vs baseline vs SAGA, with paired CIs | `T_ttr.csv` |
| T_addr | 5 | Address: concentration vs null, seed ρ, cross-recipe ρ, variant ρ, gate–address ρ | `sink_address.csv` |
| T3 | 7 | Fine-grained, 3 seeds ViT-S, ViT-B fill pending | `T3_finegrained.csv` |
| T4 | 7 | COCO, all six APs + per-category summary | `T4_coco.csv` |
| T5 | 7 | ADE20K + background rows | `T5_ade20k.csv` |
| T8 | App | Threshold sweep (8 definitions × cells) | `sink_robustness*.csv` |
| T9 | App | Per-category AP_S and per-class IoU distributions | appendix CSVs |
| T10 | App | Hyperparameters, seeds, statistics protocol, run-id map | `REPRODUCE.md` |

## 6. Figures

**F1 teaser** — three rows, columns Input | Baseline | Registers | TTR | SAGA, each attention panel captioned with the localization score averaged over the 200-image probe set (never the displayed images). Bottom strip: baseline sink-frequency map, the ring-1 mask, SAGA's layer-7/8 gate map. *Source: TASK 11 Phase C.*
**F2 method** — gate placement + one learned gate map. *Source: `phi_e299.npz`.*
**F3 the address** — 14×14 frequency maps per repeat showing seed stability, plus the ring profile; the figure that proves the phenomenon. *Source: `Faddr.npz`.*
**F4 two-axis scatter** — sink (canon τ) vs oversmoothing, one point per run, colored by variant. Shows SAGA improving both and registers trading. *Source: `e2_pooled.csv`.*
**F6 relocation** — address correlation vs baseline for SAGA / registers / TTR, plus CLS-norm ratio across depth. *Source: `sink_address.csv`, diag JSONs.*
**F7 dense qualitative** — ADE20K predictions and COCO small-object crops. *Exists.*
**Appendix:** A1 norm histograms + threshold sweep · A2 full gate atlas · A3 per-category AP_S distribution (the honest version of the −0.844) · A4 teaser extras and failure cases.

## 7. How to present the three uncomfortable results

**AP_S −0.844.** Report the aggregate first, then the distribution: SAGA is ahead in 43/80 categories with median +0.055; the aggregate is carried by a few sparse categories (toaster −34, bed −20, bear −17); the within-run spread over the final evaluations is 0.39–0.76, the same order as the delta; n = 1. Then state the mechanism reading: small objects disproportionately occupy the border region that the gate suppresses, which is the same trade-off the fine-grained results show. Do not claim significance in either direction.

**CUB −0.36 vs Aircraft +0.96.** Present as a *pair*, not two separate results. This is the cleanest instance of the trade-off: Aircraft's background (sky) carries no class evidence; CUB's (habitat) does. If TASK 13's ring ablation confirms that masking ring 1 costs CUB more than Aircraft, this becomes a demonstrated mechanism and moves to Section 7's opening. If it doesn't, keep it as a reported limitation with the hypothesis marked untested.

**ViT-B gate–address undecided.** Say it plainly in Section 5: the gate–address correlation is significant in all four ViT-S/mixup repeats and undecided at ViT-B, where SAGA instead shows consolidation (fewer corrupted positions, larger extremes). Two repeats cannot settle it. This is a limitation, not a contradiction — and consolidation is itself a finding.

## 8. Reviewer attacks and answers

1. *"Isn't this just LayerScale / a favourable rescaling?"* — **Currently unanswered. This is the one gap that can sink the paper.** The matched-init ablation (frozen-0.5, per-head scalar, LayerScale, spatial at init 0.5 and +4) is the answer; run it.
2. *"Why not test-time registers — it's free?"* — Answered: below SAGA in all four cells (−0.22 to −1.23), with paired CIs on the same footing, and it relocates the address rather than emptying it.
3. *"Registers don't hurt accuracy in Darcet."* — Ours do, in every measured cell, with corrected metrics and provenance; we also report that registers worsen oversmoothing. Cite the 2026 cross-architecture reassessment.
4. *"+0.46 is small."* — Agreed, and it is not the contribution; the address, the measurement methodology, and the mechanism are. It is seeded, paired, and significant, which is more than most papers in this line offer.
5. *"Baselines are ~1 pt under DeiT."* — drop_path = 0 across all variants; deltas are matched-training. The strong-recipe pair (queued) gives one appendix row.
6. *"Sinks are provably necessary; suppressing them should hurt."* — We don't delete the function: it consolidates and the CLS norm ratio rises in 4/4 cells. That's the relocation analysis, and it is consistent with the theory.
7. *"Dense results are weak."* — Reported honestly with distributions and n = 1 stated; overall AP is up (+0.43, 57/80 categories), background IoU up, AP_S down. We characterize rather than oversell.
8. *"One recipe."* — True-nomix cell exists and the gain persists (+0.33/+0.32); the address itself is recipe-dependent, which we report as a finding, not a caveat.

## 9. What must happen before submission

**Blocking:** the matched-init ablation (attack 1). **High value:** TASK 13 ring ablation (converts three mixed results into one mechanism), TASK 11 Phase C teaser (F1 doesn't exist yet), ViT-B fine-grained seed fill. **Then:** collectors/plotters, REPRODUCE.md, and the writing itself. **Optional if compute frees:** detection seed 2, seeded registers runs, strong-recipe pair, IN-C/R.

## 10. Honest assessment

Odds at ~35–40%, and the two levers that move it are both cheap. The ablation moves it because it closes the last clean attack. The ring ablation moves it because a paper whose mechanism *predicts its own failure cases* reads as mature rather than mixed — and right now you have three results (CUB, AP_S, the ring's location) that a single experiment could unify. The paper's real strength is that it is unusually hard to attack on rigor: corrected metrics, permutation nulls, paired CIs, provenance on every number, and negative results reported with their distributions. That is not a consolation prize — for this literature, it is a differentiator.
