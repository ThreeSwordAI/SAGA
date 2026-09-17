# TASK C handoff — the correspondence readout (I5) and incoming attention (I7)

**GENERATED** by `analysis/build_C_handoff.py` at git `ecd1d0ce30962136ce6d7ea695868035f23a93c0`. Every number is read from a committed file under `results/frozen/I5_readout/`, `results/frozen/I7_attention/` or `results/ttr*/`; none is typed by hand. The surrounding prose is fixed text. Re-run the script after new results land and this document updates itself.

## 0. The parameter declaration, as it actually stands

**D9 present in `docs/LOCKED_ANALYSIS.md`: NO. D10 present: NO. Document frozen: NO.**

This has to be stated plainly, because it is the difference between a pre-registered analysis and an ordinary one:

- The PARAMETER VALUES were fixed in committed code before any Phase B job ran. `configs/frozen/I5_readout.yaml` and `configs/frozen/I7_attention.yaml` carry the transforms, stages, descriptors, blocks, alignment, bootstrap seed and resample count, and `saga/frozen/runner.py` REFUSES anything not in them. That is verifiable from the git history.
- The D9/D10 lines were NOT added to `docs/LOCKED_ANALYSIS.md`, and the human's signature — the act that makes that document authoritative — did not happen before the runs.

So: the analysis configuration was fixed in advance and can be proved to have been, but this is **not** a signed pre-registration. The paper should say the former and must not claim the latter.

## 1. I7 — the wording rule (D10), and what it returned

**Verdict: `attention_sink`** — the paper's term is **attention sink**.

The sentence `analysis/i7_wording.py` returned, COPIED verbatim and never edited:

> Across both fresh ViT-S/mixup baseline checkpoints and both measured blocks, patch queries send at least 6.15x more attention mass to high-norm outlier tokens than to other tokens (smallest bootstrap CI lower bound 6.02, above the pre-declared and conventional threshold of 3x), so this paper calls these tokens attention sinks.

The rule, as it was declared before the measurement existed: D10: the paper uses 'attention sink' only if, in every fresh ViT-S/mixup baseline, at both blocks, for patch queries, mean incoming mass per exceedance token is at least 3x that per non-exceedance token with the bootstrap CI excluding 3. The 3x is conventional.

The four required cells — both fresh ViT-S/mixup baselines, both blocks, patch queries:

| checkpoint | block | ratio | CI low | CI high |
|---|---|---|---|---|
| `e2r_vits_mixup_baseline_s1` | 10 | **6.95** | 6.79 | 7.11 |
| `e2r_vits_mixup_baseline_s1` | 11 | **13.75** | 13.33 | 14.18 |
| `e2r_vits_mixup_baseline_s2` | 10 | **6.15** | 6.02 | 6.29 |
| `e2r_vits_mixup_baseline_s2` | 11 | **11.85** | 11.46 | 12.25 |

Smallest ratio 6.15, smallest CI lower bound 6.02, against the conventional threshold 3.0. Every cell passed.

## 2. I5a — the correspondence readout

Primary cut: D1 stage `s11_out`, primary descriptor `l2`, `native`. **Every I5 comparison is a SECONDARY endpoint under D7.**

Chance is `0.005102` (1/196) and is a column on every readout table. `T0` is the identity transform and scores 1.0000 on every checkpoint — the matcher's ceiling, and a control rather than a result.

| checkpoint | variant | T1 flip | T2 (1 patch) | T3 (2 patches) |
|---|---|---|---|---|
| `e2r_vitb_mixup_baseline_s1` | baseline | 0.8026 | 0.8383 | 0.8106 |
| `e2r_vits_mixup_baseline_s1` | baseline | 0.9058 | 0.8981 | 0.8882 |
| `e2r_vits_mixup_baseline_s2` | baseline | 0.8956 | 0.9040 | 0.8867 |
| `e2r_vits_nomix_baseline_s1` | baseline | 0.9913 | 0.9686 | 0.9644 |
| `e2r_vits_nomix_baseline_s2` | baseline | 0.9954 | 0.9617 | 0.9554 |
| `legacy_e2_vit_base_nomixdir_baseline` | baseline | 0.8490 | 0.8696 | 0.8478 |
| `legacy_e2_vit_small_mixupdir_baseline` | baseline | 0.8929 | 0.9051 | 0.8945 |
| `legacy_e2_vit_small_nomixdir_baseline` | baseline | 0.8988 | 0.9039 | 0.8869 |
| `legacy_e2_vit_base_nomixdir_registers` | registers | 0.8573 | 0.8726 | 0.8538 |
| `legacy_e2_vit_small_mixupdir_registers` | registers | 0.9621 | 0.9569 | 0.9449 |
| `legacy_e2_vit_small_nomixdir_registers` | registers | 0.9630 | 0.9587 | 0.9468 |
| `e2r_vitb_mixup_saga_s1` | saga | 0.8496 | 0.8795 | 0.8663 |
| `e2r_vits_mixup_saga_s1` | saga | 0.8837 | 0.9096 | 0.8958 |
| `e2r_vits_mixup_saga_s2` | saga | 0.9060 | 0.9091 | 0.8958 |
| `e2r_vits_nomix_saga_s1` | saga | 0.9865 | 0.9726 | 0.9682 |
| `e2r_vits_nomix_saga_s2` | saga | 0.9945 | 0.9744 | 0.9697 |
| `legacy_e2_vit_base_nomixdir_saga` | saga | 0.8468 | 0.8832 | 0.8661 |
| `legacy_e2_vit_small_mixupdir_saga` | saga | 0.9124 | 0.9174 | 0.9000 |
| `legacy_e2_vit_small_nomixdir_saga` | saga | 0.8712 | 0.9084 | 0.8902 |

Across T1-T3 the readout runs at **157x to 195x chance**, so it is a measurement with room to move in either direction rather than a ceiling or a floor.

## 3. I5 — the method contrast (T_I5b_methods), at `s11_out`

Paired by seed where seeds match; legacy pairs are labelled `pair_kind=legacy` and are NOT seed pairs. SECONDARY under D7, and worded as a measurement, not a claim.

| contrast | pair | kind | T1 | T2 | T3 |
|---|---|---|---|---|---|
| registers_minus_baseline | `legacy_e2_vit_base_nomixdir_registers − legacy_e2_vit_base_nomixdir_baseline` | legacy | 0.0083 | 0.0030 | 0.0060 |
| registers_minus_baseline | `legacy_e2_vit_small_mixupdir_registers − legacy_e2_vit_small_mixupdir_baseline` | legacy | 0.0692 | 0.0518 | 0.0503 |
| registers_minus_baseline | `legacy_e2_vit_small_nomixdir_registers − legacy_e2_vit_small_nomixdir_baseline` | legacy | 0.0642 | 0.0548 | 0.0599 |
| saga_minus_baseline | `e2r_vitb_mixup_saga_s1 − e2r_vitb_mixup_baseline_s1` | seed | 0.0470 | 0.0412 | 0.0558 |
| saga_minus_baseline | `e2r_vits_mixup_saga_s1 − e2r_vits_mixup_baseline_s1` | seed | -0.0221 | 0.0115 | 0.0076 |
| saga_minus_baseline | `e2r_vits_mixup_saga_s2 − e2r_vits_mixup_baseline_s2` | seed | 0.0104 | 0.0052 | 0.0092 |
| saga_minus_baseline | `e2r_vits_nomix_saga_s1 − e2r_vits_nomix_baseline_s1` | seed | -0.0048 | 0.0040 | 0.0039 |
| saga_minus_baseline | `e2r_vits_nomix_saga_s2 − e2r_vits_nomix_baseline_s2` | seed | -0.0010 | 0.0127 | 0.0143 |
| saga_minus_baseline | `legacy_e2_vit_base_nomixdir_saga − legacy_e2_vit_base_nomixdir_baseline` | legacy | -0.0022 | 0.0136 | 0.0183 |
| saga_minus_baseline | `legacy_e2_vit_small_mixupdir_saga − legacy_e2_vit_small_mixupdir_baseline` | legacy | 0.0195 | 0.0124 | 0.0054 |
| saga_minus_baseline | `legacy_e2_vit_small_nomixdir_saga − legacy_e2_vit_small_nomixdir_baseline` | legacy | -0.0276 | 0.0044 | 0.0032 |

LOCKED_ANALYSIS §10 asks first whether a direction is CONSISTENT across the fresh ViT-S/mixup pairs. For `saga_minus_baseline` on those pairs:

- **T1**: -0.0221, +0.0104 — NOT consistent in sign
- **T2**: +0.0115, +0.0052 — consistent in sign
- **T3**: +0.0076, +0.0092 — consistent in sign

## 4. I5 — can a feature consumer see the terminal constant? (T_I5c_terminal)

I2 measured that a CLS-only classifier cannot see the terminal patch gate at all: max abs logit difference exactly **0** on 8 of 8 SAGA checkpoints. I5 asks the same question of a consumer that reads the patches. `s11_out` is the control — the bypass swaps the LAST block's gate, so that stage cannot move.

| checkpoint | stage | native | `term_1.00` | delta | CI |
|---|---|---|---|---|---|
| `e2r_vitb_mixup_saga_s1` | `hist` | 0.8396 | 0.8296 | -0.00994 | [-0.01044, -0.00945] |
| `e2r_vitb_mixup_saga_s1` | `s11_out` | 0.8496 | 0.8496 | 0.00000 | [0.00000, 0.00000] |
| `e2r_vitb_mixup_saga_s1` | `s12_post_norm` | 0.8542 | 0.8496 | -0.00458 | [-0.00489, -0.00427] |
| `e2r_vits_mixup_saga_s1` | `hist` | 0.8832 | 0.8806 | -0.00254 | [-0.00281, -0.00228] |
| `e2r_vits_mixup_saga_s1` | `s11_out` | 0.8837 | 0.8837 | 0.00000 | [0.00000, 0.00000] |
| `e2r_vits_mixup_saga_s1` | `s12_post_norm` | 0.8891 | 0.8869 | -0.00224 | [-0.00249, -0.00199] |
| `e2r_vits_mixup_saga_s2` | `hist` | 0.9034 | 0.9020 | -0.00144 | [-0.00166, -0.00122] |
| `e2r_vits_mixup_saga_s2` | `s11_out` | 0.9060 | 0.9060 | 0.00000 | [0.00000, 0.00000] |
| `e2r_vits_mixup_saga_s2` | `s12_post_norm` | 0.9080 | 0.9070 | -0.00102 | [-0.00122, -0.00082] |
| `e2r_vits_nomix_saga_s1` | `hist` | 0.9830 | 0.9833 | 0.00030 | [0.00016, 0.00044] |
| `e2r_vits_nomix_saga_s1` | `s11_out` | 0.9865 | 0.9865 | 0.00000 | [0.00000, 0.00000] |
| `e2r_vits_nomix_saga_s1` | `s12_post_norm` | 0.9856 | 0.9857 | 0.00010 | [-0.00000, 0.00019] |
| `e2r_vits_nomix_saga_s2` | `hist` | 0.9933 | 0.9933 | 0.00006 | [-0.00003, 0.00015] |
| `e2r_vits_nomix_saga_s2` | `s11_out` | 0.9945 | 0.9945 | 0.00000 | [0.00000, 0.00000] |
| `e2r_vits_nomix_saga_s2` | `s12_post_norm` | 0.9940 | 0.9941 | 0.00008 | [0.00002, 0.00014] |
| `legacy_e2_vit_base_nomixdir_saga` | `hist` | 0.8369 | 0.8257 | -0.01121 | [-0.01163, -0.01079] |
| `legacy_e2_vit_base_nomixdir_saga` | `s11_out` | 0.8468 | 0.8468 | 0.00000 | [0.00000, 0.00000] |
| `legacy_e2_vit_base_nomixdir_saga` | `s12_post_norm` | 0.8512 | 0.8454 | -0.00574 | [-0.00605, -0.00544] |
| `legacy_e2_vit_small_mixupdir_saga` | `hist` | 0.9090 | 0.9054 | -0.00357 | [-0.00385, -0.00329] |
| `legacy_e2_vit_small_mixupdir_saga` | `s11_out` | 0.9124 | 0.9124 | 0.00000 | [0.00000, 0.00000] |
| `legacy_e2_vit_small_mixupdir_saga` | `s12_post_norm` | 0.9143 | 0.9116 | -0.00270 | [-0.00295, -0.00245] |
| `legacy_e2_vit_small_nomixdir_saga` | `hist` | 0.8669 | 0.8630 | -0.00394 | [-0.00427, -0.00362] |
| `legacy_e2_vit_small_nomixdir_saga` | `s11_out` | 0.8712 | 0.8712 | 0.00000 | [0.00000, 0.00000] |
| `legacy_e2_vit_small_nomixdir_saga` | `s12_post_norm` | 0.8775 | 0.8739 | -0.00365 | [-0.00395, -0.00335] |

The control holds on the real records: `max_abs_s11_diff_vs_native` is ['0'] at `s11_out`, and the measured delta there is exactly 0 for every checkpoint.

**The answer: yes, and by a small amount.** At `hist` and `s12_post_norm` the bypass moves the readout by at most **0.0112** in accuracy — a real, interval-excluding difference on most checkpoints, but one to two orders of magnitude smaller than the readout itself. A classifier sees nothing; this consumer sees a little.

## 5. I5b — the frozen ADE20K heads

**I5b: RUN.** Gate: matched at epochs_completed=80: baseline=seg_vitb_baseline_s1, saga=seg_vitb_saga_s1.

| run / condition | mIoU_ss | pixel acc | images |
|---|---|---|---|
| `seg_vitb_baseline_s1/native` | 43.5768 | 79.8750 | 2000 |
| `seg_vitb_saga_s1/native` | 43.3350 | 79.7288 | 2000 |
| `seg_vitb_saga_s1/term_1.00` | 41.1391 | 78.6539 | 2000 |

`T_I5f_seg`: SAGA − baseline is **-0.2418 mIoU_ss** at the dataset level; the paired per-image difference is -0.00863 [-0.01297, -0.00431] over 2000 images (secondary).

`T_I5g_seg_terminal`: under the terminal-gate bypass the SAGA head goes 43.3350 → **41.1391 mIoU_ss**, a change of **-2.1959** (descriptive).

This is EXPLORATORY and must be read as such: the head was TRAINED on gated features, so a drop under the bypass measures sensitivity to an input distribution it never saw. It is not evidence that the gated features are better.

## 6. I7 — incoming attention at the exceedance positions

MAD is the PRIMARY basis. Patch queries, both blocks, all ten ViT-S/mixup checkpoints. Ratio = mean incoming mass per exceedance token / mean per other token, as a ratio of means with an image-level bootstrap.

| checkpoint | variant | block | ratio | CI | AUC |
|---|---|---|---|---|---|
| `e2r_vits_mixup_baseline_s1` | baseline | 10 | **7.29** | [7.12, 7.45] | 0.904 |
| `e2r_vits_mixup_baseline_s1` | baseline | 11 | **13.09** | [12.67, 13.53] | 0.987 |
| `e2r_vits_mixup_baseline_s2` | baseline | 10 | **6.51** | [6.36, 6.66] | 0.863 |
| `e2r_vits_mixup_baseline_s2` | baseline | 11 | **11.58** | [11.20, 11.97] | 0.983 |
| `legacy_e2_vit_small_mixupdir_baseline` | baseline | 10 | **8.83** | [8.62, 9.04] | 0.930 |
| `legacy_e2_vit_small_mixupdir_baseline` | baseline | 11 | **12.02** | [11.56, 12.48] | 0.974 |
| `legacy_e2_vit_small_nomixdir_baseline` | baseline | 10 | **7.29** | [7.13, 7.45] | 0.877 |
| `legacy_e2_vit_small_nomixdir_baseline` | baseline | 11 | **15.24** | [14.62, 15.89] | 0.981 |
| `legacy_e2_vit_small_mixupdir_registers` | registers | 10 | **1.54** | [1.46, 1.61] | 0.625 |
| `legacy_e2_vit_small_mixupdir_registers` | registers | 11 | **19.44** | [18.59, 20.33] | 0.975 |
| `legacy_e2_vit_small_nomixdir_registers` | registers | 10 | **1.61** | [1.55, 1.67] | 0.670 |
| `legacy_e2_vit_small_nomixdir_registers` | registers | 11 | **14.92** | [14.33, 15.53] | 0.979 |
| `e2r_vits_mixup_saga_s1` | saga | 10 | **7.50** | [7.32, 7.68] | 0.926 |
| `e2r_vits_mixup_saga_s1` | saga | 11 | **6.67** | [6.52, 6.83] | 0.983 |
| `e2r_vits_mixup_saga_s2` | saga | 10 | **7.20** | [7.04, 7.36] | 0.921 |
| `e2r_vits_mixup_saga_s2` | saga | 11 | **3.02** | [2.96, 3.08] | 0.965 |
| `legacy_e2_vit_small_mixupdir_saga` | saga | 10 | **6.33** | [6.18, 6.49] | 0.940 |
| `legacy_e2_vit_small_mixupdir_saga` | saga | 11 | **3.48** | [3.41, 3.56] | 0.975 |
| `legacy_e2_vit_small_nomixdir_saga` | saga | 10 | **8.16** | [7.94, 8.38] | 0.930 |
| `legacy_e2_vit_small_nomixdir_saga` | saga | 11 | **6.28** | [6.12, 6.43] | 0.975 |

`T_I7b_registers` (DESCRIPTIVE, n = 2 register checkpoints, both legacy): where a register model's patch-query mass lands.

| checkpoint | block | CLS | 4 registers total | per register | patch exceedances |
|---|---|---|---|---|---|
| `legacy_e2_vit_small_mixupdir_registers` | 10 | 0.06137 | 0.26279 | 0.06570 | 0.03459 |
| `legacy_e2_vit_small_mixupdir_registers` | 11 | 0.12357 | 0.47252 | 0.11813 | 0.14292 |
| `legacy_e2_vit_small_nomixdir_registers` | 10 | 0.02050 | 0.28666 | 0.07166 | 0.04420 |
| `legacy_e2_vit_small_nomixdir_registers` | 11 | 0.10718 | 0.36185 | 0.09046 | 0.18651 |

`T_I7c_value_norm` (EXPLORATORY): the value-norm ratio at exceedance vs other positions runs 0.449 to 0.936 across the cell. It decides nothing about the wording rule.

## 7. The TTR operating curve (§6)

Built from committed files only; **no new TTR runs**. The two layer ranges are NOT a crossed grid — they were swept on different neuron grids over different cell sets, and the coverage table says so rather than presenting a cross product nobody intended to run.

| layer range | cells present | cells MISSING | n_neurons swept | points |
|---|---|---|---|---|
| `all` (0,12) | 1 | `e2r_vitb_mixup_baseline_s1`, `e2r_vits_nomix_baseline_s1`, `legacy_vits_baseline` | `0|9|10|11|12|13|14|15` | 8 |
| `midlayer` (3,12) | 4 | — | `0|8|10|12|16|24` | 24 |

**32 points total.** The chosen operating point is marked on 4 cells, all at `n_neurons=24` in the midlayer range. The gate lines are read from each run's own `validate.json` and agree across every file.

## 8. What Track C does NOT settle

- **Correspondence under exact grid-aligned transforms is ONE utility, not utility.** T1-T3 are a flip and two whole-patch translations. They say nothing about scale, rotation, or any transform whose correspondence is not exact — which is precisely why those were excluded rather than approximated.
- **The 3x in D10 is a convention.** It is not derived from anything, and the sentence the wording module returns says so. A different threshold would be a different sentence, and the measured ratios are reported so a reader can apply their own.
- **The register checkpoints are n = 2**, both legacy repeats with no recorded seed. Every register row is labelled DESCRIPTIVE and no register comparison is a paired seed contrast.
- **I5b's `term_1.00` pass is EXPLORATORY.** The head was trained on gated features; a drop under the bypass measures sensitivity to an unseen input distribution, not feature quality.
- **The value-norm ratio and the register-token mass are EXPLORATORY** (§5) and decide nothing.
- **Everything I5 and I7 measure is SECONDARY under D7.** The three primary contrasts of the paper are C1-C3 in Track B. No number in this document decides a primary claim.

