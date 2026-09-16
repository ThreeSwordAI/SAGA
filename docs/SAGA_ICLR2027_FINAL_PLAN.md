# SAGA — ICLR 2027 paper plan using existing models and inference only

**Recommended title:** **Beyond Outlier Counts: Spatial Interventions in Vision Transformers**

**One-line takeaway:** Outlier prevalence, spatial organization, and effects on model predictions are different properties; spatial output gates and register-based interventions let us test their relationship without training another model.

**Revision date:** 16 September 2026.

**Hard constraint:** **No new backbone training, fine-tuning, distillation, optimization of gates, or learned probes.** The existing baseline, SAGA, trained-register, ablation, and downstream checkpoints are the research assets. All additional empirical work is inference, deterministic intervention, measurement, statistical analysis, or figure/table generation.

**This version supersedes the previous training-heavy plan.** It does not require its 81-run program, stronger-recipe retraining, extra seeds, augmentation factorial training, or newly trained segmentation heads.

**Status:** Complete revised research and manuscript plan. Existing results and new calculations are labeled separately from proposed inference experiments. No proposed inference outcome is presented as already measured.

**Reading guide:** Sections 0–2 settle the framing; Section 3 inventories current evidence; Sections 4–7 specify the mathematics and inference work; Sections 8–12 lay out every figure, table, main-paper section, and appendix; Sections 13–16 cover execution, adversarial review, sources, and the acceptance estimate.

---

## 0. The decision in one page

### 0.1 The strongest affordable paper

Build a **mechanistic empirical study around the models already trained**, with SAGA as the central receiver-side intervention and trained/test-time registers as contrasting interventions.

The paper asks:

> When an intervention reduces high-norm patch tokens, what actually changes: the amount of high-norm activity, its spatial distribution, attention allocation, or the model’s use of the affected computation?

The key move is to turn the existing collection of models into a controlled study of **what outlier-removal measurements do and do not explain**. The paper should establish a useful inference-time evaluation protocol and deliver a substantive finding from it.

This is a better fit to the available evidence than claiming that SAGA universally replaces registers or that the project has discovered spatial outliers for the first time.

### 0.2 What we keep

- Every eligible completed baseline and SAGA classification checkpoint.
- The trained-register checkpoints that already exist.
- The six completed 100-epoch ablation arms.
- The existing TTR evaluations and their calibration/provenance.
- Completed fine-grained, COCO, and ADE experiments as scope evidence.
- Gate trajectories and population spatial maps.
- Valid existing accuracy, norm, similarity, and rank results.

The existing training investment remains central. This is not a proposal to discard those models and start again.

### 0.3 What we change from the previous plan

| Previous requirement | Revised decision |
|---|---|
| Train five fresh seeds for every method | Removed; characterize the actual existing cohort |
| Train new content-gate and canonical LayerScale baselines | Removed; cite them and avoid an architecture-superiority claim |
| Train a stronger recipe | Removed; optionally evaluate a public pretrained strong model |
| Train a MixUp/CutMix factorial | Removed; report the existing recipe contrast as a bounded association |
| Train a local-feature probe | Removed; use existing heads or a training-free geometric correspondence evaluation |
| Make an 11-block gate the new primary method | Withdrawn for this submission; retain the trained 12-block SAGA definition and use final-gate bypass as a labeled inference control |
| Treat matched retraining as necessary for the main conclusion | Change the conclusion to a frozen-model intervention claim that inference can establish |
| Publish all experiments | Publish complete relevant families; move secondary families to the appendix; omit invalid or unrelated work with documented reasons |

The final-gate control is still important. However, we will not quietly rename edited checkpoints as a newly trained method or attach historical patch statistics to the edited version.

### 0.4 The three intended contributions

**C1 — Separate prevalence from spatial structure.** Quantify how the existing interventions change absolute high-norm occurrence and the conditional distribution of its coordinates, with scale, occupancy, boundary, and sparse-map controls.

**C2 — Test spatial dependence in frozen models.** Use mean-preserving gate edits and matched receiver-output perturbations to measure whether learned spatial arrangement and high-prevalence coordinates matter to the existing computation.

**C3 — Connect diagnostics to the actual readout.** Establish when norm or spatial changes coincide with classification/dense-output changes, and when they can occur without them. Include the exact terminal-patch-gate control and a useful readout evaluation.

C1 has substantial existing evidence. The architectural basis for one C3 control is exact. The functional strength of C2 and the broader C3 conclusions must come from the new inference results.

### 0.5 What would make this ICLR-level

A strong result would be a clear, replicated explanation of an evaluation failure or a meaningful spatial dependence that simpler metrics miss. For example:

> In the studied checkpoints, two interventions that produce similar reductions in high-norm events have different spatial and functional effects; scale-controlled spatial edits identify the difference, while terminal-only improvements expose a readout-dependent diagnostic failure.

That statement is a **target**, not a result to assume. It must survive the prescribed controls.

The contribution cannot be only “our maps look cleaner,” “our accuracy mean is higher,” or “outliers prefer the border.”

### 0.6 Direct answer: can we claim to be better than registers?

**Not as a general statement with the current evidence.**

The defensible comparison is:

> Spatial output gating is an already-trained, no-extra-token intervention with a different measured tradeoff from the available register models.

We can report that SAGA has higher observed classification accuracy in the matched available pairs, while registers often remove more fixed-threshold exceedances. We cannot infer universal superiority, better dense representations, or that register tokens are unnecessary.

If an inference-only evaluation reveals a clear advantage for SAGA on a prespecified task, report that **specific** advantage with the actual cohort and protocol. Do not turn it into a claim about all ViTs or all register methods.

---

## 1. Novelty: what the literature already establishes and what this paper can add

### 1.1 An important correction to the previous framing

The original register paper already contains population maps of high-norm-token frequency. Appendix A discusses spatial nonuniformity, interpolation-related patterns, and concentration near image borders. Therefore, a title or contribution based on “attention sinks have an address” alone would substantially overlap prior work. [Vision Transformers Need Registers, Appendix A](https://proceedings.iclr.cc/paper_files/paper/2024/file/0b408293619f725fd30162af057e531a-Paper-Conference.pdf).

The earlier plan should have made this overlap explicit. This revision does.

### 1.2 Closest work and the boundary of our claim

| Prior work | What it already contributes | What this paper must add rather than repeat |
|---|---|---|
| [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588), ICLR 2024 | High-norm tokens, spatial distributions, information probes, trained registers | Controlled separation of prevalence, spatial alignment, and functional response in the present intervention cohort |
| [Denoising Vision Transformers](https://arxiv.org/abs/2401.02957), ECCV 2024 | Persistent positional feature artifacts and denoising | A specific high-norm/receiver-output intervention analysis, not another claim that spatial artifacts exist |
| [Quantizable Transformers](https://arxiv.org/abs/2306.12929), 2023 | Gated attention and outlier reduction, including ViT/ImageNet experiments | Input-independent spatial gating as an experimental instrument; no claim to invent gating in vision |
| [Vision Transformers Don't Need Trained Registers](https://arxiv.org/abs/2506.08010), NeurIPS 2025 | Test-time neuron edits that steer outlier locations and construct registers | Comparison with receiver-side output modulation under explicit scale, spatial, and readout controls |
| [Vision Transformers with Self-Distilled Registers](https://arxiv.org/abs/2505.21501), NeurIPS 2025 | Adds registers through post-hoc self-distillation | Related work only unless already-produced official weights can be evaluated; we will not run its training procedure |
| [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708), 2025 | Query-dependent post-attention gating and associated effects | Distinguish static positional modulation from content-dependent nonlinear gating |
| [Value-State Gated Attention](https://arxiv.org/abs/2510.09017), 2025/2026 preprint | Data-dependent value-state gating for extreme-token phenomena | Do not transfer its mechanism or theoretical guarantees to static spatial output gates |
| [DINOv3](https://arxiv.org/abs/2508.10104), 2025 | Modern dense features and a distinction between stable patch norms and deteriorating locality | Do not pretend that norm quality and dense quality were previously assumed identical |
| [Do All Vision Transformers Need Registers?](https://arxiv.org/abs/2603.25803), 2026 preprint | Cross-architecture reassessment of register/outlier claims | A controlled within-checkpoint intervention result, not just another small-model survey |
| [Registers Matter for Pixel-Space Diffusion Transformers](https://arxiv.org/abs/2605.16147), 2026 preprint | Reports benefits from registers in diffusion transformers without patch-token outliers | The broad statement that register utility is not exhausted by outlier suppression is already supported elsewhere; focus on the controlled supervised-ViT question |
| [The Spike, the Sparse and the Sink](https://arxiv.org/abs/2603.05498), 2026 preprint | Separates activation and attention-sink mechanisms in language models | Measure the relevant distinction in the present vision models rather than importing its causal explanation |
| [Deep ViT Features as Dense Visual Descriptors](https://arxiv.org/abs/2112.05814), 2021/2022 | Uses frozen ViT descriptors for training-free visual tasks | A standard basis for the optional correspondence endpoint; that evaluation itself is not our novelty |

The TTR paper also demonstrates functional effects of moving outliers. We must not claim to introduce inference-time spatial intervention or to be the first to link outlier location to behavior. Our potential contribution is the **controlled comparison across receiver modulation and register interventions, including which measurements are informative after removing simpler explanations**.

The literature check includes sources available on 16 September 2026. This is a focused review of the closest mechanisms and claims, not a guarantee that no overlapping work exists.

### 1.3 The research gap

Use this formulation:

> Existing interventions can reduce high-norm patch activity, but their effects are often summarized with quantities that conflate absolute scale, spatial occupancy, and the feature stage being read. For the available supervised ViT cohort, it remains unresolved whether the observed spatial organization and learned output-gate arrangement explain changes in the frozen model’s behavior after controlling for mean attenuation, boundary structure, perturbation size, and terminal readout effects.

This is a concrete empirical question. It does not depend on asserting that all prior work overlooked functionality.

### 1.4 What is and is not novel

**Potentially distinctive after successful inference experiments:**

- A coherent comparison of different intervention mechanisms using the same definitions and checkpoint identities.
- A measured distinction between event suppression and preservation/change of coordinate structure, after a scale-only null.
- A spatial-arrangement result that survives ring and perturbation-energy controls.
- A demonstrated readout-dependent failure of an attractive outlier statistic, with a practical evaluation remedy.
- A compact, reproducible inference protocol that changes how this class of interventions should be assessed.

**Not sufficient as novelty:**

- A sigmoid after attention.
- A border ring in a heatmap.
- A larger effective rank.
- A lower average cosine.
- A few higher classification means.
- The elementary terminal-gate invariance proof by itself.
- Showing that shuffling a learned parameter can hurt a model.
- Rebranding a known test-time register mechanism.

### 1.5 Framing alternatives considered

| Candidate framing | Decision | Reason |
|---|---|---|
| “SAGA is a better replacement for registers” | Reject as the headline | Existing transfer and outlier metrics do not establish dominance |
| “Attention sinks have a fixed address” | Reject as the headline | Prior work already studies position; occurrence is probabilistic and content-dependent |
| “MixUp/CutMix causes spatial sinks” | Reject as the headline | Existing combined-recipe comparisons do not isolate the components or all confounds |
| “SAGA prevents oversmoothing” | Reject as the headline | True no-mixing and some rank results contradict a universal statement |
| “No new training is needed to improve every ViT” | Reject | Existing SAGA checkpoints were trained with gates; inference-only follow-up does not make their original training free |
| **“Beyond outlier counts: spatial interventions and functional response”** | **Select** | Uses the existing models fully and makes new claims that controlled inference can test |

We are not changing the story to make every result favorable. We are choosing a question for which the existing positive, null, and negative observations are scientifically relevant.

---

## 2. Final paper identity, abstract, and introduction

### 2.1 Title and positioning

**Title:** Beyond Outlier Counts: Spatial Interventions in Vision Transformers

**Internal project name:** SAGA.

**Method name in the manuscript:** spatial output gating, with SAGA defined once if desired.

An unrelated linear-attention paper already uses the acronym SAGA. Keep it out of the title to reduce ambiguity. [SAGA: Selective Adaptive Gating for Efficient and Expressive Linear Attention](https://arxiv.org/abs/2509.12817).

**Paper type:** empirical/mechanistic representation study with an intervention protocol and an existing trained method. It is not a universal architecture proposal or a leaderboard paper.

**Primary audience:** researchers interpreting token outliers, attention artifacts, gated transformers, and register interventions.

### 2.2 One-line takeaway

> Reducing high-norm patch counts does not specify how spatial structure or model behavior changed; frozen-model spatial and readout controls make that distinction measurable.

This is the organizing idea. The final abstract must say exactly which distinctions the completed experiments establish.

### 2.3 Evidence-only abstract that is defensible now

> High-norm patch tokens and their spatial organization are established properties of several vision transformers, but changes in outlier counts do not uniquely characterize an intervention’s effects. We study an existing cohort of supervised ViTs with spatial output gating, trained registers, and test-time register edits. In ViT-S under the corrected mixing recipe, spatial gating reduces the mean number of patch norms exceeding a fixed baseline-calibrated threshold from 16.48 to 8.40, while its prevalence map remains strongly correlated with the corresponding baseline maps. The fraction of remaining exceedances in the first interior ring increases, illustrating that lower occurrence and weaker spatial concentration are different outcomes. Across recipes and model scales, absolute and relative outlier measures, feature similarity, and downstream performance do not move uniformly together. We formalize the receiver-side action of the gate and identify a terminal patch gate whose output is invisible to a CLS-only classifier. These findings motivate an inference-only evaluation that separates event prevalence, spatial alignment, and functional response, and cautions against interpreting cleaner patch statistics as sufficient evidence of better representations.

This version uses current evidence and exact architectural reasoning. It should be strengthened with the decisive inference result before submission; alone it risks being too descriptive.

### 2.4 Final abstract structure after inference

Use about 180–210 words:

1. Existing context and unresolved measurement problem.
2. Existing model cohort and the inference-only study design.
3. One quantitative result separating prevalence and spatial shape.
4. The decisive controlled spatial-intervention result.
5. The functional/readout result and scope boundary.
6. A practical conclusion about evaluating interventions.

A result-bearing sentence to fill **only after measurement**:

> At matched [gate means / ring structure / injected update energy], [specified spatial intervention] changes [specified functional endpoint] by [effect and interval], whereas [specified simpler diagnostic] [does or does not track the change].

A second sentence can report the terminal control and the useful readout:

> Terminal patch edits [measured diagnostic effect] with [measured classifier invariance], while [earlier-layer or existing dense-head evaluation] shows [measured response].

Do not assume that spatial shuffling will hurt, that SAGA will win correspondence, or that the final-gate control will account for a particular fraction of its reported improvement.

### 2.5 Introduction: five paragraphs

**Paragraph 1 — A practical interpretive problem.** Researchers use outlier counts and attention maps to assess modifications to ViTs. These are useful measurements, but different operations can change them for different reasons.

**Paragraph 2 — Give prior work its due.** Registers, positional denoising, gated attention, and test-time registers already establish relevant phenomena and interventions. Acknowledge spatial maps in the original register paper.

**Paragraph 3 — State the precise question.** Does the observed change identify less high-norm activity, different coordinates, different attention allocation, or a change in the computation that affects predictions?

**Paragraph 4 — Explain the available experiment.** The project already has baseline, spatial-gate, register, and ablation checkpoints. Holding each checkpoint fixed lets us isolate inference responses without adding new optimization trajectories.

**Paragraph 5 — State completed findings and contributions.** Use the three contributions in Section 0, updated with actual measured effects. The primary intervention conclusion must be visible here, not deferred to a long appendix.

### 2.6 Main claims and forbidden overextensions

| Claim we can pursue | What it does not imply |
|---|---|
| “The learned gate arrangement affects this frozen model’s predictions” | A spatial-gated model is better than every scalar model trained from scratch |
| “High-prevalence coordinates respond differently to a specified perturbation” | High norms themselves are the causal mechanism or the positions are semantically unimportant |
| “Outlier counts and functional responses diverge in these conditions” | Every use of outlier counts in prior work is invalid |
| “The supplied SAGA models have higher observed accuracy in these matched pairs” | A statistically established universal accuracy improvement |
| “The existing transfer results are mixed” | The central inference study has no value |
| “No additional training is used for this study” | SAGA was originally obtained without training |

---

## 3. Existing evidence and how it supports the revised story

### 3.1 The existing classification cohort

The corrected 300-epoch cohort has **19 completed model conditions**:

- Eight baseline checkpoints: four ViT-S mixing, two ViT-S true no-mixing, two ViT-B mixing.
- Eight matching SAGA checkpoints.
- Three trained-register checkpoints: two ViT-S mixing and one ViT-B mixing, all from legacy provenance.

The four displayed TTR evaluations are edits of existing baseline checkpoints, not four new trained models.

There are also six completed 100-epoch ablation arms, 24 corrected fine-grained fine-tuning runs, and the recorded detection/segmentation runs. Do not count evaluation variants or multiple saved epochs as independent training seeds.

The uploaded archive contains tables, configurations, diagnostic summaries, maps, and gate arrays, but not the model-weight files needed for new inference. The user’s saved HPC checkpoints are the inputs to the proposed evaluation code. No new training is required to produce those inputs.

### 3.2 Classification, absolute occurrence, and relative outliers

Accuracy is in percent; differences are percentage points.

| Model / actual recipe | Repeats | Baseline accuracy | SAGA accuracy | Paired mean difference | Fixed-threshold count: baseline → SAGA | MAD count: baseline → SAGA |
|---|---:|---:|---:|---:|---:|---:|
| ViT-S / mixing | 4 | 78.720 | 79.176 | +0.456 | 16.480 → 8.401 | 16.276 → 14.435 |
| ViT-S / true no-mixing | 2 | 73.035 | 73.362 | +0.327 | 4.051 → 1.433 | 3.094 → 2.822 |
| ViT-B / mixing | 2 | 76.925 | 77.363 | +0.438 | 8.404 → 4.534 | 9.921 → 16.929 |

Source: results/tables/e2_pooled.csv, eligible final-checkpoint fp32 rows.

**Role in the paper:** these results motivate separating measurements. They show useful existing observations, not a proven architecture ranking.

The ViT-S mixing differences are −0.126, +0.570, +0.490, and +0.890. Their nominal paired-t 95% interval is approximately [−0.220, +1.132], with p approximately 0.121. The legacy pairing is less controlled than a fresh matched-seed design. We will report the effect and uncertainty rather than the old “2 × SE” significance flag.

No new seeds will be added. The resulting limitation remains explicit.

### 3.3 Geometry does not follow a universal direction

| Model / actual recipe | Pairwise cosine: baseline → SAGA | Effective rank: baseline → SAGA | Key boundary |
|---|---:|---:|---|
| ViT-S / mixing | 0.431440 → 0.301816 | 89.752 → 103.855 | Strongest existing consistent geometry result |
| ViT-S / true no-mixing | 0.123012 → 0.130855 | 131.498 → 133.228 | Mean cosine rises; individual cosine effects have opposite signs |
| ViT-B / mixing | 0.561624 → 0.312294 | 73.446 → 80.938 | One repeat decreases effective rank |

The ViT-S no-mixing cosine changes are −0.033251 and +0.048936. The ViT-B rank changes are approximately −9.390 and +24.374.

These are informative counterexamples to a single “cleaner representation” axis. They belong in the scope of the argument. They are not reasons to discard valid runs.

### 3.4 A useful existing observation: fewer events, persistent spatial organization

For the fixed-threshold maps:

| Condition | Baseline → SAGA event count | Mean baseline/SAGA map correlation | Ring-one share of events: baseline → SAGA |
|---|---:|---:|---:|
| ViT-S / mixing | 16.480 → 8.401 | 0.873 | 36.77% → 42.73% |
| ViT-B / mixing | 8.404 → 4.534 | 0.793 | 47.56% → 51.94% |
| ViT-S / true no-mixing | 4.051 → 1.433 | 0.865 | 24.53% → 23.47% |

Ring one is one patch inside the boundary and contains 44 of 196 positions. Its uniform-area share is about 22.45%.

The event-share values were recomputed during this revision from results/figures_data/Faddr.npz as the mean of each checkpoint’s ring-event fraction. They are new analyses of existing arrays, not new model evaluations.

**What this establishes descriptively:** lower total occurrence does not require a flatter normalized distribution of the remaining events.

**What it does not establish:** a distinct learned mechanism beyond global scaling, harmful concentration, or a transfer of computational function. The scale-only and occupancy controls below are necessary.

This observation is a stronger use of the current maps than calling a low-correlation sparse register map “relocation.”

### 3.5 Spatial reproducibility and effect magnitude

ViT-S mixing baseline maps have mean pairwise Spearman correlation approximately 0.912, with range 0.892–0.937 across six dependent pairs from four models.

An exploratory ring-adjusted rank analysis retains correlations approximately 0.693–0.837. This suggests structure beyond distance to the boundary, but the maps use common evaluation images.

The position-only plug-in Brier variance fractions are about 2.07%, 2.30%, 2.45%, and 2.28% for those four maps. Thus, the map can be reproducible while capturing only a modest fraction of individual event variability.

Do not describe position as determining most outliers. The inference study tests whether that reproducible component is functionally informative.

### 3.6 Trained registers

The ViT-S register mean is 78.337% across two legacy runs. Its paired difference against the corresponding two baselines is −0.381 points. The single ViT-B register run has 76.126%, −0.610 points against its matching baseline.

Registers remove more fixed-threshold patch exceedances in these recorded comparisons. The ViT-B register count of 0.0233 per image gives only about 233 events across 10,000 images. A spatial correlation from that sparse map is unstable and does not show where computational work moved.

Report the complete relevant comparisons, with the correct matched baseline subset. The asymmetry in sample counts is a limit on method ranking, not a request for new training.

### 3.7 Existing TTR results

| Base checkpoint | Baseline accuracy | TTR accuracy | Difference | Fixed-threshold events: baseline → TTR |
|---|---:|---:|---:|---:|
| ViT-S mixing s1 | 78.862 | 78.650 | −0.212 | 19.677 → 1.912 |
| ViT-B mixing s1 | 77.114 | 76.138 | −0.976 | 11.286 → 5.054 |
| ViT-S no-mixing s1 | 73.196 | 73.138 | −0.058 | 3.671 → 2.148 |
| ViT-S mixing legacy | 78.876 | 78.684 | −0.192 | 15.391 → 0.973 |

These are meaningful tradeoffs. Do not convert the project’s development “PASS/FAIL” rule into a scientific verdict on TTR.

The displayed implementation is a documented reimplementation with adaptations. Its neuron count and layer-range choices have exploratory history. Keep those settings fixed for the main follow-up evaluation and report the available operating-point sweep as such.

Stored paired image intervals compare TTR with its original baseline. They are not intervals for TTR versus a pooled SAGA mean. The corresponding descriptive TTR-minus-SAGA differences by provenance are −0.702, −1.128, −0.392, and −0.066 points, respectively.

### 3.8 Existing 100-epoch controls: useful, but do not oversell them

| Arm | Actual operation | Accuracy | MAD count | Pairwise cosine | Effective rank |
|---|---|---:|---:|---:|---:|
| A | Ungated baseline | 76.526 | 11.713 | 0.2680 | 115.527 |
| B | Frozen patch gate 0.5 | 76.238 | 4.433 | 0.2921 | 125.181 |
| C | Learned patch head-scalar | 76.178 | 6.656 | 0.2745 | 122.394 |
| D | All-token pre-projection channel scale, init 1e−6 | 75.982 | 17.458 | 0.3461 | 100.598 |
| E | Spatial gate, logit init 0 | 76.462 | 4.685 | 0.2767 | 124.674 |
| F | Spatial gate, logit init 4 | 76.702 | 10.026 | 0.3043 | 118.233 |

We already have these models. Use them to show that several forms of scaling alter geometry and that geometry/accuracy orderings differ.

Important corrections:

- Each arm is one training run.
- E exceeds C by 0.284 points, but is below A by 0.064.
- F has the highest accuracy in this table and must not disappear.
- D is not a faithful canonical LayerScale comparison.
- The old fixed-threshold counts are saturated near 196/196. Recompute them by inference with an explicit 100-epoch calibration version.
- A frozen gate is a valid control. Uniform scaling of a measured norm vector would not change its MAD outlier count; the actual network changes are more complex.

No retraining of any arm is required or planned.

### 3.9 Transfer: useful boundaries, not the headline

| Existing result | What to report |
|---|---|
| CUB / ViT-S | Mean paired fine-tuning difference −0.362 points |
| CUB / ViT-B | +1.047 points |
| Aircraft / ViT-S | +0.960 points |
| Aircraft / ViT-B | +1.090 points |
| COCO AP | 34.912 → 35.339, +0.427 |
| COCO AP-small | 18.460 → 17.616, −0.844 |
| ADE single-scale mIoU | 43.5768 → 43.3350, −0.2418 |
| ADE multi-scale mIoU | 43.9663 → 43.9685, +0.0022 |

The fine-grained family has three fine-tuning seeds per condition but only one pretrained backbone per method/architecture. Dense results are single-run. Some register transfer comparisons use legacy fallback backbones.

Use the complete family in the appendix. Existing trained dense heads are valuable for new inference controls, but do not claim that these current aggregate results demonstrate universal dense-task improvement.

### 3.10 What we will not use as strong evidence

- Mislabeled legacy “nomix” runs as true no-mixing.
- Incomplete ViT-B s2 runs.
- The old training-dynamics figure’s recipe labels/endpoints.
- The stylized teaser as if it contained actual model outputs.
- Selected favorable heads, layers, images, thresholds, or datasets without disclosure.
- The 20-image localization subset as decisive localization evidence.
- Effective rank or cosine as a direct synonym for semantic quality.
- Class-wise variation or checkpoint-epoch variation as independent training uncertainty.

---

## 4. The scientific method: three axes and a frozen-model intervention protocol

### 4.1 Three axes, measured separately

**Prevalence:** how many patches satisfy a precisely defined high-norm event?

**Spatial structure:** how are those events distributed across coordinates, after accounting for event mass and boundary geometry?

**Functional response:** how do the model’s outputs or a fixed useful task change under a declared intervention?

The paper’s method is to measure all three under controlled edits of the same saved checkpoint, and to interpret between-checkpoint comparisons with their actual training provenance.

### 4.2 Frozen means frozen

During every new evaluation:

- Load the checkpoint and preserve its parameter hash.
- Use evaluation mode.
- Disable parameter gradients and all optimizer steps.
- Do not update BatchNorm or other running statistics.
- Apply temporary forward edits through an explicit intervention configuration.
- Restore the original computation after each condition.
- Record the native prefix layout, feature stage, precision, and sample IDs.

Computing empirical frequencies, quantiles, confidence intervals, SVDs, or nearest-neighbor descriptor matches is analysis. It is not new model training.

Forbidden work includes even a “small” learned linear probe, fitting a new segmentation decoder, optimizing gates for a few steps, distilling a student, or adapting model weights at test time.

### 4.3 Exact SAGA operation

For layer l and head h:

$$
A_{\ell h}
=
\operatorname{softmax}\left(
Q_{\ell h}K_{\ell h}^{\mathsf T}/\sqrt{d_h}
\right),
\qquad
Z_{\ell h}=A_{\ell h}V_{\ell h}.
$$

For patch p:

$$
g_{\ell hp}=\sigma(\phi_{\ell hp}),
\qquad
\widehat Z_{\ell hp}=g_{\ell hp}Z_{\ell hp}.
$$

For CLS, the output is unchanged. The attention residual is

$$
Y_\ell=X_{\ell-1}+
\operatorname{Concat}_h(\widehat Z_{\ell h})W^O_\ell+b^O_\ell,
$$

followed by the original tokenwise MLP residual.

The gate modulates **receivers**, after attention aggregation and before output projection. It does not mask a key column in the already-computed attention matrix. It can influence attention in later layers.

The trained implementation has one gate per layer/head/patch position:

- ViT-S: 12 × 6 × 196 = 14,112 gate parameters.
- ViT-B: 12 × 12 × 196 = 28,224.

The last block’s patch-only parameters have no path to a CLS-only task loss. The task-connected counts under that objective are 12,936 and 25,872. This is a property to disclose and control, not a reason to relabel the existing checkpoints.

Initialization is logit zero, hence gate 0.5, not identity. Sigmoid values above 0.5 mean less attenuation than initialization, not multiplication above one. For a new resolution, the code interpolates sigmoid gate values.

### 4.4 Why a gate can change behavior without “removing a sink”

At a fixed layer/head/receiver, gAV can be written as an attention-like weighted sum over the existing values plus a zero-valued slot:

$$
gAV=[gA,\;1-g]
\begin{bmatrix}V\\0\end{bmatrix}.
$$

This algebra illustrates reduced update strength. It is not a claim that SAGA implements a learned register: a trained register can have nonzero, input-dependent contents and can interact through subsequent layers.

For 0 < g < 1, an equivalent added zero-slot logit can be expressed as

$$
s_{\bot}=\operatorname{LSE}(s)+\log\frac{1-g}{g},
$$

where s are the original key logits. The added logit depends on the original log-sum-exp; it is not a fixed learned register key.

Use this as explanatory algebra, not a new theorem or a priority claim. “No-op” interpretations already motivate earlier gated attention. [Quantizable Transformers](https://arxiv.org/abs/2306.12929).

### 4.5 Why the gate’s learned pattern is not automatically an outlier detector

The task-loss gradient has the local form

$$
\frac{\partial \mathcal L}{\partial\phi_{\ell hp}}
=
g_{\ell hp}(1-g_{\ell hp})
\left\langle
\frac{\partial\mathcal L}{\partial Y_{\ell p}},
Z_{\ell hp}W^O_{\ell h}
\right\rangle.
$$

This integrates task-gradient alignment across training images at a fixed coordinate. It does not force the gate to track high norms, foreground, or semantic importance.

A gate/prevalence correlation is therefore an observation to test with interventions, not the mechanism itself. No gradient computation or optimization is required for the proposed inference study.

### 4.6 The proposed protocol in one sentence

> Compare the original checkpoint with scale-controlled, spatially rearranged, and readout-controlled versions of its forward computation, then measure prevalence, spatial structure, and functional response on the same images.

This is the central method section of the paper. SAGA is a useful part of the study because its coordinate-dependent parameters can be manipulated cleanly while all trained backbone weights remain fixed.

## 5. Mathematical definitions and what the theory actually establishes

The theory should clarify the interventions and prevent invalid interpretations. It should not claim a general theorem that SAGA improves accuracy, removes harmful information, or finds an optimal gate.

### 5.1 Norm events and their spatial distribution

Let X at a declared feature stage contain N patch vectors, excluding CLS and register tokens. Define

$$
n_{ip}=\|X_{ip}\|_2,
\qquad
b_{ip}(\tau)=\mathbf 1[n_{ip}>\tau],
\qquad
c_i=\sum_{p=1}^{N}b_{ip}.
$$

For M images,

$$
f_p=\frac{1}{M}\sum_i b_{ip},
\qquad
\bar f=\frac{1}{N}\sum_p f_p,
\qquad
q_p=\frac{f_p}{\sum_j f_j}.
$$

The average count is N times bar f. The map f measures absolute occurrence; q measures the location distribution conditional on an event. If there are no events, q is undefined and must be marked as such.

For a 14 × 14 grid, define a patch's boundary ring by

$$
r(p)=\min(u_p,v_p,13-u_p,13-v_p).
$$

The event share in ring r is

$$
R_r=\sum_{p:r(p)=r}q_p.
$$

Compare this with the ring's area fraction. Ring one has 44/196 of the positions. A fall in total count and a rise in R_1 are mathematically compatible. Neither establishes a causal effect on predictions.

Use both absolute maps f and normalized maps q. Do not independently rescale every heatmap and then compare color intensity as event prevalence.

### 5.2 Fixed thresholds and relative thresholds answer different questions

The canonical current fixed thresholds are 20.8515625 for ViT-S mixing, 22.859375 for ViT-S true no-mixing, and 127.3125 for ViT-B mixing. They come from the current canonical calibration file and must not be merged with the older threshold version.

Fixed-threshold counts are sensitive to absolute feature scale. They remain useful when calibration and readout are stated clearly.

The implemented per-image MAD event is

$$
b^{\mathrm{MAD}}_{ip}
=
\mathbf 1\left[
n_{ip}>
\operatorname{median}_{j}(n_{ij})
+
5\,\operatorname{median}_{j}
\left|n_{ij}-\operatorname{median}_{k}(n_{ik})\right|
\right].
$$

This uses the unscaled MAD and the implementation's lower-median convention.

**Proposition 1 — Positive affine invariance of this MAD event.** If every norm in one image is replaced by a n + b, with a > 0, its median becomes a median(n) + b and its MAD becomes a MAD(n). Therefore every strict threshold comparison is unchanged.

**Implication:** a change in MAD count cannot be explained solely by a positive affine transformation of the measured norm vector. It can still arise from geometry changes induced by attenuation earlier in a residual network. This proposition does not make MAD a measure of semantic quality.

Recompute any 100-epoch fixed-threshold diagnostics with a separately versioned 100-epoch calibration. Preserve historical numbers as historical; do not overwrite their provenance.

### 5.3 Reproducible maps need not explain much individual variation

Consider an event B at a uniformly sampled position P and a random image. The ideal coordinate-only event probability is f(P). Under Brier loss,

$$
\mathcal R(\bar f)-\mathcal R(f)
=
\operatorname{Var}_{P}(f(P)).
$$

This follows by conditioning on P and expanding the squared error. The total unconditional Bernoulli variance is bar f times (1 − bar f), so an illustrative position-associated fraction is

$$
\frac{\operatorname{Var}_{P}(f(P))}
{\bar f(1-\bar f)}.
$$

The existing 2.07–2.45% plug-in values are descriptive, in-sample quantities. In the new analysis, estimate the frequency table on calibration images and evaluate its Brier loss on different images. This is direct empirical-frequency analysis, not a trained neural or linear probe.

Compare a global frequency, a frequency by boundary ring, and a frequency by exact position. Their held-out differences show whether the coordinate map adds prediction beyond boundary geometry. They do not establish a content-independent cause.

### 5.4 Separate mean attenuation from spatial arrangement

For every layer and head,

$$
\mu_{\ell h}=\frac{1}{N}\sum_p g_{\ell hp},
\qquad
\delta_{\ell hp}=g_{\ell hp}-\mu_{\ell h},
\qquad
\sum_p\delta_{\ell hp}=0.
$$

The mean-collapsed intervention uses g = mu. A rearrangement uses

$$
g^{\pi}_{\ell hp}=\mu_{\ell h}+\delta_{\ell h,\pi(p)}.
$$

A permutation preserves each head's gate histogram and mean. For the primary permutation, apply the same position permutation to every head in the edited layer; this also preserves the joint collection of head-gate vectors. Independent permutations across heads are a separate, more disruptive control.

The mean-zero condition does not imply that spatial contributions cancel: the output vectors being multiplied vary across positions and heads. This is precisely why the frozen-model intervention is needed.

A gate permutation preserves gate values, but generally does **not** preserve the norm of the resulting attention update. An additional activation-energy control is required before attributing a loss change specifically to spatial alignment.

### 5.5 Match injected update energy without optimizing anything

For a fixed layer input, let the patch attention update excluding the projection bias be

$$
U_{ip}
=
\sum_h \widehat Z_{ihp}W^O_h.
$$

For a spatial mask M, a weak attenuation injects

$$
\Delta Y_{ip}
=
-\alpha_{iM}\,\mathbf 1[p\in M]\,U_{ip}.
$$

For a prespecified family of candidate masks, define

$$
\kappa_i=\epsilon\min_M\|M\odot U_i\|_F,
\qquad
\alpha_{iM}=\frac{\kappa_i}{\|M\odot U_i\|_F}.
$$

Then every candidate receives the same Frobenius-norm perturbation kappa for that image, and every alpha is at most epsilon. This is a closed-form normalization of a forward-pass perturbation; no parameters are fitted.

If the minimum norm is zero, all conditions for that image receive zero perturbation; report its frequency. Do not silently discard images according to the eventual loss effect.

This control matches **injected update energy at the intervention site**. It does not match the later activation trajectory, information content, or semantic importance of the selected patches.

For the gate-arrangement control, compute the spatial residual component

$$
D^{\pi}_{ip}
=
\sum_h \delta_{h,\pi(p)}Z_{ihp}W^O_h.
$$

Compare the ordinary gate permutation with the activation intervention

$$
Y^{\pi,\mathrm{energy}}_i
=
Y^\mu_i+
\frac{\|D^{\mathrm{id}}_i\|_F}{\|D^\pi_i\|_F}D^\pi_i.
$$

This preserves the total energy of the spatial component relative to the original. It is an analysis-only activation edit, not another valid sigmoid-gate architecture. Handle a zero denominator explicitly and log it. Use single-layer edits for this control so the incoming activations are identical.

### 5.6 The terminal readout control

**Proposition 2 — Terminal patch-gate invariance for a CLS-only classifier.** Suppose the gate acts only on patch rows after attention aggregation in the last transformer block. Suppose all remaining operations before classification are tokenwise, and the classifier reads only CLS. Then changing those terminal patch gates cannot change the classification logits.

**Proof.** The attention matrix and aggregated CLS output are computed before the patch-output gate. Changing the gate leaves the CLS row unchanged. The output projection, residual addition, MLP, and final normalization do not mix rows. Therefore the CLS vector and its classifier logits remain unchanged. The derivative of the classification loss with respect to those patch-gate parameters is zero.

The proposition assumes evaluation mode and the declared readout. It does not apply to mean-patch pooling, later token mixing, or a dense head that consumes patch features. Regularization updates, such as weight decay, are separate from task-loss gradients.

**Empirical role:** vary the final patch gate while checking classification invariance and recording the patch diagnostics. This can demonstrate a readout-dependent diagnostic change. It is a correctness control and an interpretive example, not the main novelty of the paper.

If the last gate of a dense-task checkpoint was fine-tuned through patch features, it may have received task gradients. Do not transfer the classification argument to that checkpoint.

### 5.7 Why “smaller gate means smaller feature norm” is not a theorem

Even a simplified residual vector x + g u has

$$
\frac{d}{dg}\|x+gu\|_2^2
=
2\langle x,u\rangle+2g\|u\|_2^2.
$$

The sign depends on alignment. Multihead projection, residuals, normalization, and subsequent blocks make the full effect more complex.

Therefore the mathematical explanation is:

1. The gate changes a receiver's attention contribution.
2. Its effect depends on the current update direction and downstream readout.
3. The empirical controls measure whether spatial arrangement contributes beyond simpler attenuation.

It is not a proof that SAGA must reduce every norm, improve every rank statistic, or improve accuracy.

### 5.8 Functional quantities and feature quality

Use per-image negative log likelihood as the primary classification-response endpoint:

$$
\Delta_i^{(v)}
=
-\log p_v(y_i\mid x_i)+\log p_0(y_i\mid x_i).
$$

Positive values mean the intervention worsens that image's true-label loss. Report top-1 change and prediction-distribution divergence separately; a changed prediction distribution is not necessarily worse accuracy.

For feature geometry, keep the exact existing definitions:

- Pairwise cosine is the mean off-diagonal cosine among normalized patch vectors.
- Effective rank is the exponential entropy of **singular values normalized by their sum**, not squared singular values.
- The existing rank calculation uses the uncentered, unnormalized patch matrix.
- Principal historical diagnostics are at the stated pre-final-LayerNorm block output.
- If reporting cosine with outliers excluded, use a common mask as a control: different per-model masks otherwise select different token populations.

For a cosine-normalized patch matrix with N rows, the mean off-diagonal cosine can be calculated from the squared norm of the sum of its rows. There is no need to materialize every pair.

Measure geometric correspondence or an existing task head separately. Neither cosine nor effective rank is a substitute for a useful readout.

### 5.9 High norm, incoming attention, and CLS visualization are different

For patch-key position p, define incoming patch attention at a declared layer as

$$
a_{ip}
=
\frac{1}{H N_{\mathrm{patch}}}
\sum_h\sum_{q\in\mathrm{patches}} A_{ihqp}.
$$

Retain attention to CLS and registers in the normalization; do not renormalize it away without labeling the alternative statistic.

Distinguish:

- Patch queries attending to the CLS key.
- The CLS query attending to patch keys.
- Patch queries attending to other patch keys.
- Norms measured before or after the block.

The stored CLS-attention-share quantity and a typical CLS-to-patch teaser map use opposite directions. They cannot be substituted for one another.

A post-block norm cannot explain an attention matrix that was already computed earlier in that block. For mechanistic timing, compare a norm at the block input with attention in that block, or an output norm with the next block's attention. Use association language unless the corresponding intervention was performed.

---

## 6. The complete new work: inference and analysis only

### 6.1 Budget and priority table

“Required” means required for the proposed claim, never permission to train.

| ID | Work | Inputs | New training? | Priority |
|---|---|---|---|---|
| I0 | Checkpoint, recipe, split, and metric audit | Existing configurations, logs, checkpoints | No | Required first |
| I1 | Scale/occupancy/boundary-controlled spatial analysis | Existing per-image norms or fresh frozen forward passes | No | Required for C1 |
| I2 | Terminal patch-gate/readout control | Existing SAGA classifiers | No | Required correctness control |
| I3 | Gate mean, arrangement, and energy controls | Existing SAGA classifiers | No | Required for a gate-specific C2 claim |
| I4 | Matched coordinate perturbations across methods | Existing baseline/SAGA/register checkpoints | No | Required for the main functional spatial claim |
| I5 | A useful frozen readout: correspondence, optionally existing dense heads | Existing backbone/head checkpoints | No | Required to discuss local-feature usefulness |
| I6 | Bounded external confirmation on public pretrained weights | Official frozen checkpoints | No | Highest-value optional extension |
| I7 | Attention-direction and TTR tradeoff analysis | Existing TTR settings, saved baselines, forward outputs | No | Compact supporting analysis |

The minimum strong submission is I0–I5 with one clear, nontrivial result, plus the existing complete comparison tables. I6 strengthens generality if the inference resources permit. Do not add a new architecture-training campaign under any name.

### 6.2 Shared data and execution protocol

**Discovery material:** the existing 10,000-image diagnostic subset, current maps, and previous aggregate validation scores. They have already informed the hypotheses.

**New calibration subset:** 2,000 validation images outside that diagnostic subset, if IDs can be recovered. Use it only for empirical frequency estimates, threshold calibration, and label-independent setup.

**Locked evaluation subset:** a disjoint 10,000-image subset outside discovery/calibration for the new interventions. Fix its IDs and the analysis configuration before inspecting its intervention outcomes.

The original full-validation accuracies have already been seen. Accordingly, call this a **locked evaluation protocol for the new analyses**, not a completely untouched test set or a retrospectively preregistered study.

If historical IDs cannot be reconstructed, regenerate clearly versioned calibration/evaluation splits and state that independence from historical exploration is uncertain. Do not invent split provenance.

Use a fixed 1,000-image subsubset for expensive SVD and full attention extraction, and a fixed 2,000-image subsubset for correspondence. These subsets must be identical across eligible methods and chosen without regard to performance.

For computational efficiency, cache the quantities actually needed:

- Per-image logits or sufficient paired-loss/correctness records.
- Per-layer patch norms and stage labels.
- Gate values and layer/head summaries.
- Only the attention reductions required for the defined endpoint.
- Feature descriptors for the correspondence subset.

Do not save every attention tensor for every layer and intervention unless a specified analysis needs it.

### 6.3 I0 — Reconstruct the eligible cohort and lock the protocol

**Action.** Produce one manifest row per saved checkpoint with actual architecture, training recipe, source run, seed evidence, checkpoint epoch, prefix layout, hash, gate variant, and evaluation provenance.

**Reuse.** All 19 completed 300-epoch conditions remain in the descriptive cohort. Keep 100-epoch arms and downstream checkpoints in separate strata.

**Corrections.**

- Resolve actual augmentation settings from effective configurations, not a filename containing “nomix.”
- Resolve seeds from training metadata; a diagnostic JSON's default seed field may only describe evaluation.
- Separate the legacy matched pairs from the two fresh ViT-S pairs.
- Mark incomplete runs and epoch-mismatched dense checkpoints.
- Record dirty working-tree information when a git SHA is not sufficient.
- Separate historical canonical thresholds from newly calibrated controls.

**Output.** A machine-readable manifest and a human-readable eligibility table.

**Why this matters.** A mechanism study with mislabeled recipes or mixed feature stages will not survive review, regardless of the diagrams.

### 6.4 I1 — Does lower occurrence mean weaker spatial structure?

**Primary question.** Does the observed spatial change contain information beyond reduced event mass, a simple scale change, and boundary distance?

**I1a: absolute versus conditional maps.** Recompute counts, f, q, ring shares, spatial entropy, and split-half map reliability for the complete eligible cohort. Keep the original threshold comparisons for continuity.

**I1b: a scale-only statistical null.** On calibration images, choose a single positive rescaling of baseline norms that matches the SAGA mean fixed-threshold event rate as closely as possible. Apply that scalar unchanged to the baseline's evaluation norms. Compare its evaluation map, ring share, and reliability with SAGA.

This is a statistical counterfactual on recorded norm values. It is not a newly trained baseline and is not a forward computation whose classification accuracy can be claimed. Label it “rescaled baseline norms,” never “baseline model with matched accuracy.”

If a global rescaling reproduces SAGA's map change, the result limits C1: the absolute-count/spatial-shape observation alone does not require a spatial-gating mechanism.

**I1c: equal occupancy.** For every image, select the top eight norm positions. Construct rank-occupancy maps; use k = 4 and 16 as declared sensitivity checks. Each image contributes the same number of positions.

These are **top-k norm-rank maps**, not outlier-incidence maps. They are particularly useful when registers leave too few threshold events for a reliable map.

**I1d: boundary controls.** Report frequency by ring, within-ring residual maps, and held-out Brier improvement from global → ring → full-position empirical frequencies. Examine class and simple local-image-variance strata as descriptive controls, without fitting a new classifier.

**I1e: sparse-map controls.** Report total events and split-half reliability beside every correlation. Compare event-mass-matched thinning where feasible. A nearly empty register map cannot be used to claim randomization or relocation.

**Statistical null.** A per-image random mask preserving event count can describe a uniform-location reference, but it assumes spatial exchangeability. Ring-preserving masks relax only the boundary part of that assumption. Do not call toroidal shifts an exact hypothesis test for nonstationary images.

**Output.** Figure 1 population panels, Figure 3 spatial analysis, and a complete appendix table.

**Interpretation.** This analysis can establish a descriptive separation of prevalence and spatial organization. Only I3/I4 can establish intervention effects on the model's computation.

### 6.5 I2 — Can a patch statistic change while classification stays fixed?

**Checkpoints.** All existing SAGA classifiers for a low-cost correctness pass; perform detailed patch diagnostics on the representative paired ViT-S cohort.

**Conditions.** Original final gate, then replace only that final patch gate by constants 0.25, 0.5, 0.75, and 1.0. The 1.0 condition is a bypass of that gate. Do not change earlier layers.

**Record.**

- Maximum and distribution of absolute logit differences in fp32.
- Top-1 agreement with the original.
- Patch norms, fixed and MAD counts, cosine, and rank at the final block output.
- Corresponding penultimate-stage quantities as an unchanged-stage check.
- Gate values and the native classifier readout.

**Expected architectural property.** Classification logits are invariant under the assumptions of Proposition 2. Check a tight documented numerical tolerance; investigate failures as a readout/hook/implementation problem before reporting them.

**Unknown empirical quantity.** The size and direction of the diagnostic change. Do not assume a large “improvement.”

**Readout extension.** Run the same kind of edit through an already-trained dense head only if available, with that checkpoint's own gate values and architecture. Its output may change because it reads patches.

**Use in the paper.** A concise panel showing diagnostic movement versus zero classifier response. It prevents a false mechanism claim. It cannot alone justify acceptance.

### 6.6 I3 — Does the learned spatial gate arrangement matter?

**Primary cohort.** The four existing ViT-S mixing SAGA checkpoints. Report the two fresh runs separately from the two legacy runs. Extend the decisive contrast to the existing ViT-S no-mixing and ViT-B SAGA pairs if it is informative; do not begin with a huge grid over every architecture.

**Layer choice.** Start with paper blocks 8 and 9, selected because the existing exploratory correlations highlighted them. Explicitly disclose that selection. Use one joint early-through-middle edit as a sensitivity check only if needed; do not select the winning layer after inspecting evaluation losses.

**Core variants at one layer at a time.**

1. Original gate.
2. Mean-collapsed gate mu for each head.
3. Gate mu + 0.5 delta.
4. Ten fixed position permutations, shared across heads.
5. Ten fixed within-ring permutations, shared across heads.

Use the same saved permutation indices across checkpoints on the same grid. Permutation draws are repeated interventions on a checkpoint, not new model seeds.

**Two additional controls on a fixed smaller subset.**

- A small prespecified set of grid rotations/reflections, which preserve local adjacency more than arbitrary shuffling.
- The spatial-component energy-matched activation edit from Section 5.5.

The energy control is necessary when the headline interpretation depends on arrangement rather than the amount of the modified update. Do not infer that histogram preservation already provides it.

**Endpoints.** Paired NLL difference is primary. Top-1 difference, logit-distribution change, norms, and spatial maps are secondary.

**Critical interpretation.**

- Shuffling that harms performance shows reliance on the learned arrangement in this checkpoint.
- A stronger harm than ring-preserving shuffling suggests information beyond ring membership.
- Mean collapse with little functional change limits the claim that fine spatial variation is functionally important.
- An effect that disappears under energy matching may be due largely to update magnitude.
- None of these outcomes establishes the accuracy of a scalar model trained with a matched recipe.

**Decision rule.** Claim a meaningful arrangement effect only if the effect direction is consistent in the fresh pair, its conditional image-level interval excludes zero, and the observed scale is practically interpretable. Show the legacy and cross-recipe results rather than turning their larger number of images into extra training replication.

If the result is merely “any shuffled parameter hurts,” make it supporting evidence and let I4/I5 carry the paper, if they yield a more informative finding.

### 6.7 I4 — Do high-prevalence coordinates have a distinct functional response?

This is the most important new inference experiment. It asks whether the spatial maps are informative about a controlled effect on the frozen network, beyond count, boundary location, and perturbation energy.

**Common intervention site.** At the selected block, edit the effective patch attention output after any native SAGA gate and before the attention residual addition. Apply the corresponding operation to baseline and register models. Exclude CLS and register rows; preserve their normal computation.

**Temporal alignment.** Define the discovery map from norms at the input to that block, or from the preceding block output. Do not use a later final-layer norm as though it caused an earlier attention output.

**Masks.**

- Select the 16 highest-prevalence patch coordinates from the discovery map.
- Generate ten fixed random control masks with the same number of selected coordinates in each boundary ring.
- Use the common baseline-derived coordinate mask as the primary cross-method comparison on the same grid.
- Use each method's own map only as a clearly labeled secondary question.

The random masks may overlap the high-prevalence mask. This reflects the declared constrained null; report overlap rather than secretly rejecting draws to amplify contrast.

**Perturbation.** Use weak output attenuation with epsilon = 0.10 and the per-image energy matching in Section 5.5. A single epsilon = 0.25 sensitivity condition tests stability, not an operating point to optimize.

**Primary contrast.**

$$
\theta_m
=
\mathbb E_i\left[
\Delta_i^{\mathrm{high},m}
-
\frac{1}{R}\sum_{r=1}^{R}\Delta_i^{\mathrm{ringcontrol}(r),m}
\right].
$$

A positive theta means attenuation at high-prevalence coordinates worsens loss more than matched controls. A negative value means less worsening or more improvement. Either sign needs interpretation in the context of the actual perturbation.

**Cross-method contrast.** Compare theta between matched baseline/SAGA checkpoints on the same images and coordinates. Add the available registers with their true provenance and sample count. This is a comparison of intervention susceptibility, not an estimate of a training treatment effect.

**Controls.**

- Equal mask size and ring composition.
- Equal injected update energy within each image/checkpoint.
- Identical image set and intervention stage.
- Native prefix tokens preserved.
- Actual perturbation norms logged.
- No parameter optimization and no mask choice using evaluation losses.

Energy matching is within a checkpoint. Report the native update scale and normalized perturbation size when comparing checkpoints; equal absolute energy across differently scaled representations would introduce another interpretive problem.

**What it establishes if successful.** High-prevalence coordinates carry a distinct response to a specific receiver-output perturbation in the studied computation, beyond the measured controls.

**What it does not establish.** High norms are the mediator, the coordinates are causally harmful, or image content has been fully removed as an explanation.

**Optional extension only if necessary.** A full per-position response map on a small fixed image subset for one baseline/SAGA pair. Do not make 196 single-position perturbations across every model a mandatory workload. Correlate response and prevalence only after ring adjustment, and do not treat 196 coordinates as independent model replications.

### 6.8 I5 — Does the distinction matter for a useful readout?

The main paper needs a useful endpoint beyond “the logits changed.” Use frozen geometric correspondence as the default because it requires no new head. Existing dense heads provide a valuable complementary route.

#### I5a: training-free geometric correspondence

**Inputs.** Existing backbone checkpoints; no learned projection, PCA model, linear probe, or decoder.

**Views.** Produce deterministic patch-aligned translations by two patches left/right/up/down. Use a fixed padding convention and exclude padded and ambiguous border regions from both query and candidate sets. The correct mapping is known exactly. Add one fixed photometric transformation only as a secondary robustness check.

**Descriptors.** Use native patch vectors, L2-normalized for cosine matching. Evaluate the declared final pre-LayerNorm stage, native final-LayerNorm stage, and penultimate stage. Name the primary stage in advance; do not select each method's best stage.

**Matching.** For each valid patch in the first view, find the nearest descriptor among valid patches in the second view. Report exact mapped-patch accuracy and accuracy within one patch of the mapped location. Aggregate within image before calculating intervals.

**Baselines.**

- Native baseline, SAGA, and trained-register descriptors.
- A same-coordinate correspondence rule to reveal trivial positional matching.
- A fixed raw-patch descriptor as a simple image-level reference.
- SAGA with terminal-gate bypass, labeled as an inference edit.

The true mapping changes coordinates, so reporting only same-coordinate matches would defeat the test.

**Scope.** This measures geometric consistency under a known transformation. It is not a semantic correspondence benchmark, object localization proof, or dense-feature state of the art. Training-free use of ViT descriptors already has a substantial literature. [Deep ViT Features as Dense Visual Descriptors](https://arxiv.org/abs/2112.05814).

**Contribution.** Determine whether the diagnostic ordering predicts this fixed utility endpoint and whether terminal-only changes alter its result differently from the classifier.

#### I5b: already-trained dense heads, if their checkpoints are accessible

Select one existing ADE model family as the first dense evaluation. Keep the entire pretrained backbone and decoder frozen.

Evaluate original computation, a prespecified gate-mean edit, and a terminal edit where the architecture supports it. Use the original dataset protocol and a common checkpoint epoch for any directly paired comparison. If the required matched epochs do not exist, report the mismatch and do not claim a controlled model ranking.

Report mIoU and paired changes in per-image loss or segmentation output. The final gate can affect this readout; that is expected.

Do not train a replacement head to repair a representation edit. Performance after such repair would answer a different question and violate this plan's budget.

**Stopping rule.** One well-controlled useful endpoint is more valuable than a hurried collection of weak transfer analyses. Do not require both every COCO edit and every ADE edit to complete the paper.

### 6.9 I6 — Optional external confirmation without training

This is the best optional addition if inference capacity remains.

**First choice:** one official pretrained DeiT III classifier with a documented native head. Repeat the spatial-map measurement and the decisive I4 receiver perturbation. There is no SAGA gate in this model; do not imply that SAGA has been evaluated under its stronger training recipe. [DeiT III](https://arxiv.org/abs/2204.07118).

**Second choice:** an official DINOv2 with/without-register pair, using native frozen patch features for I1/I5. Respect its actual patch grid, register indices, preprocessing, and head availability. If no native classifier is used, do not attach a newly trained classifier. [DINOv2](https://arxiv.org/abs/2304.07193).

**Modern context:** DINOv3 may be included with official frozen weights if it directly answers the remaining question. It is not necessary to evaluate every modern family. [DINOv3](https://arxiv.org/abs/2508.10104).

**What external confirmation buys.** Evidence that a measurement distinction is not confined to the project's weaker supervised recipe.

**What it does not buy.** A matched stronger-recipe SAGA training comparison or broad register superiority.

If the optional study cannot be completed, retain the supervised-cohort scope in the title/abstract discussion and limit general claims.

### 6.10 I7 — Existing TTR and attention analysis

Keep the recorded TTR construction fixed for the principal follow-up. The source reimplementation records an upstream commit and a maximum of 24 neurons, with the restricted layer range 3:12 in zero-based indexing. The exploration that led to that range belongs in the appendix.

For each existing baseline/TTR pair, collect the same population and functional endpoints where the intervention interface supports them. Do not choose a new test-set-optimal neuron count to compete with SAGA.

Use the available TTR sweep as an operating-point plot with all evaluated points, identifying the setting chosen before follow-up evaluation.

For attention, plot properly directed incoming attention against the temporally aligned norm/prevalence measure. Use all declared layers or a fixed layer selection. Treat this as association unless I4 directly perturbs that computation.

Do not require a new attention algorithm, a new trained register model, or a large prompt/attack benchmark.

---
## 7. Statistics, result selection, and the standard of evidence

### 7.1 Distinguish three sources of variation

**Across images:** paired bootstrap intervals quantify uncertainty for a fixed checkpoint and fixed intervention on the sampled image population.

**Across intervention draws:** permutation or mask variation describes sensitivity to the selected intervention. Ten permutations are not ten trained models.

**Across trained checkpoints:** the existing two or four repeats describe training variability, with the stated legacy limitations. Thousands of images cannot replace missing training replication.

Show per-checkpoint points beside means. For inference, calculate paired differences on the same image IDs and resample images as units. Keep all tokens, transforms, and intervention conditions for an image together.

When averaging across fixed control masks, bootstrap the per-image mean contrast. Separately display its distribution across masks. Label whether an interval is conditional on the observed models and masks.

For the small number of trained repeats, avoid narrow population claims from a bootstrap of only two checkpoints. Give raw paired effects and descriptive ranges; any t interval should show its assumptions and wide uncertainty.

### 7.2 Primary endpoints and multiple comparisons

Lock a small primary family:

1. I1: held-out ring-versus-position Brier difference and the scale-null comparison.
2. I3: NLL change under within-ring gate rearrangement at blocks 8 and 9.
3. I4: high-prevalence-versus-ring-control NLL contrast at those blocks.
4. I5: exact known-transform correspondence at the prespecified feature stage.

The other plots explain these endpoints. Do not search all layers, heads, thresholds, and transformations for the smallest p-value.

If reporting confirmatory p-values across the primary layer contrasts, use an explicit family correction such as Holm. Prefer effect sizes, intervals, and raw checkpoint consistency over a table of significance stars.

Discovery analyses remain labeled exploratory, including the original layer selection and gate/prevalence correlations.

### 7.3 Null results and practical importance

A nonsignificant difference is not evidence of equivalence. If a “no meaningful change” statement matters, specify an equivalence margin before evaluating the locked subset and justify it in units of the endpoint.

For classification, a possible prespecified reporting margin is ±0.10 percentage points, clearly described as a practical convention rather than a scientific constant. A wide interval crossing that margin is inconclusive.

For NLL, report the absolute difference, its fraction of native mean NLL, and the associated top-1/prediction-change rate. A statistically detectable microscopic loss change should not become the central contribution.

Terminal classifier invariance is an architectural statement checked numerically; it does not depend on an equivalence test.

### 7.4 What to publish and what to leave out

Use **relevance and validity**, not favorable sign, as the selection rule.

| Evidence family | Main paper | Appendix / archive |
|---|---|---|
| Corrected baseline/SAGA/register classification cohort | Compact complete summary with matched counts | Every eligible run and pairing |
| Spatial prevalence and controls | Representative population panels plus all group summaries | Every map, threshold, ring, and reliability result |
| New primary functional interventions | All prespecified primary contrasts | All draws and declared sensitivities |
| Six 100-epoch arms | Small context panel if needed | Complete six-arm table, corrected calibration |
| TTR | One matched tradeoff summary | All recorded operating points and implementation details |
| Fine-grained transfer | One honest scope sentence | Complete dataset/architecture/seed family |
| COCO and ADE aggregates | Scope sentence; main dense inference only if it serves C3 | Complete metrics and checkpoint/provenance caveats |
| 20-image localization exploration | Omit from main argument | Optional exploratory appendix, or archive with a reason |
| Mislabeled, incomplete, saturated, or invalid summaries | Do not use as evidence | Explain exclusion/correction in the provenance ledger |
| Old exploratory jobs unrelated to the selected question | Not required | Retain internally; no need to clutter the paper |

A negative result that changes the meaning of a central claim must be discussed in the main text. Moving full numbers to an appendix does not permit a contradictory headline.

This policy allows a focused paper without pretending that every experiment succeeded.

### 7.5 What each experiment can establish

| Proposed conclusion | Minimum supporting evidence | Observation that defeats or narrows it |
|---|---|---|
| Occurrence and conditional spatial shape differ | I1 count/map decomposition on held-out images | Identity holds descriptively, but a scale-only null may fully explain the change |
| Exact coordinates add information beyond rings | Held-out position frequencies outperform ring frequencies, with reported magnitude | No held-out gain or unstable maps |
| Learned gate arrangement is functionally used | I3 paired effects that survive appropriate controls | Mean collapse/permutation produces negligible or energy-explained effects |
| High-prevalence coordinates have a distinct perturbation response | I4 ring- and energy-controlled contrast | Contrast vanishes under either control |
| Norm diagnostics alone do not predict a useful readout | I2 plus a concrete I5 disagreement or other complete paired counterexample | All apparent disagreements arise from incompatible feature stages or invalid comparisons |
| SAGA offers a useful specific tradeoff | Matched existing results plus the relevant I5 endpoint | Registers or TTR dominate that endpoint, or uncertainty is too wide |
| Finding extends beyond this recipe | I6 replication of the same declared phenomenon | External model does not show it; retain narrower scope |

The paper should present the strongest **surviving** conclusion. It should not reinterpret every failed test as confirmation of the original story.

---

## 8. Figures: exact designs and the fate of the supplied images

### 8.1 Figure 1 — A population-level teaser that earns its space

**Replace demo_teaser_image.PNG.** Its visual idea can inspire layout, but stylized heatmaps or illustrative numbers must not appear as empirical results.

**Proposed title:** “Fewer high-norm patches does not identify what changed.”

Use three aligned panels across the page:

**Panel A: absolute prevalence.** A 14 × 14 population map for a prespecified ViT-S baseline/SAGA pair, using a shared absolute color scale. Put mean fixed-threshold count underneath each map. Identify the checkpoint and evaluation split.

**Panel B: where the remaining events occur.** Ring shares for baseline, SAGA, and the rescaled-baseline-norm null. Include ring area as the reference. This makes the distinction between event amount and conditional distribution immediately visible.

**Panel C: functional response.** The decisive I4 high-prevalence-versus-control contrast, with checkpoint points and image-conditional intervals. If I4 is uninformative, substitute the substantive I5 result; do not insert a fabricated favorable outcome.

The teaser should have one scientific sentence, not multiple architecture claims. For example, after results support it:

> Absolute event suppression and spatial/functional change are different quantities, even within a fixed family of ViTs.

If the only completed content is A/B, label the figure descriptive and do not let it imply a functional result.

**Why this is stronger than the original teaser:** it uses population evidence, an explicit alternative explanation, and a measured consequence. A single attractive image cannot do those jobs.

### 8.2 Figure 2 — Where each intervention acts

Rebuild the supplied fig2_saga_diagram.PNG as a precise vector schematic.

Show one attention block with:

1. Queries/keys/values and the attention aggregation.
2. Patch receiver outputs.
3. The head/position sigmoid gate before output projection.
4. An unchanged CLS path.
5. The attention residual and tokenwise MLP.

Alongside, show registers as added token rows/columns participating in attention, and TTR as its declared neuron-edit operation. Do not depict SAGA as directly deleting attention-key columns.

A small inset should show the last block and the CLS-only readout, marking why the final patch gate has no classification-loss path.

Label the gates as input-independent and learned during the original training. State that the new experimental edits are forward-only.

Keep this figure compact, about a third page or less. Move tensor dimensions and full register/TTR implementation detail to the appendix.

### 8.3 Figure 3 — Reproducible structure with scale and boundary controls

Four panels:

- Baseline, SAGA, register absolute/conditional maps with event totals and split-half reliability.
- Ring profiles, including the scale-only null.
- Held-out Brier improvements: global → ring → exact coordinate.
- A compact summary across ViT-S mixing, true no-mixing, and ViT-B mixing.

Use aggregate population maps or a prespecified checkpoint, not the seed with the most striking ring. Put all checkpoint maps in the appendix.

For sparse register maps, display “insufficient event mass for reliable correlation” where appropriate. Show top-k rank maps as a separately labeled row, not a silent replacement for incidence.

Annotate the small position-associated effect size. High map correlation should not visually masquerade as nearly deterministic coordinate behavior.

### 8.4 Figure 4 — The central controlled intervention result

Allocate the largest evidence figure to this.

**Panel A:** I3 original, mean collapse, ordinary permutation, and within-ring permutation. Plot NLL changes, not only changed gate pictures.

**Panel B:** gate-arrangement effects with and without spatial-component energy matching.

**Panel C:** I4 high-prevalence versus ring-matched masks under identical injected-energy rules.

**Panel D:** the same primary contrast in the fresh ViT-S pair, legacy pair, and available cross-recipe/scale confirmation.

Use a clear zero line. Show each checkpoint; use different markers for legacy and fresh provenance. Colors encode method consistently throughout the paper.

The figure's conclusion must mention the control that makes the result informative. If the effect disappears under that control, show the disappearance and narrow the claim.

### 8.5 Figure 5 — Readout dependence and useful feature behavior

Two or three panels:

- Final-gate constant versus patch diagnostic, alongside the invariant classifier response.
- Known-transform correspondence for the declared native feature stages.
- Existing frozen dense-head response, if completed and interpretable.

An optional small image pair can illustrate correspondence with the **same predetermined patch sample** across methods. The numerical population result remains primary.

Do not use a dual y-axis that visually equates arbitrary norm units and accuracy. Separate aligned plots are clearer.

### 8.6 Supplied gate-evolution figure

fig3_gate_evolution.png belongs in the appendix unless trajectories become necessary to explain a measured inference result.

Before reuse:

- Recover the exact run, architecture, checkpoint epochs, and head IDs.
- The shown high-numbered heads require a compatible architecture; do not label a ViT-B figure as ViT-S.
- Plot sigmoid gate values or explicitly centered values, with consistent limits.
- Do not label positive g − 0.5 as “amplification.”
- Display all heads in a compact grid, or disclose the head-selection rule.
- Do not infer semantic specialization from appearance alone.

No training is needed; use saved gate snapshots. If snapshots are missing, remove the temporal claim rather than reconstructing a fictional trajectory.

### 8.7 Supplied layer-analysis figure

fig4_layer_analysis_with_bar.png can become an appendix analysis after correction.

Use:

- Head-wise spatial RMS of g minus its spatial mean.
- Mean attenuation per head/layer as a separate quantity.
- Within-layer cross-checkpoint agreement.
- The terminal classification-inactive layer explicitly marked.

Avoid a mean-absolute-logit bar labeled “spatial activity”: it mixes layer offsets and spatial variation. Head-averaged maps can also cancel opposite head patterns, so report head-wise variability.

The pooled gate correlations in the current summary are inflated by common layer-depth means. The observed mean within-layer agreements are materially lower: about 0.484 within mixing, 0.386 across recipe, and 0.556 within no-mixing, compared with pooled values near 0.938, 0.908, and 0.972. Use the correct level of comparison.

Head identities across independently trained runs are not automatically aligned. A matching procedure chosen to maximize agreement would require disclosure and a null; it is unnecessary for the main claim.

### 8.8 Supplied training-dynamics figure

Retire fig5_training_dynamics.png from the main paper.

It contains old recipe/end-point interpretations that do not match the corrected cohort. If retained in an appendix, regenerate solely from the existing verified logs and include all eligible runs.

Training curves are supporting provenance, not a new contribution, and the revised plan requires no additional training to complete them.

### 8.9 Scientific figure production rules

- Use vector PDF/SVG for plots and diagrams, with a PNG preview if useful.
- Generate charts from a versioned metric table, not manually typed values.
- Use consistent method colors and annotate units, sample counts, feature stage, and uncertainty level.
- Prefer paired dot plots over bars that hide the two/four-run structure.
- Give all comparison heatmaps the declared shared normalization.
- Choose qualitative examples by a fixed image-ID rule or a declared distributional criterion; include typical and failure examples.
- Do not generate scientific heatmaps or model outputs with an image-generation model.
- Write captions that state observation, control, and limit in that order.

The figure-generation code should be runnable without GPUs after the inference outputs exist.

---

## 9. Tables and manuscript result order

### 9.1 Main Table 1 — The existing cohort and its tradeoffs

Use one row per method within each architecture/recipe group. Include:

- Number of trained checkpoints and provenance.
- Top-1 accuracy.
- Fixed-threshold count.
- MAD count.
- One spatial statistic chosen in advance, with reliability.
- A note pointing to geometry and all raw runs in the appendix.

For SAGA versus baseline, show paired differences. For registers, use the matching baseline subset. For TTR, use its own source baseline and identify it as an inference edit.

Do not pool 100-epoch ablations with 300-epoch models. Do not compare an unmatched register mean with the full baseline mean and call the difference paired.

The heading should say “observed tradeoffs in the existing supervised cohort,” not “state of the art.”

### 9.2 Main Table 2 — The decisive functional comparisons

Rows are prespecified intervention contrasts, not every exploratory setting.

Columns:

- Checkpoint group and layer.
- Intervention/control definition.
- Mean paired NLL change.
- Top-1 change.
- Actual injected-energy summary.
- Image-conditional interval.
- Number of checkpoints and per-checkpoint effect range.

If this duplicates Figure 4, move the full numeric version to the appendix and retain a compact main-table subset. The paper should not spend a page showing the same evidence twice.

### 9.3 Main Table 3 — Useful frozen readout

Keep this compact or merge it with Figure 5.

Report geometric correspondence at the designated stage, native classification for context, and any completed existing-head result. Show the terminal-bypass variant under a separate label.

Do not make one “quality score” by averaging accuracy, rank, cosine, and correspondence. Their disagreement is part of the paper.

### 9.4 Appendix tables

Include:

- All eligible classification runs and actual recipes.
- Six 100-epoch arms with the original and corrected threshold versions distinguished.
- Spatial/threshold/reliability statistics by checkpoint.
- Full gate/permutation/energy controls.
- TTR operating points and matched source-checkpoint comparisons.
- Complete CUB/Aircraft family with fine-tuning-seed variation.
- COCO AP, AP-small/medium/large, and other reported task metrics.
- ADE single-/multi-scale results and checkpoint epochs.
- Inference throughput/memory if measured.
- Exclusion/correction ledger.

### 9.5 The order in which results should persuade the reader

1. Existing observations reveal why a single outlier score is insufficient.
2. Scale, occupancy, and boundary controls show which part of the spatial observation survives.
3. Within-checkpoint interventions test the surviving mechanism question.
4. Readout controls establish the functional scope.
5. Existing transfer and external frozen evidence delimit generality.

This order supports an argument. A sequence of “ImageNet, CUB, Aircraft, COCO, ADE” tables would return the paper to an architecture-benchmark claim that the evidence does not support strongly.

---

## 10. Full nine-page main-paper plan

The current ICLR 2027 author guidance allows nine main-text pages at submission; references and appendices are outside that limit. The main argument must stand on its own. [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

The following page budget includes figures/tables within each section.

| Paper section | Pages | Content and purpose |
|---|---:|---|
| Title, abstract, introduction, teaser | 1.50 | State the interpretive problem, prior-work boundary, and strongest completed finding |
| Related work and precise research question | 0.65 | Registers, positional artifacts, gating, TTR, and the narrow unresolved comparison |
| Models, measurements, and intervention protocol | 1.45 | Gate math, three measurement axes, cohort and readout definitions; Figure 2 |
| Prevalence versus spatial structure | 1.25 | Existing evidence plus I1 controls; Table 1 / Figure 3 |
| Functional spatial interventions | 2.00 | I3/I4, energy/ring controls, main causal scope; Figure 4 |
| Readout dependence and useful frozen evaluation | 1.10 | I2/I5, compact Figure 5 / task table |
| Scope, limitations, and existing transfer | 0.70 | Mixed results, small/legacy cohort, recipe limits, optional external confirmation |
| Conclusion | 0.35 | Concrete completed finding and practical evaluation recommendation |
| **Total** | **9.00** | No essential control hidden exclusively in the appendix |

### 10.1 Section-by-section drafting instructions

**Introduction.** Open with the scientific question, not the compute constraint. The fact that no new training was needed is a property of this study, not a claim that SAGA was free to obtain.

**Related work.** Give the original register spatial analysis and TTR functional intervention explicit sentences. Acknowledge earlier vision gating. This is essential to a credible novelty argument.

**Method.** Present the frozen-checkpoint protocol as the analytical method, then explain SAGA as a controllable intervention already represented in the cohort. Keep only the two simple propositions and required equations in the main text.

**Spatial results.** Lead with the actual joint observation, then immediately test its simplest explanation. Do not put the scale-only null five appendix pages away.

**Functional results.** State whether spatial effects survive energy and boundary controls. An effect that does not survive cannot be called a spatial mechanism.

**Readout results.** Explain why the terminal control is expected, then show the nontrivial useful endpoint. The theorem is a guard against misinterpretation, not a substitute for an empirical discovery.

**Scope.** Say directly that existing transfer is mixed, that registers suppress some outlier measures more strongly, and that training-recipe superiority has not been established.

**Conclusion.** Give a conditional practical recommendation supported by the data: report absolute and relative diagnostics at declared feature stages, include functional controls, and assess the readout actually used. Do not claim all outliers are harmless or all register training is unnecessary.

### 10.2 Suggested final contribution paragraph

Use this only after replacing the bracketed part with completed results:

> We make three contributions. First, we characterize how spatial gating and register interventions change event prevalence and spatial organization in a documented cohort of pretrained supervised ViTs. Second, we introduce a controlled comparison of receiver-output interventions that separates gate means, spatial arrangement, boundary composition, and injected update energy, and find [the measured result]. Third, we show how the interpretation depends on the feature stage and readout, using an exact terminal-gate control and [the completed useful endpoint]. Together, these results identify [the specific limitation or mechanism] and provide a reproducible inference protocol for evaluating such interventions.

Do not use “introduce” to claim the first spatial intervention in the literature. It refers to the particular controlled comparison presented here.

---
## 11. Explaining the existing results without inventing a mechanism

A strong paper distinguishes an observed fact, a plausible explanation, and a test that could reject that explanation.

| Existing observation | Plausible explanation | What remains unproven / how the plan checks it |
|---|---|---|
| Fixed-threshold counts fall substantially under SAGA | Receiver-output attenuation changes feature scale and downstream geometry | I1 tests a simple norm-rescaling explanation; MAD and rank are separate diagnostics |
| The normalized first-interior-ring share rises in the mixing groups | Nonuniform norm distributions can cross a common threshold at different rates | The scale-only null may reproduce the change; do not assume selective learned relocation |
| Baseline/SAGA occurrence maps remain correlated | Common data preprocessing, positions, architecture, and learned content statistics can preserve spatial structure | Ring controls, held-out frequencies, and I4 test which part is informative |
| Frozen 0.5 gating changes MAD/rank in the 100-epoch cohort | Attenuation during the original training changes the residual computation and the learned solution | It is not merely an affine transform of final norms; the single-run design cannot isolate every training mechanism |
| Spatial gate E exceeds scalar C but not baseline A in the 100-epoch table | Spatial degrees of freedom may help relative to that scalar arm, with training variation and optimization differences | I3 tests use of spatial arrangement within saved E/SAGA-like checkpoints; it cannot estimate a matched retraining advantage |
| Logit-init-4 arm F has the highest ablation accuracy | Its near-open gate changes the initial computation and sigmoid gradient scale | Init 4 versus 0 also changes optimization dynamics; no new factorial is permitted, so do not claim the cause is identified |
| ViT-B MAD counts increase despite lower fixed counts | Absolute scale and within-image relative tails can move differently | Report both; inspect their distributions with frozen inference rather than choosing the favorable definition |
| No-mixing cosine results disagree by repeat | Effects depend on recipe/model realization; cosine is not a universal quality measure | Preserve both repeats and test the same functional contrast |
| Registers/TTR remove many more threshold events but do not win every accuracy comparison | Different interventions alter different computations; suppression is not the sole determinant of classification | Existing pairs are descriptive; I4/I5 provide controlled within-model responses |
| Transfer is mixed | Task readout, fine-tuning trajectory, and representation changes can interact | Existing-head inference can localize readout sensitivity, but it does not recover a counterfactual training experiment |

For the initialization discussion, sigmoid(4) is approximately 0.982. Its initial sigmoid derivative is roughly 14.15 times smaller than the derivative at zero. Weight decay and the different initial residual computation also matter. State these facts without attributing the final accuracy to any one of them.

For gate/prevalence correlations, a negative value may reflect task-gradient alignment with location-dependent computation. It does not show that the gate was trained to detect sinks, and the selected-layer correlation may not generalize across recipes.

The causal language should attach to the actual temporary intervention: “attenuating these outputs changes this checkpoint's loss.” The explanation of why separately trained SAGA and baseline models differ remains less identified.

---

## 12. Full appendix plan

The appendix provides auditability and complete relevant evidence. It does not hide the controls needed to believe the main conclusion.

### Appendix A — Models, recipes, and provenance

Include the complete checkpoint manifest, actual augmentation flags, nominal versus verified seeds, training schedules, epochs, pretraining/fine-tuning relationships, prefix layout, and source-code state.

Explain the legacy “nomix” correction and the incomplete ViT-B s2 exclusion. Document register fallback backbones and the ADE epoch mismatch.

Provide a diagram or table showing which downstream runs share a pretrained backbone. Three fine-tuning seeds do not become three independent pretraining seeds.

### Appendix B — Implementation and mathematical details

Give gate shapes, ordering relative to SDPA/output projection/residuals, initialization, interpolation behavior, and patch/CLS handling.

Provide the proofs of the two propositions, the zero-slot algebra with its limits, and the matched-energy formulas.

Explain why the existing channel-scale arm is not canonical LayerScale. Cite the canonical method without claiming to have evaluated a faithful replacement. [Going Deeper with Image Transformers](https://arxiv.org/abs/2103.17239).

### Appendix C — Splits, thresholds, and feature stages

List sample-ID manifests and overlap checks; state which information was already inspected.

Record fixed thresholds, calibration provenance, MAD convention, strict inequality, precision, readout stages, register exclusion, and metric implementation versions.

Explain why the historical 100-epoch fixed-threshold counts saturate and provide the corrected calibration separately.

### Appendix D — Complete classification and ablation tables

All eligible baseline/SAGA/register rows, paired differences, and actual repeat counts.

All six 100-epoch arms, including the best initialization arm and the frozen gate. Include the original top-1 values and the corrected inference diagnostics.

No stars based on “mean exceeds two standard errors” heuristics.

### Appendix E — Spatial distributions and reliability

All checkpoint maps, ring profiles, entropy, event counts, top-k rank maps, scale-only nulls, split-half reliability, ring-residual correlations, and held-out empirical-frequency results.

Explain each spatial null's exchangeability assumptions. Include sparse-map failure cases.

### Appendix F — Gate structure and saved evolution

All supported layers/heads, spatial RMS, mean attenuation, within-layer agreement, and the terminal inactive layer.

Use only real saved trajectories with verified run/epoch identifiers. State that raw head IDs across seeds need not represent matching functions.

### Appendix G — Full frozen intervention results

All I3/I4 conditions, fixed permutation/mask indices, paired-loss distributions, injected-energy records, zero-norm cases, tolerance checks, and declared sensitivities.

Include ordinary versus energy-matched perturbations and examples where a naive result disappears under controls.

### Appendix H — Useful readouts and external frozen models

Full correspondence protocol, valid-overlap masks, feature stages, normalization, matching rule, same-coordinate/raw-patch controls, and all method results.

If used, document existing dense heads and public checkpoint identities/licenses/preprocessing. Clearly separate models with native classifiers from feature-only backbones.

### Appendix I — TTR reproduction and attention-direction checks

Upstream provenance, local adaptations, chosen layers/neuron count, exploratory selection history, all available operating points, and paired comparisons.

Define every attention direction and layer timestamp. Distinguish incoming-attention statistics from CLS-to-patch visualizations.

### Appendix J — Complete existing downstream results

CUB and Aircraft full families, COCO full reported metric family, ADE single-/multi-scale results and epochs. State the shared-backbone and single-run limits.

If retaining localization exploration, show all 20 images or a deterministic index and the random/area controls. The existing subset does not establish a localization advantage: baseline/SAGA/register in-box mass is approximately 0.451/0.444/0.550, while the uniform area reference is approximately 0.456.

Do not use the proposed “ring habitat” explanation for transfer as an established fact; the existing controls and rankings do not support it.

### Appendix K — Computational cost and reproducibility

Report original training costs only if logs support them. Separate that historical investment from the new inference budget.

Record hardware, batch sizes, actual forward throughput, feature/attention extraction overhead, and total evaluated image-conditions.

SAGA adds O(LND) gate multiplications. Registers add token interactions: for r added tokens at a sequence length T, the attention-matrix increment is 2rT + r² per layer/head, in addition to other token costs. For fixed r, this increment is linear in T; avoid an exaggerated “quadratic overhead versus free gating” claim.

Measure latency if making a speed claim. The nominal parameter counts alone do not establish end-to-end efficiency.

### Appendix L — Limitations, exclusions, and reproducibility manifest

State:

- Limited pretrained-checkpoint replication and legacy comparisons.
- Weaker supervised recipes than some modern ViT training.
- No new matched strong-recipe training.
- No matched canonical content-gate/LayerScale retraining.
- Simple deterministic translations are not semantic correspondence.
- Perturbation effects are local to the intervention and checkpoint.
- Frozen-head performance can reflect coadaptation.
- Mechanistic masks do not perfectly control all image content.
- Statistical maps and readout controls alone are not a broad theory of registers.

List invalid/unrelated exclusions by reason. Include the scripts/configuration hashes that regenerate every manuscript table and figure.

---

## 13. Implementation handoff and a bounded execution order

This section specifies work to implement. It does not claim that these new scripts or inference results already exist.

### 13.1 Reuse the project instead of creating another training pipeline

Reuse the current model loaders, gate extraction, metrics, and data utilities. Add an inference-only analysis layer with explicit checkpoint/condition manifests.

Suggested new entry points:

| Proposed module | Responsibility |
|---|---|
| analysis/inference_manifest.py | Resolve checkpoint identities, effective recipes, saved epochs, and eligibility |
| analysis/frozen_interventions.py | Temporary gate edits and receiver-output perturbations |
| analysis/run_frozen_eval.py | Shared inference loop and per-image response records |
| analysis/spatial_controls.py | Scale null, rank maps, ring controls, reliability, and empirical-frequency evaluation |
| analysis/frozen_correspondence.py | Known-transform matching with no learned head |
| analysis/build_paper_tables.py | Recompute all manuscript tables from declared input manifests |
| analysis/build_paper_figures.py | Generate the exact figure panels from saved results |

Use the existing code where its implementation matches the audited definition. Do not overwrite historical results or silently update an old metric file in place.

### 13.2 Required output records

Each evaluation record should identify:

- Checkpoint hash and source run.
- Architecture, actual recipe, prefix tokens, and readout.
- Image-ID split hash.
- Native versus edited computation.
- Layer indices in both code and paper conventions.
- Intervention type, mask/permutation ID, epsilon, and measured perturbation norm.
- Metric stage and precision.
- Code/configuration version.
- Per-image target loss, correctness, and relevant diagnostic summaries.

Keep labels and model predictions aligned by image ID, not by assumed dataloader order.

For the implemented TTR construction, load the immutable baseline, add the declared zero-initialized register tokens, and apply the recorded MLP-activation hooks at selected neurons. These are forward-pass activation edits, not optimizer updates or newly learned register embeddings. Record the selected neurons and hook settings; preserve the original saved weights.

### 13.3 Small correctness checks before expensive inference

Use a fixed 32-image smoke subset. These checks target actual risks in this study:

1. Native loader reproduces recorded accuracy/logits within the expected implementation/precision tolerance on a shared subset.
2. Identity intervention produces the native output.
3. The terminal patch gate preserves CLS-only logits.
4. Prefix rows remain untouched by patch masks.
5. Gate permutations preserve the declared means/histograms.
6. Within-ring masks preserve ring counts.
7. Energy-matched edits have the declared injected norm.
8. The geometric transform's patch mapping and overlap mask are correct.
9. Original parameter hashes/state are restored after each temporary edit.

No training loop or optimizer is needed for any check.

The existing SAGA gate implementation assumes one CLS prefix. The shared analysis adapters must infer the actual prefix count for register models rather than reuse that hard-coded slice. Applying a patch intervention to a register row would invalidate the comparison.

### 13.4 The work order

**Pass 1 — Make existing evidence clean.** Complete I0, regenerate the existing tables, audit old figures, and calculate the map/ring summaries already available in arrays.

**Pass 2 — Run the decisive small cohort.** On the two fresh ViT-S mixing baseline/SAGA pairs, complete I1/I2 and the declared I3/I4 contrasts. Keep all existing register results in the descriptive comparison and add their frozen evaluation where checkpoint loading is verified.

**Pass 3 — Complete the relevant replication.** Apply the same locked contrasts to the remaining eligible ViT-S mixing pair(s) and the available register cohort. Use the already-trained no-mixing and ViT-B checkpoints to test scope of the decisive contrast, not a new hyperparameter search.

**Pass 4 — Add a useful endpoint.** Complete I5a, or the already-trained dense-head route if that is better supported operationally. Include a task result that can substantiate any local-feature claim.

**Pass 5 — Optional external confirmation.** Run I6 only for the finding whose generality matters most. Prefer one well-documented model over a broad incomplete survey.

**Pass 6 — Freeze manuscript outputs.** Generate tables/figures, check their numerical provenance, write the full results section, and revise the abstract to the observed outcomes.

This order is a computational prioritization, not a rule to hide unfavorable later results. Any started primary evaluation should be completed or its interruption documented.

### 13.5 Inference is cheaper than retraining, but not zero-cost

Do not promise a wall-clock duration without hardware and a throughput measurement.

A useful workload estimate is

$$
T_{\mathrm{inference}}
\approx
\sum_{m,v}\frac{N_{m,v}}{\mathrm{throughput}_{m,v}}
+
T_{\mathrm{feature/attention\ analysis}}.
$$

For example, two layers with original, mean-collapsed, half-residual, ten ordinary permutations, and ten ring permutations create about 45 distinct conditions per SAGA checkpoint when the shared original is counted once. At 10,000 images, that is roughly 450,000 image-forwards per checkpoint before the smaller energy-control evaluation.

Four checkpoints would therefore mean roughly 1.8 million such image-forwards for that I3 configuration. This is **zero training**, but it should be scheduled honestly. Caching prefixes can reduce repeated computation if implemented and validated correctly.

To control cost:

- Use cached native logits across compatible contrasts.
- Extract expensive SVD/attention diagnostics only on the fixed smaller subset.
- Batch compatible intervention variants where memory permits.
- Restrict replication to the prespecified decisive contrast rather than repeating every diagnostic everywhere.
- Omit the optional full 196-position response map.
- Do not perform a new resolution sweep or every possible dense-task edit by default.

If inference capacity becomes the bottleneck, reduce the breadth of optional analyses and narrow the claim. Do not replace a necessary control with an unsupported assertion.

### 13.6 What is explicitly removed until a future project

No new:

- Training seeds.
- Backbone architectures.
- SAGA or register training.
- Eleven-block SAGA training.
- Content-dependent gate baseline.
- Canonical LayerScale baseline.
- MixUp/CutMix factorial.
- Gate initialization/weight-decay factorial.
- Linear probes or trained feature projections.
- Dense decoders, fine-tuning, distillation, or test-time adaptation.

These are not hidden acceptance prerequisites in this revised plan. Their absence limits the questions we claim to answer.

---

## 14. Adversarial review, revision decisions, and acceptance criteria

### 14.1 The strongest likely objections

| Reviewer objection | Required response in the paper | Residual limitation |
|---|---|---|
| “The original register paper already shows spatial frequency maps.” | Explicitly acknowledge Appendix A; make controlled functional comparison the contribution | Spatial maps alone are not novel |
| “TTR already moves outliers and tests a functional consequence.” | Compare intervention mechanisms and controls; avoid priority claims | A trivial variation on that result would be too incremental |
| “Gated attention in vision already exists.” | Cite the vision gating precedent and distinguish the static receiver gate | Architecture novelty is modest |
| “You are just scaling norms.” | Show scale-only statistical null, relative/rank metrics, and functional controls | If the null explains everything, narrow the claim substantially |
| “Shuffling any trained parameter hurts.” | Include mean, ring, energy, and useful-readout controls | Even a surviving effect is within-model dependence |
| “Your terminal-gate result is obvious.” | Agree that it is a control; do not sell it as the main discovery | A paper based only on that control is weak |
| “The spatial effect explains only a few percent of events.” | Report the actual effect size and test whether it predicts a meaningful response | Reproducibility alone does not imply importance |
| “Your baseline is weaker than modern recipes.” | Scope the cohort and optionally confirm the phenomenon on public strong weights | No stronger-recipe SAGA ranking is available |
| “Registers/TTR look better on the metric you introduced.” | Show that outcome and the complete tradeoff | No universal superiority claim |
| “You have too few trained seeds.” | Show all eligible runs and distinguish image from training uncertainty | No analysis can create missing training replication |
| “You hid negative transfer results.” | Discuss mixed transfer in main text and report complete families in the appendix | Broad local-feature improvement remains unproven |
| “The correspondence task is synthetic.” | Describe it as known-transform consistency and, if possible, add an existing dense head | It is not semantic localization |
| “Your comparisons use mismatched stages/epochs.” | Provide native readout and epoch provenance; exclude invalid direct contrasts | Some historical comparisons remain descriptive |
| “What should a researcher do differently after reading this?” | Give a concrete evaluation protocol and one measured example where it changes interpretation | If the protocol does not alter any substantive conclusion, impact is limited |

### 14.2 Outcome-dependent final phrasing

| Completed result | Final wording |
|---|---|
| Spatial contrasts survive scale/ring/energy controls and appear in a useful readout | “Spatial organization contributes to the measured functional response beyond the tested global and boundary explanations.” |
| Counts and useful readout diverge, but fine gate arrangement is weak | “Outlier suppression is an incomplete readout-dependent diagnostic; fine spatial gate structure is not necessary for the observed effect in this cohort.” |
| A simple scale null explains the maps, while functional differences remain | “Similar apparent outlier improvements can arise from different computations; the functional endpoint, rather than the map alone, distinguishes them.” |
| Only terminal edits create the apparent disagreement | “Terminal patch diagnostics can change independently of a CLS-only classifier.” Treat this as a limited control study, not a full mechanistic breakthrough. |
| All controlled spatial effects are negligible and useful endpoints show no substantive distinction | The intended strong paper has not been established. Report the honest finding and accept a lower acceptance outlook; do not invent novelty through wording. |

The title can remain broad enough for the first three outcomes. The last two require a narrower abstract and a candid reassessment of contribution strength.

### 14.3 Final revision decisions after cross-checking the plan

This revision has deliberately removed several attractive but unsupported claims:

- “SAGA replaces registers.”
- “Spatial sinks were previously undiscovered.”
- “Gate/prevalence correlation proves the mechanism.”
- “Lower norms or higher rank prove better local features.”
- “No new training” means the original SAGA models were training-free.
- The terminal classification-inactive gate should silently disappear from the method.
- Every missing baseline can be replaced by an inference shuffle.
- A null result is automatically evidence of equivalence.
- A visually stronger map should determine which run is published.

The retained position is more defensible: a controlled study using the trained models to distinguish measurements, interventions, and readouts.

### 14.4 Conditions for a credible weak-accept/accept case

Before claiming that the paper has reached that level, all of the following should be true:

1. The abstract states at least one consequential completed finding beyond a known spatial map or the elementary terminal control.
2. The most plausible simple alternative explanation has been tested.
3. The main intervention effect has interpretable magnitude and is not driven by one selected legacy run.
4. The comparison includes the already-trained registers and existing TTR evidence fairly.
5. Any feature-quality statement has a useful frozen endpoint behind it.
6. The full claim is consistent with the no-mixing, ViT-B, and transfer boundaries.
7. Every central figure is reproducible from declared outputs.
8. The closest prior-work overlap is explicit.
9. The paper remains understandable in nine pages.
10. The authors can explain exactly which conclusion would change if a key control were removed.

This standard aims at a defensible expert review. An AI model's “accept” response is neither evidence of correctness nor a calibrated forecast.

### 14.5 Writing and submission details that matter

Use the official anonymous template and keep the essential evidence in the main text. The required AI-use statement should accurately cover assistance with framing, analysis design, source interpretation, and writing where applicable; this work is more than grammar correction. The authors remain responsible for checking every claim, citation, equation, and result. [ICLR 2027 AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).

Add a concise reproducibility statement pointing to the manifest, intervention configurations, metric definitions, and figure-generation scripts.

The venue's reviewer guidance explicitly permits valuable empirical work beyond leaderboard advances. The aim is a new, well-supported insight that changes interpretation or practice, not a larger number of small favorable accuracy differences. [ICLR 2027 Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines).

---

## 15. Source ledger and reference priorities

### 15.1 Supplied material incorporated

| Supplied material | Role in this revision |
|---|---|
| old_SAGA_1_method (1).md | Historical method description, checked against executable code |
| old_SAGA_2_results (1).md | Historical results, superseded where corrected tables disagree |
| old_SAGA_3_project_memory (1).md | Project decisions, training history, and known caveats |
| 1_THEORY_AND_METHODOLOGY.md | Theory hypotheses, revised to match the implemented gate and readout |
| 2_FRAMING.md | Earlier framing options, tested against current evidence and prior work |
| 3_RESULTS_STRATEGY.md | Result organization, replaced by complete relevant-family reporting |
| 4_ICLR_PAPER_PLAN.md | Earlier manuscript structure, replaced by the present nine-page argument |
| SAGA_ICLR2027_FINAL_PAPER_PLAN.md, previous version | Training-heavy plan, superseded by this no-new-training revision |
| SAGA.zip | Primary source for code, configurations, current notes, and result tables |
| results.zip | Legacy outputs and consistency checks |
| fig2_saga_diagram.PNG | Method-figure concept, corrected as specified in Section 8 |
| fig3_gate_evolution.png | Conditional appendix reuse after run/head provenance checks |
| fig4_layer_analysis_with_bar.png | Conditional appendix reuse with corrected spatial-variation quantities |
| fig5_training_dynamics.png | Retire or regenerate from verified saved logs |
| demo_teaser_image.PNG | Replace empirical-looking stylization with actual population/control evidence |

The primary source hierarchy is executable implementation plus verified configuration/checkpoint provenance, then machine-readable results, then project notes, then prior narrative plans. A stronger sentence in an old plan is not additional evidence.

### 15.2 Key repository evidence

- Method: saga/gate.py and saga/vit.py.
- Metric definitions: saga/metrics.py.
- Spatial analysis: tools/sink_address.py, analysis/address_analysis.py, analysis/gate_structure.py.
- Current classification cohort: results/tables/e2_pooled.csv.
- Ablations: results/tables/T2_ablation.csv and T2_ablation_address.csv.
- Gate/spatial summaries: results/tables/gate_agreement.csv and sink_address.csv.
- Population arrays: results/figures_data/Faddr.npz.
- Canonical thresholds: results/diagsplit/fixed_thresholds_canon.json.
- TTR: results/tables/T_ttr.csv, T_ttr_sweep.csv, and third_party/ttr/PROVENANCE.md.
- Transfer: results/tables/T3_finegrained.csv, T4_coco.csv, and T5_ade20k.csv.
- Localization: results/tables/T_localization.csv.
- Corrections and context: results/notes/recipe_erratum.md, ablation.md, sink_address.md, gate reports, and ttr_baseline.md.

The archive does not contain the weight files required for the proposed new forward evaluations. Those analyses are a plan for the user's existing saved models, not results executed during this document revision.

### 15.3 References to prioritize in the manuscript

**Essential closest work:** [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588); [Denoising Vision Transformers](https://arxiv.org/abs/2401.02957); [Quantizable Transformers](https://arxiv.org/abs/2306.12929); [Vision Transformers Don't Need Trained Registers](https://arxiv.org/abs/2506.08010); [Vision Transformers with Self-Distilled Registers](https://arxiv.org/abs/2505.21501).

**Modern mechanistic context:** [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708); [Value-State Gated Attention for Mitigating Extreme-Token Phenomena in Transformers](https://arxiv.org/abs/2510.09017); [DINOv3](https://arxiv.org/abs/2508.10104); [Do All Vision Transformers Need Registers?](https://arxiv.org/abs/2603.25803); [The Spike, the Sparse and the Sink](https://arxiv.org/abs/2603.05498). Distinguish preprints from accepted papers and vision evidence from language-model evidence.

**Evaluation and baseline context:** [Deep ViT Features as Dense Visual Descriptors](https://arxiv.org/abs/2112.05814); [DeiT III](https://arxiv.org/abs/2204.07118); [DINOv2](https://arxiv.org/abs/2304.07193); [Going Deeper with Image Transformers](https://arxiv.org/abs/2103.17239).

**Broader register-function context:** [Registers Matter for Pixel-Space Diffusion Transformers](https://arxiv.org/abs/2605.16147), a 2026 preprint in a different task family. Cite it to delimit the broad claim, not as a directly comparable supervised baseline.

Use citations to delimit novelty and justify an evaluation choice. Avoid a long list of vaguely related papers that does not alter the argument.

---

## 16. Acceptance assessment — subjective, conditional, and not a guarantee

### 16.1 My assessment of the current evidence

The existing project has real assets: numerous completed models, corrected paired comparisons, repeatable spatial structure, several forms of intervention, and useful mixed outcomes. It is not an empty idea requiring a new training campaign.

However, the existing tables and maps alone leave a substantial novelty gap. Spatial artifacts, register interventions, vision gating, and test-time relocation already have close precedents. The present accuracy gains are modest and uncertain, and transfer is not uniformly positive.

**My subjective estimate for a submission consisting mainly of the current results with polished framing is roughly 15–25%.**

This is a judgment range, not a statistically calibrated probability or a calculation from ICLR acceptance rates.

### 16.2 What the no-training plan could achieve

**If the primary inference experiments establish a substantive distinction that survives the scale, ring, energy, and readout controls, and the manuscript is executed well, my subjective estimate is roughly 30–45%.**

For planning purposes, I would use **about 35%**, conditional on that evidence actually being obtained and reported convincingly. I would not assign that probability to the plan document by itself.

A particularly clear finding that generalizes to an official strong pretrained model could improve the outlook. I cannot responsibly translate that into a precise extra percentage.

If the apparent findings reduce to global scaling, an obvious terminal-readout effect, or generic sensitivity to parameter shuffling, I would place the outlook closer to **10–20%**, even with excellent figures.

These ranges are not confidence intervals. The outcome depends on results not yet measured, reviewers, novelty judgments, and writing quality.

### 16.3 The target worth pursuing

The practical target is a paper that an informed reviewer can reasonably call **ICLR-level and a weak accept or accept**, because it establishes a useful controlled finding and handles counterevidence honestly.

No framing can ensure that all AI systems or human reviewers will recommend acceptance. Optimizing for a favorable automated verdict is less useful than making the central claim survive the strongest plausible objection.

**Final recommendation:** proceed with the selected framing, retain the models already trained, and spend the remaining research effort on the decisive frozen-model controls and one useful readout. There is no new training requirement in this plan. Its acceptance potential comes from the scientific result those analyses establish, not from suppressing unfavorable evidence or overstating the method.
