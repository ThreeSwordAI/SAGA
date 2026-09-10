# third_party/ttr — PROVENANCE

## Status: NOT VENDORED. `saga/ttr.py` is a reimplementation.

TASK-10 A1 says: *"First try to fetch the authors' public implementation. If
reachable: vendor it under `third_party/ttr/` **with its LICENSE file
intact** ... If not reachable: reimplement from the method description in
`saga/ttr.py` and say so explicitly in the note."*

The repository **is** reachable, but it **cannot be vendored on the task's own
terms**: it has no license file of its own. Neither branch of A1 applies
verbatim, so this file records the reconciliation (CLAUDE.md: *"When a task
file and the actual repo disagree, say so explicitly and propose the smallest
reconciliation — do not silently improvise"*).

**Smallest reconciliation, and what was actually done:** the authors' code was
READ to implement the method exactly — no guessing about which variant was
built — but no line of it was copied into this repository. `saga/ttr.py` is
original code written against the method as their implementation defines it,
with every deviation listed below. Reading public code to understand an
algorithm is ordinary practice; redistributing unlicensed code inside a
repository headed for publication is not.

## Upstream

| | |
|---|---|
| Paper | Nick Jiang\*, Amil Dravid\*, Alexei A. Efros, Yossi Gandelsman, *Vision Transformers Don't Need Trained Registers*, NeurIPS 2025 Spotlight |
| arXiv | https://arxiv.org/abs/2506.08010 |
| Code | https://github.com/nickjiang2378/test-time-registers |
| Commit read | `860df43515c8d8e9e90952af25a46c26e4469570` (2025-09-19T02:38:44Z, `main`) |
| Date read | 2026-09-09 |
| Stars at read | 187 |

### The licensing finding, in full

- The GitHub API reports `"license": null` for the repository.
- A recursive tree listing at the pinned commit contains exactly **one**
  license file: `dinov2/LICENSE`. That is Meta's license, and it covers the
  **vendored DINOv2 subtree** inside their repo — not the authors' own TTR
  code.
- There is no top-level `LICENSE`, `LICENSE.md`, `COPYING`, or `NOTICE`.
- The authors' own contribution — `shared/algorithms.py`, `shared/hook_fn.py`,
  `shared/hook_manager.py`, `shared/utils.py`, `custom_model/*`,
  `clip/clip_hook_manager.py`, `clip/clip_state.py`,
  `dinov2/dinov2_hook_manager.py`, `dinov2/dinov2_state.py`,
  `register_neurons.ipynb` — therefore carries **no grant of rights**. Under
  default copyright that is all-rights-reserved.

If the authors add a license later, vendoring becomes possible and this
decision should be revisited; the pinned commit above makes the comparison
exact.

## Files read (for the record)

`shared/algorithms.py`, `shared/hook_fn.py`, `shared/hook_manager.py`,
`shared/utils.py`, `custom_model/custom_state.py`,
`custom_model/custom_hook_manager.py`, `dinov2/dinov2/hub/hook_fn.py`,
`dinov2/dinov2/hub/register_neurons.py`, `dinov2/dinov2_hook_manager.py`,
`README.md`.

## The method as their code defines it

**Step 1 — register-neuron detection** (`shared/algorithms.py::find_register_neurons`):

1. For each of `processed_image_cnt` images (their default 500), run the
   unpatched model with logging hooks.
2. Take the block output at `detect_outliers_layer` (default `-1`, the last
   block) and compute each token's L2 norm.
3. "Register locations" = tokens whose norm exceeds `register_norm_threshold`
   (their default `30`). Images with none are skipped.
4. For every layer, take the MLP hidden activations at those token positions,
   take `abs()`, and average over the register locations → one score per
   neuron per image.
5. Average over the images that had register locations, flatten, and sort
   descending → `[(layer, neuron, score), ...]`.
6. An optional sparsity filter (`apply_sparsity_filter`, default **off**)
   multiplies scores by an indicator that the neuron is mostly inactive.

**Step 2 — the intervention** (`shared/hook_fn.py::activate_on_registers`,
registered as a forward hook on each selected layer's activation module by
`shared/hook_manager.py::finalize`):

1. The model must support extra tokens **initialised to zeros** — stated in
   their README under *Adding New Models*: *"you should modify the model code
   to enable adding in extra tokens initialized to zeros. This is necessary
   for creating our 'test-time' registers to shift outliers from the image
   to."*
2. At each layer holding selected neurons, the extra tokens' activations at
   those neurons are set to `scale * sign_max(activations over all tokens at
   those neurons)`, where `sign_max` (their `shared/utils.py`) returns the
   element of largest **absolute** value with its sign preserved.
3. The same neurons are then **zeroed on the patch tokens**
   (`normal_values="zero"`, their default; `"mean"`, `"only_outliers"` and
   `"same"` also exist).
4. CLS is never modified — their slice is `output[0, 1:-num_registers]`.

**Hook point.** `Dinov2HookManager.neuron_activation_component` returns
`model.blocks[layer].mlp.act` — the activation function's output, i.e. the
hidden state `fc2` consumes. `layer_output_component` returns
`model.blocks[layer]`. Both map directly onto this repo's timm-based ViTs.

## Deviations in `saga/ttr.py`, and why

| # | Theirs | Ours | Why |
|---|---|---|---|
| 1 | Extra tokens **appended**: `[CLS][PATCH][REG]` | Extra tokens as **prefix**: `[CLS][EXTRA][PATCH]` | TASK-10 A2.2 requires patch count and ordering to be untouched so `saga/metrics.py`, `tools/diagnose.py` and `tools/sink_address.py` work unchanged — they all slice `x[:, P:, :]`. Self-attention is permutation-equivariant and the extra tokens carry no positional embedding, so the placements are equivalent; `test_prefix_placement_matches_appended_placement` measures it (max abs difference ~2e-6 on tokens of norm ~16, i.e. fp32 round-off). |
| 2 | Outliers detected over the **whole** token sequence, CLS included | Outliers detected over **patch tokens only** (`x[:, P:]`) | Every τ in this repo is calibrated on `last_block_patch_norms` (see `fixed_thresholds_canon.json`'s own `definition` string). CLS norm is a different distribution — `cls_norm_ratio` exists precisely because it differs from the patch median — so thresholding CLS with a patch-calibrated τ would be a category error and would pollute the scores with CLS-specific activations. |
| 3 | Absolute threshold `register_norm_threshold=30`, tuned to CLIP/DINOv2 | The base cell's **canon τ**, read from `results/diagsplit/fixed_thresholds_canon.json` | Same criterion, threshold sourced from this repo's per-cell calibration instead of a constant from another model family. Our norm scales differ by an order of magnitude across cells (`vit_small\|mixup` 20.85, `vit_base\|mixup` 127.31, `vit_small\|nomix` 22.86), so one hardcoded number cannot serve. Never recalibrated on a patched model (TASK-10 acceptance). |
| 4 | Batch size 1 (`output[0, ...]` throughout) | Batched, with `sign_max` reduced **per image** | 10 000- and 50 000-image passes are not feasible one image at a time. Reducing per batch element is the faithful generalisation of a batch-1 reduction. |
| 5 | Hook mutates the activation in place | Hook returns a modified **copy** | Safe under autograd as well as `no_grad`; the in-place version would fail if anything ever needed gradients through the patched model. |
| 6 | `normal_values` ∈ {`zero`, `mean`, `only_outliers`, `same`} | {`zero`, `mean`, `same`} | `only_outliers` uses a mean+1σ within-patch rule that is a third outlier definition on top of τ and MAD; adding a third convention to this repo needs a reason, and TASK-10 does not ask for one. `zero` (their default) is what the sweep uses. |
| 7 | Optional sparsity filter | Not implemented | Off by default upstream, and unused by the sweep. Recorded here so the omission is explicit rather than silent. |
| 8 | Score = mean \|activation\| at outliers | Same, **plus** an optional contrast criterion | TASK-10 A2.1 suggests "outliers minus rest". Their criterion is the **default** (`mean_abs_act_at_outliers_v1`); the contrast variant is selectable (`mean_abs_act_outliers_minus_rest_v1`). The criterion string is written into every `neurons.json`, so no result is ever ambiguous about which was used. |

## What is NOT reimplemented

Their hook-manager abstraction (`HookMode`, log/intervene registries,
attention-map capture) has an equivalent in this repo already —
`saga/attn_extract.py` for attention capture, and plain forward hooks
elsewhere — so `saga/ttr.py` uses the repo's own idioms instead. Their CLIP,
DINOv2 and LLaVA adapters are irrelevant here: TASK-10 applies TTR to this
project's supervised ViT-S/B checkpoints.
