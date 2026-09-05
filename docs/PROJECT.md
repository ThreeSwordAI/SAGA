# PROJECT.md — SAGA project state

Last updated: 2026-09-05

> **ERRATUM (2026-09-05):** The legacy "nomix" label is false — the legacy
> config loader never applied the nomix augmentation block, so every legacy
> run trained WITH MixUp+CutMix. All statements below that contrast recipes
> on legacy data are void; legacy ViT-S "nomix" runs are additional repeats
> of the ViT-S/mixup cell; legacy ViT-B "nomix" recipe is pending
> verification. See `results/notes/recipe_erratum.md`. The first true-nomix
> runs are the e2r ViT-S nomix chains. Full rewrite at the next milestone.

*(Reconciliation note: TASK-06B prescribed prepending the erratum to an
existing PROJECT.md, but no PROJECT.md existed anywhere in the repo. This
stub was created to carry the erratum; the full project description is
queued for the milestone rewrite the erratum refers to. Until then, the
running project record is `docs/TASK_LOG.md`.)*
