# TASK_LOG — session memory (append-only, newest last)

Read this at the start of every session. One entry per task: what was done,
commits, and what is pending from the HPC.

---

## 2026-08-27 — TASK 00 (bootstrap)

**Done (local, by Claude Code):**
- Tagged `pre-fix-audit` on pre-task HEAD (`ac4121b`).
- Extended `.gitignore`: `*.pth`, `*.pt`, `results/**/ckpt/`, `results/**/attn/`,
  `results/**/*_norms.npz`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `wandb/`.
  Verified with `git check-ignore` that results JSON/CSV/`gates/phi_e*.npz`/
  `figures_data/` stay trackable.
- Installed local `.git/hooks/pre-push` guard — pushes require `I_AM_HUMAN=1`.
- Created `results/` contract tree + `results/README.md` (with `.gitkeep`s).
- New `saga/run_registry.py` (`create_run` / `finalize_run` / `file_sha256`).
- New `tools/make_manifest.py` (checkpoint inventory CSV with best-effort
  exp/arch/recipe/variant/seed guesses).
- New `docs/HPC_WORKFLOW.md` (this round-trip, one page).
- `requirements-dev.txt` added; `requirements.txt` timm pin raised to
  `timm>=1.0.0` (register-token models + `forward_intermediates`; audit M6)
  — **HPC env must be updated** (`pip install -r requirements.txt -r requirements-dev.txt`).
- Local dev venv created at `../.venv` (outside repo): torch 2.13.0+cpu,
  torchvision 0.28.0+cpu, timm 1.0.28, Python 3.12.2.

**Commit:** `[TASK-00] bootstrap: gitignore, results contract, run registry, manifest tool, docs`

**Pending from HPC:** nothing for TASK 00 itself (manifest run happens at the
end of TASK 01, see next entry).

---

## 2026-08-27 — TASK 01 (salvage kit)

**Done (local, by Claude Code):**
- `saga/metrics.py` — canonical metric module: `infer_num_prefix_tokens`
  (promoted from `figures/make_table4_sink_threshold_from_tar.py`; fixes B2),
  `token_norms`, `sink_counts_mad` (primary: median+5·MAD), `sink_counts_gauss`
  (μ+kσ, k∈2..6), `sink_counts_fixed`, `oversmoothing_pairwise` (closed form;
  fixes B3), `oversmoothing_consecutive_legacy` (comparability only),
  `effective_rank`, `cls_norm_ratio`, `reg_norm_mean`, `cls_attn_share`,
  `compute_diagnostics` (hooks on every block, fp32, B5 schema).
- `saga/attn_extract.py` — `capture_attention` context manager: explicit
  softmax(qkᵀ·scale) maps for timm `Attention` AND repo `GatedAttention`,
  outputs stay bit-identical (original forward produces them), `blocks=`
  memory guard.
- `tools/model_factory.py` — `build_model` mirrors
  `classification/tools/train.py::build_model` argument-for-argument;
  `load_checkpoint` unwraps trainer dicts, strips `module.`, strict=True.
- `tools/eval.py` — exact full-val evaluation (fixes B1): trainers' val
  transform code path (timm create_loader defaults), manual
  `indices[rank::world_size]` sharding (no DistributedSampler), float64 count
  accumulation + all_reduce, asserts n == len(dataset) == 50000 before writing.
- `tools/build_diag_split.py` — frozen seeded 10/class val split (relative
  paths + class ids), importable `DiagSplitDataset`.
- `tools/diagnose.py` — B5 diagnostics JSON + `<name>_norms.npz` (fp16 arrays).
- `tests/` — T1..T7 per spec + compute_diagnostics integration test +
  make_manifest regression test; `pytest -q`: **31 passed** (~12 s, CPU).
- Adversarial review workflow (5 reviewers + verifier) confirmed one defect,
  fixed: manifest field guesses now use the ROOT-relative path with bounded
  'saga' matching (an ancestor /SAGA/ dir had labeled every row variant=saga).

**Commit:** `[TASK-01] salvage kit: metrics, eval, diagnose, attn extraction, diag split, tests`

**Pending from HPC (before TASK 02):**
- `results/legacy/checkpoint_manifest.csv` (tools/make_manifest.py on the real
  checkpoint roots; human confirms/fills arch/recipe/variant/seed for the 27
  headline runs).
- `results/diagsplit/val_diag_split.json` (tools/build_diag_split.py on an
  extracted ImageFolder ImageNet copy with val/).
- `pytest -q` green on the HPC after env update (`timm>=1.0.0`).

**Update 2026-08-27 (later):** all three came back (commit `bffefbf`): split
valid (10000 imgs, 1000 classes, 10/class, seed 0), manifest 78 rows all
hashed, HPC pytest 31 passed. Human ground truth: e2 = 12 headline runs, ONE
surviving repeat each (~3 repeats overwrote the same dir; no seed control) —
the "27 runs × 3 seeds" premise in TASK_00_01 is obsolete. Legacy history
JSONs are partial (resume overwrote them). Standing rule from the human:
everything written from now on must be idempotent / append-safe.

---

## 2026-08-27 — TASK 02, PHASE 1 (rederive prep)

**Done (local, by Claude Code):**
- `saga/metrics.py` + `oversmoothing_pairwise_nosink` (median+5·MAD exclusion,
  same convention as `sink_counts_mad`; closed form on survivors; skip <2
  survivors) wired into `compute_diagnostics` → diagnose JSON now carries
  `oversmooth_pairwise_nosink` and `nosink_excluded_mean`. Test T8.
- Manifest filled per human authorization (separate commit `0c44771`):
  all 24 e2 rows recipe∈{mixup,nomix} (12 blanks → mixup), seed → `rlast`;
  e1/e3/e6 untouched.
- `tools/check_done.py` — requeue guard (output JSON exists + ckpt_sha256
  matches ⇒ skip step).
- `tools/compute_fixed_thr.py` — per-arch τ from nomix-baseline last diagnose
  norms; LOWER medians throughout (matches torch.median / sink_counts_mad).
- `tools/apply_fixed_thr.py` — backfills ONLY `sink_fixed_thr` +
  `fixed_thr_value` into diag JSONs from `_norms.npz`; idempotent.
- `tools/gen_rederive_jobs.py` → generated `scripts/rederive_e2.sh`:
  24 e2 checkpoints × (eval + diagnose) = 48 guarded sequential steps,
  failures append to `results/legacy/rederive_failures.log` and continue,
  ends with compute/apply fixed-thr; ~9 h on one GPU (est.).
- **e3 EXCLUDED** from re-derivation: its 3 checkpoints are full `ViTDetector`
  state dicts (backbone+neck+head, `detection/tools/train.py`), not plain
  ImageNet classifiers — strict load via model_factory would rightly fail.
- Review workflow confirmed + fixed a requeue-safety bug: `tools/diagnose.py`
  wrote its JSON (the completion marker) BEFORE the npz; now npz first,
  JSON last.
- `pytest -q`: **41 passed** (~12 s, CPU). No training-code edits.

**Commits:** `0c44771` (manifest fill), plus
`[TASK-02] nosink metric, fixed-thr tools, job generator`

**Pending from HPC (before PHASE 2):**
- 24 × `results/legacy/eval/e2_*.json`, 24 × `results/legacy/diag/e2_*.json`
  (with nosink fields), `results/diagsplit/fixed_thresholds.json`.
- `rederive_failures.log` verified absent/empty; `*_norms.npz` KEPT on the
  HPC (git-ignored; needed later for histograms).

**Update 2026-08-28:** everything came back clean (commit `ca345fd`): 24 eval
+ 24 diag JSONs (all n_images correct, all ckpt_sha256 match the manifest,
nosink + fixed-thr fields present), `fixed_thresholds.json`
(vit_small τ=19.8594, vit_base τ=68.7188, k=5), no failures log.

---

## 2026-08-28 — TASK 02, PHASE 2 (corrected tables, F3 draft, Gate-1 report)

**Done (local, by Claude Code; no GPU, no HPC needed):**
- 2.1 completeness: all 12 e2 runs have eval(best), eval(last), diag(best),
  diag(last) with manifest-matching `ckpt_sha256` — zero gaps.
- 2.2 `analysis/build_legacy_tables.py` → `results/tables/legacy_e2_corrected.csv`
  (12 rows; diagnostics from `last.pth`; sha mismatch ⇒ MISSING, never a
  wrong-checkpoint number). Every cell mechanically cross-checked against the
  raw JSONs via an independent code path — zero discrepancies.
- 2.3 memo: `top1_best_minus_last` column; flags |Δ|>0.25: ViT-B/mixup
  baseline (+0.570) and ViT-B/mixup saga (+0.750) need discussion.
- 2.4 `analysis/collect_F3.py` → `results/figures_data/F3_legacy.csv` (sha-
  validated against manifest); `plotting/plot_F3.py` →
  `results/figures/F3_draft.pdf` (committed with -f; `*/figures/` is ignored).
- 2.5 `analysis/build_gate1_report.py` → `results/notes/gate1_report.md`
  (generated programmatically — no hand-typed numbers). **Verdicts:
  ViT-S/mixup PASS, ViT-B/mixup PASS, ViT-S/nomix PARTIAL (registers did not
  worsen oversmoothing), ViT-B/nomix PARTIAL (SAGA sink 15.19 > baseline
  10.31). Overall: PARTIAL.** Gate-1 decision is the human's.
- Review workflow confirmed + fixed: collect_F3 now treats a manifest-sha
  mismatch as a gap (was: silently plottable); diag(best) completeness clause
  now test-pinned; report degrades gracefully (INCOMPLETE) on MISSING cells.
- `pytest -q`: **45 passed**. No training-code edits.

**Commit:** `[TASK-02] corrected legacy tables + F3 draft + gate1 report`

**Pending from HPC:** nothing. Next: human reads `results/notes/gate1_report.md`
and decides Gate 1; awaiting pointer to the next task.

---

## 2026-08-28 — TASK 02B, PHASE A (sink robustness + relocation, Gate-1 addendum)

**Done (local, by Claude Code):**
- A1 `analysis/build_sink_robustness.py` → `results/tables/sink_robustness.csv`
  (12 runs × 7 thresholds, sha-checked vs manifest) +
  `sink_robustness_verdict.csv` (7 thresholds × 4 cells × {S,R} vs B, with
  the per-arch-τ caveat). Key facts: ViT-B/nomix shows S<B under μ+3/4/5σ
  AND fixed τ but S>B under MAD and μ+6σ (the H1-predicted shape);
  ViT-B/mixup baseline+saga SATURATE the fixed τ (≈196/196 — τ was
  calibrated on the nomix baseline; mixup ViT-B norms sit entirely above it).
- A2 `analysis/collect_F6.py` + `plotting/plot_F6.py` →
  `results/figures_data/F6_legacy.csv` (144 rows) +
  `results/figures/F6_draft.pdf` (committed -f). SAGA's CLS-norm ratio ends
  above baseline in all 4 cells; its CLS attn share ends below baseline.
- A3 `analysis/build_gate1_addendum.py` → `results/notes/gate1_addendum.md`
  (programmatic; corrected table, robustness matrix + factual readings +
  saturation note, F6 final-block observations, best-vs-last recap, open
  questions; §5 norm-scale marked PENDING PHASE B — the script self-fills it
  from `*_normstats.json` when Phase B lands, then just re-run it).
- Phase-B script written now: `tools/summarize_norms.py` (CPU; per-arch
  shared log bin edges; lower-median convention; idempotent — skips only on
  matching sha AND matching current edges).
- Review workflow confirmed + fixed 3 memo-generator defects: MISSING
  verdict entries were rendered as "equal"; forced H1/H2 binary could print
  a false "H2" beside contradicting numbers (now 4-way: H1 / H2 / both /
  neither); saturation span pooled across arches (now per-arch). All pinned
  by tests.
- Cross-checks vs raw JSONs (independent path): robustness 12×7, verdict
  14×4 recomputed, F6 144×2 — zero errors. `pytest -q`: **54 passed**.

**Commit:** `[TASK-02B] sink robustness + F6 draft + memo (phase A)`

**Pending from HPC (Phase B, ~5 min CPU):**
- `python tools/summarize_norms.py --diag-dir results/legacy/diag` →
  commit `results/legacy/diag/*_normstats.json` and push.
- Then Phase C locally: A1 histogram draft + memo §5 finalized.

**Update 2026-08-28:** Phase B back (commit `2eab70f`): 24 normstats files,
all schema/sha/bin-edge checks clean.

---

## 2026-08-28 — TASK 02B, PHASE C (histograms + memo finalized)

**Done (local, by Claude Code):**
- C1 `plotting/plot_A1.py` → `results/figures/A1_norm_hist_draft.pdf`
  (committed -f): per-cell overlaid log-x/log-y norm histograms from the
  normstats (per-arch shared bins), fixed τ solid + per-variant MAD
  thresholds dashed. The ViT-B/mixup fixed-τ saturation is directly visible
  (baseline+SAGA bulks sit entirely right of τ).
- C2 memo §5 finalized by re-running `analysis/build_gate1_addendum.py`.
  Verified by independent recompute from raw normstats (0 errors).
  SAGA-vs-baseline relative changes (threshold / p99.9 / max):
  ViT-S mixup −23.7/−56.3/−54.5 → neither pattern (extremes shrank MORE
  than the threshold); ViT-S nomix −14.9/−52.2/−52.6 → neither;
  ViT-B mixup +28.7/−52.5/−45.7 → neither (threshold GREW, extremes
  shrank); **ViT-B nomix −23.9/+54.5/+39.4 → BOTH signatures (threshold
  shrank while extremes grew — fewer-but-larger outliers).**
- Review workflow: 0 findings. `pytest -q`: **54 passed**.

**Commit:** `[TASK-02B] histograms + memo finalized (phase C)`

**Pending from HPC:** nothing. TASK 02B complete. Next input: the human's
Gate-1 framing decision (memo: `results/notes/gate1_addendum.md`).

---

## 2026-08-28 — TASK 02C, PHASE A (metric v2 + gate/forensics tools)

Human decisions recorded in the task file: primary sink metric moves to a
per-(arch, recipe) fixed τ (v1's per-arch τ saturated on ViT-B/mixup);
the ViT-B/mixup cell is QUARANTINED pending forensics.

**Done (local, by Claude Code):**
- A1 `tools/compute_fixed_thr.py --per-cell` → v2 taus keyed
  `"<arch>|<recipe>"` from each cell's OWN baseline last norms; refuses to
  write to the v1 default path (v1 file untouched, provenance).
  `tools/apply_fixed_thr.py --version v2 --thr-file ...` → adds ONLY
  `sink_fixed_v2` + `fixed_thr_v2_value` (v1 fields never touched;
  idempotent; now ATOMIC in-place rewrite via tmp+os.replace after review
  confirmed a kill-mid-write could truncate a results JSON).
- A2 `tools/extract_gate.py`: (default) dumps SAGA gate logits φ
  (`blocks.{i}.attn.gate.phi` [H,196] per block → npz [L,H,N]) +
  per-layer sigmoid-stats sidecar (mean/std/min/max, frac<0.4/<0.25/>0.75,
  NaN/Inf), npz-first/marker-last, sha-keyed skip; (--forensics) one CSV row
  per e2 checkpoint: top-level keys, epoch, top1, best_top1, last LR,
  optimizer step count, scaler/EMA presence, model tensor count/params.
- Tests: 11 new (per-cell keying incl. distractor files, v2-only writes +
  idempotency + no-tmp-left, v1 behavior unchanged, φ round-trip on a real
  SAGA model with layer-order check, φ stats known values + NaN, non-SAGA
  rejection, forensics full/missing/raw-state-dict rows, v1-path guard).
- Review workflow: 1 confirmed defect fixed (non-atomic JSON rewrite),
  2 hardenings applied from refuted-but-noted findings (stale-marker unlink
  in extract_gate; --per-cell v1-path guard).
- `pytest -q`: **65 passed**. No training-code edits.

**Commit:** `[TASK-02C] metric v2 + gate extraction + forensics tools (phase A)`

**Pending from HPC (Phase B, ~10 min CPU):** v2 thresholds + v2 apply +
gate dumps (8 SAGA ckpts) + forensics CSV (24 ckpts); commit + push per the
task file's Phase-B block. Then Phase C locally.

**Update 2026-08-28:** Phase B back (`417f03c`), all outputs verified clean
(v2 taus incl. vit_base|mixup=226.75; v2 fields added with v1 untouched;
8 φ dumps, no NaN/Inf; 24-row forensics CSV).

---

## 2026-08-28 — TASK 02C, PHASE C (v2 robustness, gate maps, forensics note)

**Done (local, by Claude Code):**
- C1: `sink_robustness{,_verdict}.csv` rebuilt with `sink_fixed_v2` (v1
  kept; 8 thresholds × 2 × 4 cells). **No saturation under v2** (max count
  175.90/196, ViT-B/nomix registers, under the ≥95% flag). Under v2:
  SAGA < baseline in 3 cells; **ViT-B/mixup still S>B (44.73 vs 13.81)**;
  registers < baseline except ViT-B/nomix R>B.
- C2: `plotting/plot_gates_legacy.py` → `results/figures/
  gates_legacy_draft.pdf` (4 per-layer×head map pages + summary panel,
  committed -f). All four SAGA runs share a mild gate profile (means
  0.48–0.65, peak at layer 8, final layer ≈0.5).
- C3 (amended): `analysis/build_bmixup_forensics.py` →
  `results/notes/bmixup_forensics.md` with the 24-checkpoint completeness
  table. Epoch semantics verified from the trainer save code (0-indexed;
  last.pth only at k·25−1 or 299; complete ⇔ last==299). **Finding: all
  three ViT-B/mixup runs are INCOMPLETE — last.pth at epochs 199 (baseline),
  249 (registers), 74 (SAGA, LR 8.5e-4); SAGA's best.pth (89) is newer than
  its last.pth (74). All 9 other runs complete at 299, LR 1e-6.** Evidence
  table incomplete-vs-pathological, no verdict.
- C4: addendum §7 postscript (v2 taus, criterion-stated saturation line,
  v2 orderings, CSV-derived completeness summary).
- Review workflow confirmed 4 facts-discipline defects, all fixed:
  "never resubmitted" speculation replaced by the derivable interval;
  blanket "far below" saturation line now states criterion + max count;
  two hardcoded prose blocks (addendum forensics sentence, §3 gate-profile
  summary) now computed from the data.
- Note numbers verified against source files by independent recompute
  (0 errors). `pytest -q`: **65 passed**. No training-code edits.

**Commit:** `[TASK-02C] v2 robustness + gate maps + forensics note (phase C)`

**Pending from HPC:** nothing. TASK 02C complete. Next input: the human's
decision on the quarantined ViT-B/mixup cell (retrain vs drop) and Gate-1
framing.

---

## 2026-08-31 — TASK 05, PHASE A (trainer fixes + rerun launch tooling)

**Done (local, by Claude Code). First task allowed to modify the trainer.**
- `classification/tools/train.py` rewritten for the e2r runs, math
  legacy-identical (verified: build_model verbatim; timm loader args
  identical + explicit timm defaults for prefetcher/persistent workers;
  same criteria/optimizer/LR-scaling/cosine args; per-iter step_update kept,
  the legacy per-epoch scheduler.step(epoch+1) removed after EMPIRICALLY
  verifying it is a value no-op at t_in_epochs=False). New: seeding (M4,
  per-rank streams reproduce legacy independence), exact full-val with
  all_reduce (B1) + best on the reduced value, results/runs/<run_id>/
  contract (meta/config/log.csv append-safe), atomic last.pth EVERY epoch
  with per-rank RNG + sampler epoch + schedule geometry, `--resume auto`
  (same command fresh/resume; log dedup; geometry guards: steps_per_epoch
  drift = hard error, --max_epochs change = rebuilt scheduler warning),
  φ dump every epoch, diag every 10 epochs (norms-only), optional
  grad-φ logging.
- `configs/e2r_matrix.yaml` (10 runs; log_grad_phi only on the two
  designated ViT-S saga s1 runs), `scripts/gen_slurm_chain.py` →
  10 sbatch+submit chains (singleton + afterany; header per legacy e2
  script), `scripts/sync_results.sh`.
- Tests: 10 new (T-eq incl. 1-step loss/grad vs legacy path; T-resume with
  poisoned-row dedup; T-atomic kill; T-contract; geometry-drift refusal;
  fresh-start leftover-log truncation). `pytest -q`: **75 passed**.
- Review workflow (20 agents): 9 confirmed findings → 4 fixed in code
  (submit-script extend hazard → singleton; scheduler-geometry poisoning →
  guards; cross-rank RNG correlation → per-rank reseed; leftover-log
  duplicates → fresh-start truncation). **1 LAUNCH-BLOCKING question for
  the human: per the committed legacy code, variants_nomix.yaml's top-level
  augmentation block was silently DROPPED by load_config — the legacy
  "nomix" runs may have trained WITH mixup. Must be settled on the HPC
  (grep the legacy resolved config) BEFORE launching any nomix chain.**

**Commit:** `[TASK-05] trainer fixes (seeding, full-val, resume, contract) + launch tooling`

**Pending from HPC:** human answers the nomix question; smoke run; then the
chain submissions (see the task's end-of-task block). Progress arrives via
`scripts/sync_results.sh` pushes (log.csv, diag, gates, meta per run).

**Update 2026-09-04/05:** smoke PASSED (steps/epoch 1251 = legacy). Six
mixup chains ran to completion (300 contiguous epochs each; resume machinery
worked in production). SAGA wins top-1 in all three cells (S/mixup s1 +0.47,
s2 +0.83, B/mixup +0.22 on best). Ground truth from HPC: legacy "nomix"
config dump shows mixup 0.8/cutmix 1.0 — the nomix label is FALSE (see
TASK-06B). ViT-B/mixup diag: SAGA oversmooth 0.324 vs 0.733 but MAD-sinks
18.7 vs 9.5 — the ViT-B MAD pattern reproduces on a clean seeded run.

---

## 2026-09-05 — TASK 06 PHASE 1 + TASK 06B PART 1 (recipe-identity correction)

TASK 06 had NOT been started; per 06B 1.1 its Phase 1 tooling was built now,
with 06B's corrections layered on. Reconciliations recorded:
- **PROJECT.md did not exist** → created `docs/PROJECT.md` as a stub carrying
  the mandated erratum verbatim; full description queued for the milestone
  rewrite.
- **TASK-06 1.2 (extend v2 with vit_base|mixup) is unsatisfiable**: that key
  already exists in `fixed_thresholds_v2.json` (TASK-02C, calibrated on the
  VOID legacy ViT-B baseline, epoch-199). Reconciliation (recorded in the
  erratum note's "Threshold governance"): v2 is FROZEN as TASK-02C
  provenance; the e2r-calibrated ViT-B/mixup τ lives in the CANON file; v2
  is never applied to e2r run dirs.
- derive_runs is a runtime DRIVER, not a script generator (skip-guard shas
  must be hashed beside the HPC-only checkpoints; documented in the tool).

**Done (local):**
- `tools/derive_runs.py` — idempotent eval+diagnose+summarize driver over
  `results/runs/e2r_*` with a run-COMPLETION gate (end_time + final log
  epoch; mid-training checkpoints are never derived as "final"; incomplete
  runs deferred, not failed) and full failure logging.
- `analysis/check_eval_consistency.py` — eval(last) vs the epoch-299 log
  row; truncated runs are themselves WARNINGs.
- `tools/compute_fixed_thr.py --cell CELL=NPZ` — extend/create thresholds
  files; existing keys/definition immutable, k-mismatch refused, no-op
  reruns leave the file bytes untouched, v1 path unreachable.
  `--definition canon` creates `fixed_thresholds_canon.json` (per
  (arch, recipe_actual), seeded-s1-baseline calibration).
- `tools/apply_fixed_thr.py --version canon` (+ `--runs-root`): fields
  `sink_fixed_canon`/`canon_thr_value` only; legacy dirnames remapped per
  the erratum (ViT-S nomix→mixup; ViT-B nomix→pending, skipped); run-dir
  diags resolve recipe from their own config.resolved.yaml. v2 also accepts
  run-dir paths. v1/v2 fields untouched everywhere.
- `analysis/build_recipe_erratum.py` → `results/notes/recipe_erratum.md`
  (proof recomputed live: legacy nomix-vs-mixup resolved diff = ZERO keys;
  true-nomix = exactly mixup_alpha+cutmix_alpha → 0). `docs/PROJECT.md`
  erratum. Manifest gained `recipe_actual` (ViT-S e2 = mixup, ViT-B e2 =
  pending, others blank) — committed separately (`d0f385e`).
- Review workflow (11 agents): 5 confirmed findings, all fixed (the v2-key
  conflict above; k-mismatch guard; completion gate; summarize_norms
  failure capture; consistency checker epoch-299 semantics).
- `pytest -q`: **89 passed**. eval/diagnose/trainer untouched this task.

**Commits:** `d0f385e` (manifest), `[TASK-06] new-run derivation tooling
(phase 1)`, `[TASK-06B] recipe-identity correction (part 1)`.

**Pending from HPC (combined TASK-06 P2 + 06B Part 2):** derivation job over
the 6 completed runs; provenance harvest of the 12 legacy resolved configs
(ViT-B nomix grep decides `recipe_actual`); canon τ compute+apply; two HPC
commits + push. The four TRUE-NOMIX chains submitted (commands printed).
Then 06B Part 3 locally (pooled stats keyed by recipe_actual, grad-φ, 4-way
gate agreement, e2r_first_look note).

---

## 2026-09-07 — TASK 06B PART 3 (pooled stats + mechanism first look)

All four true-nomix runs complete and derived; the human's amendments
folded in: nomix as its own recipe_actual cell (n=2, SEs flagged
unreliable), grad-φ mixup-vs-nomix comparison, gate structure extended to
all SIX ViT-S SAGA sets, train_loss overfitting line in the note.

**Done (local):**
- `analysis/build_pooled_tables.py` → `results/tables/e2_pooled.csv`
  (71 rows). Cells keyed (arch, recipe_actual); legacy repeats join via
  the erratum remap; VOID ViT-B mixup-dir trio appears nowhere. Repeats
  listed individually + mean/std + per-tag paired deltas + SE +
  significant_2xSE + Welch. MISSING never averaged (n=1 → std MISSING;
  Welch needs n≥2 per side).
- Headline (exact fp32 eval, top1_last): S/mixup SAGA Δ = **+0.456**
  (SE 0.2124, significant at 2×SE; deltas −0.126/+0.570/+0.490/+0.890 —
  human's arithmetic verified, s1's +0.52 was the bf16 log value);
  S/nomix SAGA Δ = +0.327 (n=2); B/mixup SAGA Δ = +0.438 (n=2, not
  significant); registers negative in both cells that have it.
- `analysis/collect_gradphi.py` + `plotting/plot_gradphi.py` →
  `figures_data/gradphi.csv` (9672 rows) + `figures/gradphi_draft.pdf`.
  Median ‖∂L/∂φ‖: mixup 6.112e-04, nomix 8.652e-04 — ratio 1.42.
- `analysis/gate_structure.py` → gate_agreement.csv (15 pairs),
  gate_spatial_std.csv, `figures/gate_agreement_draft.pdf` (layer-8
  six-map row). Pooled Spearman: within-mixup 0.921..0.958 (6 pairs),
  within-nomix 0.972, cross-recipe 0.848..0.935 (8 pairs). Constant
  final layer (φ≈init) excluded from per-layer stats, counted in
  n_layers_defined. nomix spatial std ~1.5× mixup's.
- `analysis/build_e2r_first_look.py` → `results/notes/e2r_first_look.md`
  (ViT-B resolution, consistency + bf16-vs-fp32 cause, pooled tables,
  verified deltas, canon-vs-MAD ViT-B reversal REPRODUCES on clean seeded
  runs, train_loss gap 2.6546 vs 1.4678 while nomix val sits 5.69 pts
  lower, grad-φ, gate agreement, open items).
- `tools/apply_fixed_thr.py`: ViT-B legacy remap pending → **mixup**
  (Part-2 harvest); manifest updated (`e5200fe`). Legacy ViT-B nomix-dir
  diags still need the canon backfill — one idempotent HPC apply, in the
  protocol block.
- Review workflow findings, all fixed with regression tests:
  **operator-precedence bug in gate pooling** (computed harmonic-style
  head pooling, not sigmoid-then-mean; all first-pass gate numbers were
  wrong — regenerated), hardcoded "~5.5 pts" regime gap (now computed
  from pooled means), hardcoded τ=127.31 (now read from canon file).
- `pytest -q`: **96 passed**.

**Commits:** `e5200fe` (manifest), `[TASK-06B] pooled stats + mechanism
first look (part 3)`.

**Pending from HPC:** only the legacy ViT-B canon backfill (6 diag JSONs,
non-blocking). Open decisions for the human: optional ViT-B true-nomix
pair; registers seeded reruns; PROJECT.md milestone rewrite.

**Addendum (2026-09-07, post-backfill):** HPC canon backfill returned
(`c81d1f0`, 6 legacy ViT-B nomix-dir diags, tau=127.3125). Pooled table +
note regenerated: 13 sink_fixed_canon cells MISSING→value, 3 aggregates
recomputed n=1→n=2 (baseline mean 8.4035, saga mean 4.5336, saga
paired-delta mean −3.8699, now not significant at 2×SE); open item
resolved and removed. 3-agent adversarial verify: every value matches its
source JSON, no *_best contamination, nothing outside the intended cells
changed, note-vs-CSV exact. Fact now in the CSV: the canon-vs-MAD
ordering reversal also holds on the LEGACY ViT-B pair (canon 4.886 vs
5.5211; MAD 15.1858 vs 10.3115). pytest -q: 96 passed. Nothing pending
from HPC.

---

## 2026-09-07 — TASK 08, PHASE A (fine-grained clean protocol)

Replaces the VOID legacy e6 protocol (B7: official TEST split used as val,
best.pth selected on it, eval every 5 epochs; legacy +2.19 Aircraft /
+1.29 CUB are never cited). Legacy trainer/datasets untouched (provenance).

**Done (local, by Claude Code):**
- `evaluation/e6_finegrained/data/ft_meta.py` — ONE parsing path for the
  official CUB/Aircraft splits + label conventions (CUB cls−1, Aircraft
  variants.txt file order — both legacy-identical), shared by builder and
  trainer so split-file labels can never drift; metadata readable straight
  from the tarballs (no image extraction); legacy transforms verbatim.
- `evaluation/e6_finegrained/tools/build_ft_split.py` — frozen stratified
  10% val carve from the OFFICIAL train split (per class max(1,
  round-half-up(0.1·n)), random.Random(seed)); full items_train+items_val
  stored (relpaths+labels); disjointness enforced as raises (incl. official
  train∩test); provenance (git sha, source tar sha256); WRITE-ONCE output.
- `evaluation/e6_finegrained/tools/train_ft.py` — clean single-GPU
  fine-tune: hyperparameters MIRROR legacy e6 (AdamW backbone 1e-5 / head
  1e-3, wd 0.05, cosine eta_min 1e-7, 100 ep, batch 64, ls 0.1, clip 1.0,
  fp16 AMP, legacy transforms, 224 — any other img_size refused;
  adversarial mirror review: zero unintended math deviations). Protocol:
  val every epoch, best.pth selected on val, TEST touched EXACTLY ONCE at
  the end (only construction site + marker refusal) → eval/test_final.json
  (top1/top5, n_images, backbone+finetuned sha256, seed, git sha,
  smoke flag). Strict backbone load via model_factory (zero
  missing/unexpected) + matrix-pinned sha256 assert. Hygiene: ft_seed
  seeding (backbone fixed at e2r s1), run_registry, append-safe 4-field
  log.csv (epoch, lr-trained-with, train_loss, val_top1), atomic
  everything (fsync marker), NO resume (runs 1–3 h; restart-fresh with
  evidence-preserving cleanup after env validation). Idempotency: marker
  checked with VALIDITY (torn marker quarantined + redone), run.lock with
  atomic stale takeover (unique-rename), ownership-checked release,
  post-lock marker recheck, staging sentinel (requeue-safe), split-carve
  pinning (seed/frac), duplicate/count guards.
- `configs/ft_matrix.yaml` — 16 runs (`ft_<ds>_<arch>_<variant>_bs1_f<s>`):
  CUB-S/Aircraft-S {baseline,saga}×f{0,1,2}, CUB-B/Aircraft-B ×f0.
  Backbones = e2r mixup s1 last.pth, sha256 test-pinned to the committed
  eval JSONs.
- `scripts/gen_ft_jobs.py` → `scripts/jobs/ft_finegrained_array.sbatch`
  (16-task single-GPU array, per-task /scratch staging, set -u, positional
  mapping append-only) + `ft_smoke.sbatch` (2-epoch CUB-S saga f0 into
  gitignored results/smoke). All SLURM elements verified against
  How to Run.md + committed e2r job files (review verdict: nothing
  invented). `.gitattributes`: *.sbatch forced LF.
- `tests/test_task08_ft.py` — 21 tests (fake data, CPU): split determinism
  ×2 builds/stratification/disjointness/tar==root, label conventions,
  strict-load raises, sha mismatch refusal, non-224 refusal, real-matrix
  contract (naming/seeds/sha-vs-eval-JSONs/no-overrides), e2e contract +
  exact log schema + requeue skip + test-touched-once + fresh-restart
  determinism, torn-marker recovery, lock ownership + stale takeover,
  builder write-once, jobs-in-sync-with-matrix (order-exact).
- Review workflow (4 agents: protocol, hyperparameter mirror, idempotency,
  HPC): 15+ confirmed findings ALL fixed (lock races, marker validity,
  train∩test gap, carve pinning, evidence-destroying cleanup order,
  lr-logged-after-step, sbatch guards, …). Known+accepted: run_registry's
  non-atomic meta write (pre-existing e2r-wide), GPU kernels not forced
  deterministic (meta-noted, e2r policy), ft outputs on hpc FS (e2r
  precedent), smoke touches the test split once (flagged smoke:true,
  quarantined in results/smoke).
- `pytest -q`: **143 passed** (incl. another session's in-tree TASK-07
  tests). No edits outside e6 + new tooling/tests; TASK-07 worktree files
  left untouched and uncommitted.

**Reconciliations (task file vs reality):**
- Split files are a Phase-A deliverable on paper but can only be BUILT on
  the HPC (tars live on woody) — building+committing them is HPC step 1,
  before the smoke (CPU, login node, seconds; reads only tar metadata).
- "MIRROR their hyperparameters": legacy ran BOTH 100 ep (tool default,
  original ViT-B) and 30 ep (reruns, "peak at 15-20"). Matrix pins 100
  (val-selection makes late overfitting harmless; matches the task's
  1–3 h/run estimate) — flagged for explicit human sign-off.
- Array job used without asking: sbatch --array is documented verbatim in
  How to Run.md §0/§6 — "if supported" is settled, not unclear.

**Commit:** `[TASK-08] fine-grained clean protocol (phase A)`

**Pending from HPC (Phase B):** build+commit the two ftsplit JSONs → smoke
→ submit 16 → sync cadence (block printed at end of task). Then Phase C
locally (T3 tables + finegrained.md) after 16× test_final.json are back.

**Update 2026-09-07 (Phase B, attempt 1 — 2/16 completed, node fault):**
- Splits built + committed on the HPC (`d43a182`): CUB 5994 → 5394 train +
  600 val (3/class, 200 classes); Aircraft 6667 → 5967 + 700 (7/class,
  100 classes). Both re-verified locally: duplicate-free, disjoint, sums
  exact, every class covered. `saga_ft_smoke` COMPLETED twice (jobs
  4196869, 4196916) — the protocol runs end-to-end on real hardware.
- Array 4196932 (`--array=0-15`): tasks 8,9 COMPLETED (34-36 min on
  a0804/a0805); the other 14 FAILED at elapsed 00:00:00, ExitCode 1:0.
  **Cause: the a0801 TaskProlog fault** (see the TASK-07 entry below for
  the stderr). Node-correlated exactly — `Node list` is a0801 for all 14
  failures and only for those. Nothing repo-side to fix; the trainer
  never ran (no `meta.json` written for any of the 14, which is itself
  the proof they died before `create_run`).
- First real fine-grained numbers (commit `2d2c525`, both verified
  against their own log.csv — 100 rows, cosine LR 1e-5 → 1.02e-7, logged
  val peak == `best_epoch`, backbone sha matches the matrix, n_test 3333
  counted once):
  `ft_aircraft_vits_baseline_bs1_f2` TEST **72.247** (top5 92.859), val
  peak 73.857 @ epoch 55; `ft_aircraft_vits_saga_bs1_f0` TEST **73.687**
  (top5 93.129), val peak 76.000 @ epoch 59. **NOT a paired comparison**
  (different ft-seeds) — no delta may be quoted until the matching seeds
  land. Protocol working as designed: val peaks mid-run (55/59) and
  decays to 73.0/74.4 by epoch 99, so val-selection caught the true peak;
  val→test gap 1.6/2.3 pts is the honest generalization drop B7 hid.
- Resubmitted as job **4198114** with `sbatch --exclude=a0801
  scripts/jobs/ft_finegrained_array.sbatch` (queued, PD/Priority). The
  requeue guard skips tasks 8,9 via their `test_final.json` markers, so
  only the 14 outstanding runs burn GPU — the idempotency design paying
  off in production for the first time.
- Quota watch: hpc 86.1G of 104.9G soft; the 14 remaining runs add
  ≈9-10 GB of ft checkpoints under `results/runs/<id>/ckpt/`.

**Update 2026-09-08 (Phase B attempt 2 + PHASE C — T3 complete):**
- Array 4198114 (`--exclude=a0801`): **all 14 outstanding runs COMPLETED**
  (19-40 min each); tasks 8,9 skipped themselves via their markers exactly
  as designed. All 16 `eval/test_final.json` in hand (human commit
  `fd98185`, rebased onto `dd73aea`).
- Integrity of all 16 verified before any table was built: `smoke` false
  everywhere, `n_images` correct per dataset (CUB 5794 / Aircraft 3333),
  100 epochs and 100 log rows each, `best_epoch` and `val_top1_at_best`
  matching each run's OWN log.csv peak exactly, the four backbone sha256s
  matching `configs/ft_matrix.yaml`, and ONE frozen split sha per dataset
  across all runs of that dataset.
- C1 `analysis/build_ft_tables.py` → `results/tables/T3_finegrained.csv`
  (60 rows). Repeats listed individually before any mean; deltas paired BY
  ft-seed (the only valid pairing — both sides share the frozen val split);
  sample std (n-1), SE = sd/√n, `significant_2xSE`, unpaired Welch as a
  robustness line; MISSING never averaged; a run whose backbone sha does
  not match the matrix (or that is a smoke artifact) yields MISSING, never
  an unverified number. ViT-B cells carry an explicit `n=1` flag.
- **T3 headline (exact test top-1, official test split touched once per
  run, val-selected checkpoint):**
  - CUB ViT-S: baseline mean **80.877** (std 0.2390; 81.153/80.739/80.739),
    saga mean **80.515** (std 0.3543; 80.860/80.532/80.152). Paired Δ =
    **−0.362**, SE 0.1150, **significant at 2×SE (SAGA BELOW baseline)** —
    all three seeds negative (−0.293/−0.207/−0.587). Welch p 0.2255.
  - CUB ViT-B: baseline 80.549, saga 81.878, Δ **+1.329** (n=1, no SE, no
    significance claim possible).
  - Aircraft ViT-S: baseline mean **73.067** (std 1.0113;
    72.757/74.197/72.247), saga mean **74.027** (std 0.3617;
    73.687/74.407/73.987). Paired Δ = **+0.960**, SE 0.4419, significant
    at 2×SE but by a THIN margin (|Δ| − 2×SE = +0.076); all three seeds
    positive (+0.930/+0.210/+1.740). Welch p 0.2367.
  - Aircraft ViT-B: baseline 72.427, saga 74.227, Δ **+1.800** (n=1).
  - **The two ViT-S cells reach 2×SE significance in OPPOSITE directions**
    (CUB negative, Aircraft positive). Both ViT-B deltas are positive but
    n=1. No pooled cross-dataset claim is made anywhere.
- C2/C3 `analysis/build_finegrained_note.py` →
  `results/notes/finegrained.md` (162 lines; every number computed from the
  CSV + run JSONs, none typed). §0 states the legacy **+2.19 Aircraft /
  +1.29 CUB are VOID and never cited**, with the B7 mechanism, and that
  they must not be compared against T3 (they measure a different, invalid
  quantity). §4 tabulates per-run backbone/split/git shas.
- Sanity (task C.2): **16 of 16 runs peaked on val strictly before their
  final epoch** (best epochs 11..86 of 100) — the 100-epoch schedule
  overfits both datasets and val-selection changed the evaluated weights
  on EVERY run, which is precisely what B7's test-tuned peak concealed.
  6 of 16 runs show |val − test| > 2 pts, all positive, largest +3.784
  (CUB ViT-B baseline); 13 of 16 gaps positive, mean +1.553. Recorded with
  the computed noise floor (CUB val n=600 ⇒ binomial SE ≈ 1.53 pts;
  Aircraft n=700 ⇒ ≈ 1.65 pts) and the note that a val-selected maximum is
  upward-biased — reported, not corrected for.
- Independent recompute of every T3 value straight from the 16 JSONs via a
  separate code path (repeats, means, sample stds, paired deltas, SEs,
  2×SE verdicts): **0 errors**.
- 3 note-generator defects found and fixed before commit: binomial-SE
  lines duplicated per cell instead of per dataset; "n = 1 pairs" plural;
  and a bare "significant: YES" that could be misread as favourable on the
  CUB cell — now every verdict prints its direction and its |Δ| − 2×SE
  margin, and a thin margin (< 25% of the threshold) says so in words.
- Tests: 9 new (T3 stats/pairing/MISSING-never-imputed, sha-mismatch and
  smoke → MISSING, committed-T3-matches-run-JSONs, note-states-void).
  `pytest -q`: **220 passed** (includes the concurrent TASK-07/09
  sessions' in-tree tests).
- **Reconciliation:** the TASK-09 session was editing this worktree
  concurrently (detection/segmentation trainers, `.gitignore`,
  `results/README.md`, `scripts/sync_results.sh`, `saga/vit.py`,
  `requirements.txt` and its own TASK_LOG entry, all uncommitted). Per
  CLAUDE.md's one-task-per-commit rule this commit stages ONLY the four
  TASK-08 artifacts plus `tests/test_task08_ft.py` and this log entry;
  TASK-09's uncommitted work was left exactly as found.

**TASK 08 COMPLETE** (Phases A/B/C). Nothing pending from the HPC.
Open decisions for the human: whether to add ft-seeds f1/f2 for the two
ViT-B cells (would let the +1.329/+1.800 deltas carry an SE), and how to
frame the opposite-direction ViT-S result in the paper.

---

## 2026-09-07 — TASK 07, PHASE A (sink-address tool + optional-run prep)

Ran in parallel with the TASK-08 Phase-A session in the same worktree
(its commit `1a11988` landed mid-session); each task committed only its
own explicit paths, and neither commit contains the other's files.

**Done (local, by Claude Code):**
- A1 `tools/sink_address.py` — CPU pass over every `<stem>_norms.npz`
  under `results/legacy/diag/` + `results/runs/*/diag/` (expected on the
  HPC: 24 legacy best+last, 20 e2r final best+last = 44). Writes a small
  committable `<stem>_addr.json`: `freq_canon[196]` (cell canon τ; key
  resolved via `apply_fixed_thr.resolve_key`, erratum remap included),
  `freq_mad[196]` (per-image median+5·MAD, lower-median convention),
  `mean_norm`/`p99_norm[196]`, concentration stats per freq map (entropy
  bits vs log2(196) + normalized, Gini, top-5/top-20 mass share, top-10
  indices), provenance (ckpt sha, n_images, τ, k, git sha). Idempotent
  (skip iff sha+τ+k+n_images match), atomic writes, HARD cross-check
  `mass_canon == sibling sink_fixed_canon` (same npz + τ ⇒ exact; ERROR
  otherwise), soft MAD cross-check recorded (fp16 npz vs fp32 in-model).
  Gini/entropy verified against independent definitions (pairwise-MAD
  formula, scipy.stats.entropy) and an end-to-end synthetic-population
  CLI run (remaps, rerun-skips, planted addresses recovered).
- A2 `configs/e2r_matrix.yaml` + 4 OPTIONAL runs (S/B mixup registers s1,
  B mixup baseline/saga s2); `gen_slurm_chain.py --only` (existing 10 job
  files stay byte-identical; new chains at ports 29750-53) + refuses an
  implicit full regen (`--all` required — matrix growth reshuffles ports);
  `scripts/saga_e2r_registers_smoke.sh` (2-epoch ViT-S registers smoke,
  port 29698). Registers has NEVER run through the new trainer; smoke
  precondition stated in the protocol. NOTHING submitted by Claude —
  all four are the human's call.
- Adversarial review (2 agents): 0 confirmed numeric defects; 3 noted
  hardenings applied + tested (crosscheck re-engages after a late canon
  backfill; n_images in the skip guard; N≥2 shape guard) + the implicit-
  regen refusal. Accepted gap, stated in the protocol: a 2-epoch smoke
  cannot reach run_diag (first diag fires at epoch 9; prefix-token
  handling for reg models is already test-covered).
- `pytest -q`: **143 passed** = 96 pre-existing + 26 TASK-07 + 21 TASK-08
  (the parallel session's suite, already committed by then).

**Commit:** `[TASK-07] sink-address tool + optional-run prep (phase A)`

**Pending from HPC (Phase B):** `python tools/sink_address.py` over both
diag roots (expect 44 written / 0 partial / 0 errors) → commit the
`*_addr.json` files + push. OPTIONAL: the four submissions, registers
chains only after the registers smoke + contract check. Then Phase C
locally (`analysis/address_analysis.py`, `plotting/plot_address.py`,
`results/notes/sink_address.md`).

**Addendum (2026-09-07, A1 back + smoke blocked by cluster fault):**
- A1 outputs returned (`7ad53a1`): all 44 `_addr.json` present (24 legacy
  best+last, 20 e2r final best+last). Verified locally against their
  sources: every `tau_canon` matches `fixed_thresholds_canon.json` for
  its resolved key, the HARD cross-check engaged on 44/44 with
  `abs_diff <= 1e-8` vs the sibling `sink_fixed_canon`, mass identities
  (`sum(freq) == mass`) hold for both maps, all freqs in [0,1], all
  10000x196, concentration stats within their references, and the soft
  fp16-vs-fp32 MAD drift spans only −0.0142..−0.0036. **Phase C is
  unblocked.**
- The registers smoke was submitted TWICE (jobs 4196865 18:40:47,
  4196890 18:46:57) and both died at elapsed 00:00:00 on the SAME node
  `a0801` with, in stderr: `slurm task_prolog can not be executed
  (/etc/slurm/slurm.taskprolog) Permission denied` → `TaskProlog failed
  status=1`. `/etc/slurm/slurm.taskprolog` is site-administered and runs
  BEFORE the job script; the job `.log` contains only the epilogue stats
  block, so not one line of our script executed. **Cluster-side node
  fault, NOT a defect in the smoke script, the matrix entry, or
  env_alex.sh** — nothing to fix in this repo. The same signature (FAILED,
  00:00:00, ExitCode 1:0) hit TASK-08 array 4196932 tasks 6,7,10-15,
  while tasks 8,9 COMPLETED in 34-36 min on a0804/a0805 — i.e. the fault
  is node-correlated, not task-correlated. Human commit `2d2c525` added a
  commented-out `#module load python` to `scripts/env_alex.sh`: a no-op
  line, and irrelevant here since the job script never ran.
- Registers smoke therefore STILL PENDING: resubmit (likely lands on a
  healthy node); the standard sbatch flag `--exclude=a0801` avoids the
  bad node if it recurs (flag is NOT in `How to Run.md`). Neither
  registers chain may be submitted until a smoke passes.
- Quota watch (from the job epilogue, 2026-09-07): vault 1021.0G of
  1048.6G soft (97.4%), 146K of 200K files; hpc 85.9G of 104.9G soft.
  e2r checkpoints land on **hpc** (`results/runs/<id>/ckpt/`), and per
  the manifest a ViT-B pair (last+best) is 2.08 GB, a ViT-S pair 0.52 GB
  → the four optional runs would add ≈6.8 GB to hpc's ≈19 GB headroom.

---

## 2026-09-08 — TASK 09, PHASE A (dense prediction fixes + launchers)

First task to touch the detection and segmentation pipelines. All prior
dense numbers are void (B5/B6 in detection, B4 in segmentation), so nothing
here had to preserve a previous result — only the committed TRAINING MATH.

**Done (local, by Claude Code):**
- **B5 double normalization** (`detection/models/detector.py`): FasterRCNN is
  now built with `image_mean=[0,0,0], image_std=[1,1,1]`, so the dataloader
  Normalize in `detection/data/transforms.py` is the single normalization
  (the ONE fix, never both). New `ViTDetector.check_normalization()` runs the
  detector's own GeneralizedRCNNTransform on a real batch and asserts the
  tensor reaching the backbone equals the dataloader's image elementwise,
  that the observed channel means sit at the single- not the
  double-normalized prediction, and that the transform's mean/std are the
  identity. It compares only the UNPADDED region of each image, and runs
  inside `torch.random.fork_rng` so the transform's min_size draw cannot
  shift the run's RNG stream. The trainer calls it at step 0 of EVERY job
  (fresh and resumed), on device, and records the statistics in `meta.json`
  and in `coco_eval_best.json`.
- **B6 registers backbone** (`detection/models/backbone.py`, rewritten): for
  `registers > 0` the timm `reg_tokens=4` model is used DIRECTLY and its own
  `forward_intermediates()` supplies the intermediates (it strips all 5
  prefix tokens and resamples the pos-embed for the actual grid). It is no
  longer wrapped in `SAGAViT`, which copied only `cls_token`/`pos_embed` and
  therefore DROPPED `reg_token` (surviving only because the load was
  `strict=False`). Loading is now strict=True with a sha256 assert.
  `SAGAViT.__init__` REFUSES any model with `num_prefix_tokens != 1` or a
  `reg_token`, and `_interpolate_pos_embed` gained tripwires (prefix count,
  square source grid, grid-vs-token agreement).
  `segmentation/models/backbone.py` was a byte-identical COPY carrying the
  same bug; it is now a re-export, so there is ONE implementation.
  Verified compatibility: adding `dynamic_img_size=True` changes no parameter
  name or shape and is value-identical at 224, so the classification
  checkpoints strict-load into the dense backbones.
- **B4 mIoU ignore** (`segmentation/tools/train.py`): the metric is now a
  confusion matrix accumulated only over `gt != 255` pixels; intersection and
  union are both derived from it, so ignored pixels are structurally absent
  from both. The old formula counted ignore pixels into the union of every
  class the model happened to predict there, biasing mIoU downward by a
  variant-dependent amount. The matrix is written out (`conf_matrix.npz`), so
  the per-class table, the JSON and Phase C all read the SAME evaluation.
- **A fourth bug, found while wiring the eval and also fixed**: the COCO
  scoring compared predictions in the contiguous 1-80 label space against
  ground truth that had kept its original 1-90 category ids (and dropped
  every annotation whose original id was > 80). Predictions are now mapped
  back through `idx_to_coco_id` and scored against the OFFICIAL
  `instances_val2017.json`; the legacy `get_coco_api()` helper was fixed the
  other way (annotations remapped into the model label space) so
  `detection/tools/analyze.py` stops being silently wrong.
  Pinned by a test in which correctly mapped predictions score AP 100.0 and
  unmapped ones do not.
- **A FIFTH bug — the worst of them — found and fixed in the same session
  (independently confirmed by the adversarial review with an end-to-end
  measurement): every predicted box was scored in the DATALOADER's resized
  coordinate frame against ground truth in original image coordinates.**
  This pipeline resizes in the dataloader (`ResizeDetection`, which also
  scales the train boxes), so by the time torchvision's `postprocess` runs,
  its `original_image_sizes` are already the resized ones and its rescaling
  is a no-op — nothing ever mapped predictions back. A 640x480 COCO image is
  resized by 1.667, so a perfect detector's boxes miss their ground truth at
  every IoU threshold: measured AP **0.0** for an oracle prediction set,
  **100.0** for the same set divided by the dataloader scale. All six APs
  (and `detections_val.json`, and Phase C's small-object crops) would have
  been meaningless after ~3 GPU-days per backbone. Fix: the dataset now
  carries the true `orig_size` in every target and
  `boxes_to_original_frame()` inverts the resize per axis (torchvision's
  `resize_boxes` convention) before anything is written or scored. Pinned by
  a test where a stub perfect detector scores AP 100 through the REAL
  pipeline (dataset -> transforms -> infer_val_shard -> score_coco), verified
  to FAIL when the mapping is removed. This bug was live in the committed
  code too (its GT came from `get_coco_api()`, also original-frame), so no
  pre-fix detection number was ever meaningful.
- **Registers/SAGA feature-list asymmetry** (found by a production-resolution
  test): timm's `forward_intermediates` appends ONE entry per matching block
  in block order, so a repeated or descending `fpn_indices` came back with the
  wrong length while `SAGAViT`'s honoured the request literally — the neck
  then got too few feature maps and the anchor generator raised. The
  registers path now asks for the unique indices and re-expands to the
  requested order/multiplicity, and both paths assert one feature map per
  index. (Production uses `[3, 6, 9, 11]`, so this never reached a run.)
- Both trainers rewritten with the TASK-05/TASK-08 hygiene: `run_registry`
  provenance, append-safe `log.csv` with resume-time row sanitisation, atomic
  per-epoch `ckpt/last.pth` carrying per-rank RNG + schedule geometry,
  `--resume auto` (one command fresh-starts and resumes) with a
  steps-per-epoch drift guard, exact `rank::world_size` val sharding (no
  padded sampler; coverage asserted), completion marker LAST (`meta.json`'s
  `end_time`) and a fast-path exit for a surplus job in the dependency chain.
  Every rank-divergent decision (already-complete, resuming, whether the
  multi-scale eval still has to run) is decided by rank 0 and BROADCAST —
  ranks disagreeing on a lazily-cached network FS would otherwise hang the
  job on the next collective until the wall clock.
  New shared module `tools/dense_runtime.py` holds all of it.
- Outputs (`results/detection/<run_id>/`, `results/segmentation/<run_id>/`):
  `coco_eval_best.json` (six APs + six ARs + per-category AP/AP50/AP_S) and
  `detections_val.json` (raw COCO-format predictions, written FIRST, with its
  size and sha256 recorded in the JSON that follows); `miou_ss.json`,
  `miou_ms.json` (multi-scale+flip TTA, run ONCE at the end from the
  val-selected weights), `per_class_iou.csv`, `conf_matrix.npz`,
  `preds_fixed20/<stem>_{pred,gt}.png` + `<stem>_img.jpg`. COCOeval's -1
  "no ground truth in this slice" sentinel is written as `null`, never as
  -100.0. Segmentation evals first sync BatchNorm buffers from rank 0, so a
  reported mIoU cannot depend on which val images landed on which rank.
- `results/probe/ade20k_fixed20.json` COMMITTED NOW (before any ADE20K
  number exists), built by `segmentation/tools/build_fixed20.py` from the
  canonical `ADE_val_%08d` naming with `sorted(random.Random(0).sample(
  range(1, 2001), 20))`; write-once. The trainer re-resolves every stem
  against the staged split and hard-errors on a mismatch.
- `configs/dense_matrix.yaml` (6 runs: `det_vitb_{baseline,saga,registers}_s1`,
  `seg_vitb_{...}_s1`) resolves each run ONTO the committed
  `detection/configs/base.yaml` / `segmentation/configs/base.yaml` — no
  hyperparameter is restated, and tests diff the resolved config against
  those files. Backbones: the seeded e2r ViT-B mixup s1 baseline/saga
  (sha256 taken from their own committed eval JSONs, test-pinned); registers
  = the e2r registers run IF it has FINISHED at launch (completion marker +
  a pinned hash, see the review round below), ELSE the legacy ViT-B
  registers `nomix`-dir `last.pth` (`recipe_actual=mixup`, sha256 from the
  manifest) — `resolve_backbone` records which in `meta.json`
  (`backbone_source: primary|fallback`).
- `scripts/gen_dense_jobs.py` → 6 sbatch + 6 submit chains (singleton +
  afterany; detection 4x24h, segmentation 2x24h) + `dense_smoke_{det,seg}.sbatch`.
  Every SLURM element copied from `How to Run.md` and the committed
  e2r/e3/e4 scripts (env + staging come from
  `detection/scripts/{env_alex,stage_coco}.sh` and
  `segmentation/scripts/{env_alex,stage_ade20k}.sh` — the COCO/ADE zip paths
  are theirs, nothing invented). Ports 29850-29855 + 29860/29861, asserted
  collision-free against every committed launcher (including the
  array-computed ones). Each job checks `tools/dense_done.py` BEFORE staging,
  so a surplus chain job does not unzip 18 GB of COCO to be told there is
  nothing to do. The smokes verify pycocotools/timm/torch, then run
  `pytest -q tests/test_task09_dense.py` ON THE COMPUTE NODE, then
  2x25 (det) / 3x17 (seg) train iterations + one capped eval — epoch counts
  chosen so the LAST smoke epoch trains with the backbone UNFROZEN. Smoke
  output goes to `results/smoke/` (git-ignored) and every JSON it writes
  carries `"smoke": true`.
- `tests/test_task09_dense.py` — 83 tests (CPU, fake data, tiny real models
  at REAL dense resolutions): B5 (identity transform, single-normalized
  backbone input, padded mixed-size batch, and the bug's reintroduction
  detected), B6 (3800 = 50x76 tokens at 800x1216 for all three variants,
  gate+pos-embed interpolation, SAGAViT refusal, strict load with reg_token
  verified, sha mismatch refused, fpn_indices fidelity), B4 (hand-computed
  IoU on a 6x6 image with an ignore border, and the old formula proved
  different), COCO scoring (bijective id map, AP 100 mapped vs < 100
  unmapped, per-category extraction, -1 sentinel, include_empty), the matrix
  contracts (resolved == committed config, backbone shas == the eval JSONs,
  fallback == the manifest row, job files byte-identical to the generator,
  port uniqueness), the runtime helpers, the probe list, and two end-to-end
  run contracts asserting the full artifact set, the exact log schema and the
  provenance fields (plus the resubmission no-op).
- `requirements.txt` gained `pycocotools>=2.0.7` (never declared, though the
  legacy e3 runs used it). `.gitignore` ignores `results/**/eval_shards/`.
  `scripts/sync_results.sh` extended for the dense artifacts; above
  `DENSE_DETECTIONS_MB` (25) it gzips `detections_val.json` and commits the
  `.gz` instead, so Phase C gets the data without a ~44 MB blob per run in
  git. `results/README.md`
  documents the two new trees. The legacy `e3_train_alex.sh` /
  `e4_train_alex.sh` launchers carry a SUPERSEDED — DO NOT SUBMIT banner
  (their trainers' CLI no longer exists).

**Adversarial review round (8 review agents + 2-lens verification, then a
6-lane attack on the fixes).** One CONFIRMED critical — the frame bug above,
which had already been found and fixed in-session; the review reproduced it
independently end to end (oracle predictions: AP 0.0 unmapped, AP 100.0
mapped). One confirmed-mechanism split finding. 24 verifier agents died on a
session limit, so the remaining claims were checked by hand; the real ones
are fixed:
- **Port collision (would have broken a running job):** the dense chains used
  29800-29805, but `classification/scripts/e2_nomix_alex.sh` computes
  `29800 + SLURM_ARRAY_TASK_ID`. The first port test only globbed `scripts/`
  and could not see an arithmetic port. Dense ports moved to 29850-29855
  (+ 29860/29861 smokes); the test now derives every port in the repo,
  expanding each `$((BASE + SLURM_ARRAY_TASK_ID))` over that file's own
  `--array` range, and pins that 29800 still belongs to e2_nomix so the
  reason for the move stays on record.
- **`resolve_backbone` tested "file exists", not "run finished":** the e2r
  trainer rewrites `ckpt/last.pth` every epoch, so an in-flight registers run
  would have silently become the dense backbone — with `sha256: null` making
  the hash guard inert. A candidate now needs its file, a completion marker
  on that run (`require_complete` + `require_epochs`, the latter test-pinned
  to `classification/configs/base.yaml`'s 300) AND a pinned hash; every
  rejection prints its reason. Switching to the e2r registers backbone is now
  one deliberate edit (paste its hash from its own eval JSON).
- **`detections_val.json` carried fabricated segmentation polygons:**
  pycocotools' `loadRes` mutates the dicts it is handed (adding a box-outline
  `segmentation`, `area`, `id`, `iscrowd`) and the artifact was serialized
  from those same dicts — a mask-shaped value the model never predicted, in a
  results file, at ~2.2x the size. `score_coco` now hands loadRes a per-entry
  copy; a test asserts the artifact's keyset is exactly
  {image_id, category_id, bbox, score}.
- **`per_class_iou.csv` used `",".join`** while ADE20K class names contain
  commas ("person, individual, someone, somebody, mortal, soul") — that would
  have spliced extra columns and shifted every number in the row one field
  right. Now `csv.writer`; and `load_class_names` sniffs the delimiter (the
  official `objectInfo150.csv` is TAB-separated precisely because of those
  commas) and reports why names are unavailable instead of silently writing
  150 MISSING rows and blocking Phase C's sky/wall/floor lines.
- **A stale `miou_ms.json` survived a fresh start** (the multi-scale eval is
  gated on that file's existence), so a run could finalize carrying an mIoU
  produced by weights that no longer existed. Fresh starts now clear the
  previous attempt's artifacts — and, per TASK-08's ordering, ONLY AFTER the
  environment validates. Verified end to end: a run that aborts on a backbone
  hash mismatch leaves the previous attempt's files untouched, and the next
  valid run clears them.
- **A torn `meta.json` killed the whole chain:** `finalize_run` writes it
  non-atomically (pre-existing, e2r-wide, recorded as accepted in TASK-08)
  and the resume path did an unguarded `json.load`. Under
  `--dependency=afterany` that one exception would have taken every remaining
  job of the run with it. `read_meta()` now quarantines the torn bytes and
  rebuilds the record.
- **Rank-divergence hangs:** the already-complete / resuming / run-the-MS-eval
  decisions are read off a shared filesystem. Rank 0 now decides and
  broadcasts all three; ranks that disagreed would have blocked on the next
  collective until the wall clock.
- **Surplus chain jobs unzipped 18 GB of COCO** before reaching the
  in-process fast path. New `tools/dense_done.py` (exit 0/1/2) runs in the
  sbatch before staging. The smokes also check `pycocotools`/`timm`/`torch`
  explicitly, since a missing pycocotools would only SKIP tests and report
  green; the detection trainer now import-checks it at startup too.
- **Segmentation mIoU was shard-dependent:** DDP syncs BatchNorm buffers per
  forward, but each rank then updates its own, so at epoch end the ranks
  differ by one momentum step and the reported mIoU depended on which images
  landed where (and differed from the saved weights' own metric).
  `sync_buffers_from_rank0` runs before every eval; training math untouched.
- **Three legacy dense scripts now REFUSE TO RUN**
  (`segmentation/tools/evaluate.py`, `detection/tools/evaluate.py`,
  `detection/tools/analyze.py`). The first takes `ignore_index=255` and never
  applies it (B4) and averages per-image IoU; the other two score
  resized-frame boxes against original-frame GT — and `analyze.py` computes
  exactly the per-category AP and AP_S/AP_M/AP_L numbers Gate 2 turns on. A
  runnable path to a void number is a defect, not a convenience; the bodies
  are kept as `_disabled_main` for provenance. Phase C reads
  `coco_eval_best.json` + `detections_val.json` instead, which is why the
  task mandates them.
- Two review claims were checked and REJECTED: the frame bug is not caused by
  the B5 fix (the resize half of `GeneralizedRCNNTransform` was already a
  no-op), and the resize scale IS recoverable post hoc from the annotation
  sizes, so nothing would have been unrecoverable.
- Also hardened while there: `repo_path` keeps POSIX-absolute HPC paths
  absolute on Windows (it would otherwise rebase `/home/vault/...` under the
  repo and could load a same-named local file); COCOeval's -1 "no ground
  truth in this slice" sentinel is written as `null`, never as -100.0; the
  registers and SAGA backbones now answer any `fpn_indices` list identically
  (timm collapses repeated indices, SAGAViT does not — caught by a
  production-resolution test); and the mandated log/JSON schemas are pinned
  LITERALLY in tests rather than against the constants that wrote them.

**Second review round — a 6-lane adversarial attack on those fixes**, each
lane told to break one and to MEASURE rather than reason. It found nine more
real defects, all fixed:
- **The box mapping was not the exact inverse of the resize.** It divided by
  the per-axis size ratio `orig/round(orig*scale)`, while the forward
  transform multiplied by the scalar `scale` — the rounding residual is a
  systematic sub-pixel stretch (measured up to 0.39 px / 4.9e-4 relative)
  that costs a perfect detector several points of **AP_S**, the number Gate 2
  turns on; my roundtrip test's 1.0 px tolerance was 2.6x too loose to see
  it. `ResizeDetection` now RECORDS the scalar it applied
  (`target['resize_scale']`) and the evaluator divides by that; the test
  asserts 1e-9 over eleven shapes including rounding-worst ones, and also
  asserts the size-ratio inversion is measurably worse (so the two can never
  silently converge again).
- **A resumed run re-attributed its results to a backbone that contributed
  nothing.** `resolve_backbone` runs fresh in every chain job, but the
  checkpoint's weights overwrite whatever was just loaded — so pinning the
  registers hash mid-flight (which the matrix explicitly invites) would have
  produced results JSONs naming a backbone that never trained. Both trainers
  now compare the resolved hash against the checkpoint's own
  `backbone_sha256` and refuse, naming both digests.
- **A globbed submit instruction submits only the FIRST chain** (verified:
  bash runs the first matched file and passes the rest as positional
  arguments, which the submit scripts ignore) — two of the three Gate-2
  detection rows would silently never have been queued. Every instruction is now one command per line, and
  a test scans the repo for that shape.
- **The best-artifact set was not a transaction and no resume reconciled
  it.** A kill inside the write block left a half-updated pair; since the
  best metric comes back from the checkpoint, a re-run of that epoch is not a
  new best and would never have rewritten it — the run would finalize with
  two results files describing different epochs. New
  `check_best_artifacts()` reconciles on resume (discarding a stale set so
  the next eval rewrites it) and re-checks before `finalize_run`, refusing to
  mark a run complete otherwise. Verified end to end by deleting
  `coco_eval_best.json` mid-run.
- **`load_class_names` trusted row order** and never checked the `Idx` column
  it had already parsed: a re-sorted `objectInfo150.csv` would have attached
  every ADE20K name in the committed table to the wrong IoU. It now requires
  Idx == 1..150 in order and reports MISSING instead.
- **`per_class_iou.csv` was written in the node's locale encoding**
  (`atomic_write_text` had no `encoding=`), so a non-ASCII class name would
  land in a committed results file that Phase C's UTF-8 read would reject.
  Now UTF-8 explicitly.
- **mIoU units were ambiguous**: the JSONs report mIoU/pixel_acc/mean_acc as
  PERCENTAGES while `iou_per_class` and the CSV column are FRACTIONS, with
  nothing recording that. A 100x error in T5_ade20k.csv would have looked
  plausible. The CSV column is now `iou_frac`, both JSONs carry a `units`
  block, and results/README.md says so.
- **`atomic_torch_save`/`atomic_npz_save` skipped the fsync** the module
  docstring promised — on `ckpt/last.pth`, the one file the whole chain's
  recovery depends on. Both now fsync before the rename. And an unreadable
  `last.pth` no longer kills the chain: it is quarantined so the NEXT job
  fresh-starts.
- **`--dependency=singleton` lived only in `submit_<run>.sh`**, so `sbatch
  scripts/jobs/<run>.sbatch` — the documented way to add one more job to a
  chain — carried no serialization and could have run concurrently with a
  job of the same run, overwriting its checkpoint (exactly how the legacy e2
  repeats were lost). It is now in the job header too.
- Also from this round: the sbatch files run `tools/dense_done.py
  --require_backbone` BEFORE staging (so a surplus job does not unzip 18 GB,
  and a moved registers checkpoint aborts in seconds rather than minutes into
  a 4x24h chain — neither smoke covers the registers backbone); the
  generated jobs enforce the TRAIN image count too (the committed stage
  functions enforce only the val count, so a truncated extraction would have
  trained 25/80 epochs on partial data); and `sync_results.sh` compares
  bytes rather than truncated megabytes, keeps the raw and gzip dumps
  mutually exclusive in the index, and only ships the dump once its run is
  complete (it is rewritten at every new best, so syncing mid-run pushed the
  same blob into history repeatedly).
- Lane verdicts worth recording, all backed by their own measurements: the
  mIoU definition matched an independent confusion-matrix-free reference on
  40 randomized trials (worst delta 4.8e-5); the multi-scale TTA with
  scales=[1.0] reproduces the single-scale matrix bit for bit and the flip is
  correctly un-flipped; int64 accumulation uses 5.7e-11 of its range at
  ADE20K scale; per-shard sums are bit-identical to a single pass; the
  pre-eval buffer sync provably cannot move a training number (DDP
  re-imposes rank 0's buffers on the next forward anyway); and per-rank
  collective sequences were byte-identical across ten restart states in
  2-rank gloo runs (fresh, fast-path exit, resume-with-MS-owed,
  resume-with-nothing-owed, capped-eval smoke).

**Reconciliations (task file / repo vs reality) — stated, not improvised:**
1. "legacy 1x schedule per the repo's configs": COCO convention calls 12
   epochs 1x, but the committed `detection/configs/base.yaml` says **25**
   epochs. The instruction was "keep hyperparameters as committed", so 25 is
   what the matrix resolves — flagged for explicit human sign-off, since it
   is ~2x the GPU time of a literal 1x.
2. `warmup_epochs` exists in both committed base configs but neither
   committed trainer ever implemented warmup. It stays unimplemented;
   implementing it now would change the LR schedule, i.e. the numbers.
3. COCO val protocol: the committed dataset drops images with no annotation
   (4952 of 5000). The new val loader passes `include_empty=True` so AP is
   computed over the STANDARD 5000-image split, and the ground truth is the
   official annotation file. This changes no training math; it is recorded in
   every `coco_eval_best.json` under `val_protocol`. Flagged for sign-off.
4. `segmentation/configs/paths.yaml{,.template}` are a stale COPY of the e6
   fine-grained paths file (CUB/Aircraft tarballs, e6 output dirs — no
   ADE20K entries at all). The new trainers do not use paths.yaml (backbones
   come from the matrix, data from `--data_root`), so the stale file was left
   untouched for provenance and called out in the e4 SUPERSEDED banner.
5. `evaluation/e5_lost/tools/extract_features.py` wraps the timm registers
   model in `SAGAViT` too — the same B6 bug. The new guard turns that silent
   corruption into a loud `ValueError`. Out of scope here and not in flight;
   when e5 is revisited, its registers path must move to
   `forward_intermediates` the same way.
5b. The three legacy dense scripts were disabled (see the review round). The
   task named only the two trainers, so this is a deliberate overreach on
   "numbers are sacred" grounds — one line each to revert if the human
   disagrees.
6. Detection checkpoints are 103.7M params → `last.pth` ≈ 1.16 GiB per run;
   segmentation 91.8M → `last.pth` ≈ 1.03 GiB + weights-only
   `best_model.pth` ≈ 0.34 GiB. Six runs ≈ 7.6 GiB, against the hpc soft
   quota's ≈19 GiB headroom which TASK-08's remaining ft runs also draw on.
   Both trainers therefore accept `--ckpt_root` to redirect `ckpt/` off the
   repo filesystem; the default stays the repo run dir (e2r precedent). The
   human decides.
- `pytest -q`: **245 passed** — 83 TASK-09 tests + 143 pre-existing + 19
  from a parallel TASK-07 Phase-C session working in the same worktree
  (`tests/test_task07_address_analysis.py`, `analysis/address_analysis.py`,
  `plotting/plot_address.py`, `results/{tables,notes,figures_data}/…addr…`).
  Those files are LEFT UNTOUCHED AND UNCOMMITTED here; this commit contains
  only TASK-09 paths. One combined run showed their
  `test_build_end_to_end` failing while their session was concurrently
  regenerating its own results files; it passes alone, after my file, and in
  the next full run (235/235) — a race between the two sessions, unrelated
  to anything TASK-09 changed (that test references none of it).
  No results file was written or edited by this task.

**Commit:** `[TASK-09] dense fixes + launchers (phase A)`

**Phase B status (2026-09-08) — data confirmed, smokes PASSED, chains not
yet submitted.**

**Third review round (6-lane attack on the fixes + a completeness critic).**
The lanes confirmed the ten fixes hold — each backed by its own measurement
(the mIoU matched an independent reference on 40 randomized trials, worst
delta 4.8e-5; MS-TTA with scales=[1.0] reproduces the single-scale matrix bit
for bit; per-rank collective sequences byte-identical across ten restart
states in 2-rank gloo runs; the sha256 gate is not circumventable by '', 0,
False, [] or a truthy non-string). Five more real items, all fixed:
- **`sync_results.sh` called bare `python` with stderr suppressed.**
  `dense_done.py` imports torch, so on a login node without the conda env it
  exits non-zero — and the gate read that as "not complete" and printed a
  positive claim about run state that had never been established, silently
  never shipping `detections_val.json`. It now uses the env interpreter by
  absolute path (as every sbatch does), does not suppress stderr, and
  distinguishes exit 0 / 1 / >=2 ("CANNOT DETERMINE", with the activation
  command).
- **My "(not the smokes)" quota claim was measurably FALSE** — the smoke
  sbatch files pass only `--out_root`, so their ckpt/ also lands under
  CODE_ROOT, and they wrote 2.51 GiB onto the over-quota filesystem. Claim
  corrected in place. The matrix gained a `ckpt_root:` key (null = old
  default) and the generator now injects `--ckpt_root` into all six chains
  AND both smokes; a test asserts both states.
- **No timing projection existed** — the one pre-launch number that decides
  whether 25 epochs fit in 4x24 h, with no dense throughput datum anywhere
  in the repo to calibrate against. `check_smoke.py` now reads the last
  smoke epoch's `img_per_sec`/`wall_time` (that epoch is the unfrozen one by
  construction), projects epoch-hours onto the production schedule including
  eval passes and staging, FAILS if it overruns, and says what to raise
  `chain.<task>` to.
- **Both Phase-B updates had been filed inside the TASK-08 entry** (a
  `replace(..., 1)` hit TASK-08's `Pending from HPC` first), leaving TASK-09's
  own Pending block stale. Moved, and TASK-09's Pending rewritten as the
  four remaining steps.
- **The segmentation half of the submit instruction was still a wildcard
  shape** — in the very paragraph explaining why a wildcard drops rows,
  because the guard test only matched lines with a `bash ` prefix and that
  phrasing had none. All six commands are now written out and the pattern catches the
  bare form too (scoped to the dense names so real code is not flagged).


**Update 2026-09-08 (Phase B start — data confirmed, smokes submitted):**
- All four dataset archives EXIST on woody: COCO train2017.zip (19,336,861,798
  B), val2017.zip (815,585,330 B), annotations_trainval2017.zip (252,907,541 B)
  and ADE20K/ADEChallengeData2016.zip (967,382,037 B). ADE20K had never been
  used by a completed run, so this was the open question — it is answered.
- Backbone resolution verified on the HPC by `tools/dense_done.py
  --require_backbone` for all six runs: baseline/saga resolve to the e2r
  **primary** checkpoints (991 MiB each), and both registers runs correctly
  fall through to the **fallback** with the reason printed
  (`results/runs/e2r_vitb_mixup_registers_s1/ckpt/last.pth — file does not
  exist`). The legacy registers file is 1,039,043,001 B, matching its
  manifest row exactly.
- **QUOTA CONSTRAINT (it applies to the SMOKES TOO — an earlier version of
  this entry said "not the smokes", which was wrong: the smoke sbatch files
  pass only `--out_root`, so their ckpt/ also lands under CODE_ROOT, and
  they did in fact write 2.51 GiB there):** hpc is ALREADY OVER its soft
  quota — `101G* / 100G quota / 200G limit`, in grace,
  136k/500k files. vault is at 974G/1000G (147k/200k files). The third
  filesystem in the quota output (226G of 954G, ~728G free) is the bulk one
  How to Run.md §1 maps to woody. The six runs add ~7.6 GiB of checkpoints
  (det 1.16 GiB/run; seg 1.03 + 0.34 GiB/run, computed from parameter
  counts), and TASK-08's remaining ft runs draw on the same headroom, so the
  default (repo run dir, on hpc) should be redirected with `--ckpt_root`
  before submitting. The generated job files do not pass that flag yet —
  regenerating them needs the human's confirmed path.
- New `tools/check_smoke.py`: reads both smoke logs AND the artifacts they
  wrote and prints one PASS/FAIL per TASK-09 acceptance item (on-device deps
  + pytest, the B5 assertion, staging counts enforced, epoch count, log
  schema, six APs with no -1 sentinel, 80 per-category rows, the
  detections-dump sha and keyset, mIoU units and the B4 definition, 150
  named classes incl. sky/wall/floor, 20x3 probe files, backbone hash
  verified, world_size 4, `smoke: true` everywhere). Missing evidence is
  FAIL, never an assumption.
- Smokes submitted: `dense_smoke_det` job **4200497**, `dense_smoke_seg` job
  **4200498** (both PD/Priority at submission).

**Update 2026-09-08 (smokes BACK — both COMPLETED, verdict PASS with one
Phase-C gap):** jobs 4200497 (det, 4:39) and 4200498 (seg, 1:46), both
COMPLETED ExitCode 0:0 on a0905. `tools/check_smoke.py`: detection 44/45,
segmentation 52/55.
- **The acceptance items are demonstrated on real data.** B5 asserted ON
  DEVICE on a real COCO batch: `max_abs_diff=0.00e+00`, backbone-input
  channel means `[-0.409, -0.328, -0.170]`, `dist_to_double_normalized=3.49`
  — single-normalized, and far from the double-normalized alternative. The
  unit suite ran on the compute node (77 passed, zero skips, so pycocotools
  was really present). Both staging guards fired with the exact counts
  (118287 / 20210). Backbone hash verified against the matrix
  (`a4e0e0ccd3b4...` = e2r saga s1, source primary). world_size 4.
  Detection: 2 epochs, only the eval epoch carries an AP, `lr_backbone`
  1e-05 -> 5.05e-06 so the last epoch trained UNFROZEN, all six APs + six
  ARs present with no -1 sentinel, 80 per-category rows, n_val_images 64,
  detections dump sha-matched with keyset exactly
  {image_id, category_id, bbox, score} and OFFICIAL category ids.
  Segmentation: 3 epochs, mIoU only on the eval epoch, ss 3.8862 / ms 5.1011
  (MS > SS is the expected direction), 150 per-class entries, units block
  present, ignore_index 255, 20x3 probe files, committed probe-list sha
  recorded. Numbers are meaningless by design (50/51 iterations from a
  fresh head) and every JSON carries `smoke: true`.
- **The "TaskProlog" FAIL was MY CHECKER'S bug, not the run's.** It grepped
  the bare word, which SLURM prints in ordinary prologue output; the a0801
  fault signature is `task_prolog can not be executed` / `TaskProlog failed
  status=1` and produces NO script output (FAILED, 00:00:00, ExitCode 1:0).
  Both jobs COMPLETED 0:0 with full logs and artifacts. Pattern fixed and
  pinned by a test in both directions (benign word passes, real signature
  fails).
- **REAL GAP, Phase C only: the committed ADEChallengeData2016.zip extracts
  WITHOUT objectInfo150.csv**, so per_class_iou.csv's `name` column is 150x
  MISSING and Phase C's mandated sky/wall/floor rows cannot be surfaced by
  name. NO NUMBER IS AFFECTED — the class indices, the confusion matrix and
  every IoU are correct and complete; names are a join on class_index.
  `load_class_names` now searches four locations, reports every path it
  tried, and honours a new `class_names_file` matrix key so pointing at a
  real file is a config change. The name source must still be located on the
  HPC (see below); this does NOT block submitting the six chains.
- Smoke outputs: 1.15 GiB (det) + 1.36 GiB (seg) under `results/smoke/`,
  deletable — and worth deleting, since hpc is over quota.

**Update 2026-09-08 (chain lengths CONFIRMED on measured throughput; the
class-name file located):**
- `tools/check_smoke.py`'s projection, from the smoke's own last (unfrozen)
  epoch — the first dense throughput datum this repo has ever held:
  * detection **31.01 img/s** -> 1.06 h/epoch x 25 = 26.5 h + 5 evals 0.2 h
    + staging 1.0 h = **27.7 h vs the committed 4x24 h = 96 h budget**;
  * segmentation **120.67 img/s** -> 0.05 h/epoch x 80 = 3.7 h + 8 evals
    + staging = **4.3 h vs the committed 2x24 h = 48 h budget**.
  CAVEAT, stated because the margin is what makes this safe rather than the
  precision: the windows are tiny (6 s / 2 s, i.e. 202 / 278 images), they
  exclude dataloader steady state, and `wall_time` is training only (the
  projection charges evals separately at the training rate, which
  UNDERSTATES them since eval runs batch_size=1 forward-only). What makes
  the conclusion solid is that the committed chain lengths also cover the
  TASK-09 file's own independent estimates (~3 chained days for detection,
  ~15 h for segmentation): 72 h and 15 h both fit 96 h and 48 h. So the
  chain lengths hold whichever estimate is closer, and no change is needed.
- **The ADE20K class names DO ship with the archive — as
  `ADEChallengeData2016/objectInfo150.txt`, not `.csv`** (5,689 B; the zip
  also carries `sceneCategories.txt` and nothing else non-image).
  `load_class_names`'s candidate list already includes
  `$DATA_ROOT/objectInfo150.txt`, so no config change is needed — the smoke
  reported MISSING only because it ran at efd4f9d, before that search was
  added in 1dac02b. The parse (tab delimiter, `Idx`/`Name` headers, Idx
  order 1..150) is verified separately before the seg chains go out.
- Smoke checkpoints deleted (2.51 GiB reclaimed from the over-quota hpc
  filesystem).

**Update 2026-09-08 (data-staging policy + ckpt_root settled; chains ready):**
- The human asked whether the "unzip to /tmp, train, then close it" policy was
  honoured. It was, via the COMMITTED scripts: each job sources
  `<task>/scripts/env_alex.sh` (which sets
  `STAGE_DIR=/scratch/iwi5359h/{coco,ade20k}_${SLURM_JOB_ID}` — node-local,
  keyed by job id so chain/array jobs cannot collide) and
  `stage_{coco,ade20k}.sh`, unzips there, trains with
  `--data_root $STAGE_DIR`, and releases it. That is How to Run.md §5's
  pattern and the 2026-09-08 smokes proved it end to end (both count guards
  fired with the exact expected counts).
  **But the question exposed a real gap:** the release was a plain
  `cleanup_*` call at the END of the script, which only runs on a NORMAL
  exit — while a chain job being killed at the 24 h wall is the EXPECTED
  ending, up to 3 times per detection run, and would have left ~21 GB of
  extracted COCO on the node's scratch each time. Release is now
  `trap '<cleanup_fn>' EXIT`, installed immediately after staging, and the
  count guards no longer clean up themselves (so it cannot double-run).
  Verified against bash on all three paths: SIGTERM (the wall-clock case),
  a normal exit, and the count-guard's `exit 1` — the trap fired in all
  three. Pinned by a test that also asserts the staged copy is what training
  reads and that the cleanup function appears exactly once per job file.
- **`ckpt_root` decided from the repo's own evidence** (the human did not
  know the path and delegated it):
  `/home/woody/iwi5/iwi5359h/SAGA/dense_ckpt`. Reasons, all file-sourced:
  How to Run.md §1 designates `/home/woody/iwi5/iwi5359h` as the BULK
  filesystem (~1 T soft); the committed COCO/ADE20K/CUB/Aircraft archives
  already live there, so it is demonstrably mounted on the compute nodes
  (the smokes read their datasets from it); and the 2026-09-08 quota output
  put it at 226G of 954G (~728 G free) against hpc's 101G of 100G and
  vault's 974G of 1000G — vault's ~26 G of space headroom is too thin for
  ~10 GiB of dense checkpoints even though its ~200K file limit is
  irrelevant to a handful of large files. All six chains AND both smokes now
  carry `--ckpt_root`; `meta.json` records the resolved `ckpt_dir`, and
  `atomic_torch_save`'s tmp file stays in the same directory so the rename
  is still atomic.

**Pending from HPC (Phase B, remaining):**
1. DONE — `ckpt_root` set to `/home/woody/iwi5/iwi5359h/SAGA/dense_ckpt`
   and all eight launchers regenerated (see above).
2. DONE — the 2.51 GiB of smoke checkpoints were deleted from the
   over-quota filesystem after check_smoke.py had read them.
3. DONE — the names ship as `objectInfo150.txt` and the trainer already
   looks for it; only its parse needs the one-off verification noted above.
   `class_names_file` in the matrix stays null unless that verification
   fails.
4. Submit the six chains, SIX SEPARATE COMMANDS (a glob or brace form runs
   only the first script and passes the rest as ignored positional
   arguments, silently leaving Gate-2 rows unsubmitted — verified, and
   pinned by a test that scans the repo for that shape):
   `bash scripts/submit_det_vitb_baseline_s1.sh`
   `bash scripts/submit_det_vitb_saga_s1.sh`
   `bash scripts/submit_det_vitb_registers_s1.sh`
   `bash scripts/submit_seg_vitb_baseline_s1.sh`
   `bash scripts/submit_seg_vitb_saga_s1.sh`
   `bash scripts/submit_seg_vitb_registers_s1.sh`
5. Sync every day or two WITH THE ENV ACTIVATED (`module load
   python/3.12-conda && source activate
   /home/vault/iwi5/iwi5359h/envs/saga`), because sync_results.sh shells out
   to tools/dense_done.py, which imports torch:
   `bash scripts/sync_results.sh` then `I_AM_HUMAN=1 git push`.
Then Phase C locally (T4/T5 tables, F7 figure,
`results/notes/gate2_report.md`) once the six runs' JSONs are back.

**Note for a later task:** the rewrite saves no best-AP DETECTOR weights,
only the JSON pair (coco_eval_best.json + detections_val.json), which is all
Phase C reads. If `best_epoch != 24` the best detector's weights are not
recoverable — relevant if the "test-time-registers dense rows" task wants to
re-infer from the best model rather than the last.

---

## 2026-09-08 — TASK 07, PHASE C (sink-address analysis)

Phase B came back complete: all 44 `_addr.json` (commit `7ad53a1`), plus
the registers smoke PASSED (`fd98185`, run dir `results/runs_smoke/
e2r_vits_mixup_registers_s1`: gate false / registers 4, epochs 0-1,
val top1 2.834 -> 6.284, no gates//grads//diag/ as predicted, 1251
steps/epoch derived from img_per_sec x wall_time). **Both registers chains
are therefore unblocked; still unsubmitted, still the human's call.**

**Provenance warning (not mine):** commit `efd4f9d "Task 9"` (human,
12:42) contains ONLY these seven TASK-07 Phase-C files — a repo-wide add
swept my in-progress work under a Task-9 label. It is already pushed, so
history was left alone. That snapshot is the PRE-REVIEW version and its
registers numbers have the WRONG SIGN; `17ccb2a` supersedes it. Do not
read numbers out of `efd4f9d`.

**Done (local):**
- C1 `analysis/address_analysis.py` -> `results/tables/sink_address.csv`
  (3348 rows) + `results/figures_data/Faddr.npz`. Cell membership
  IMPORTED from `build_pooled_tables.py` (erratum remap; VOID legacy
  ViT-B mixup-dir trio excluded — pinned by a decoy test). 19 maps, last
  checkpoint only, all sha-matched (legacy vs manifest, run dirs vs
  sibling diag), zero gaps; the 5 never-trained e2r registers runs are
  reported as ABSENT, not as gaps. Stored concentration blocks
  re-derived through an independent implementation (pairwise-difference
  Gini) on all 38 member-basis pairs.
- C2 `plotting/plot_address.py` -> `results/figures/Faddr_draft.{pdf,png}`
  (committed -f): per cell a row of 14x14 freq maps (shared colorbar) +
  the extremal gate layer (shared colorbar on its TRUE range, labelled
  "NOT 0..1", with Bonferroni p and spatial std per panel), plus an
  excess-concentration bar page with noise-flagged bars hatched.
- C3 `analysis/build_sink_address_note.py` ->
  `results/notes/sink_address.md` (generated; every number, including
  prose ranges, read from the CSV — pinned by a test that re-renders the
  note from the committed table).

**Two references that are NOT zero (adversarial review, 11 findings, all
CONFIRMED and all fixed — the first two had inverted a result):**
- **Finite-sample floor.** A spatially UNIFORM ground truth does not give
  Gini 0 from 10000 images: verified null Gini 0.0167 at mass 19.7 but
  0.4886 at mass 0.023. Mass spans 3 orders of magnitude across members,
  so raw Gini/entropy/top-k are NOT comparable. Now every statistic
  carries `_null` (binomial, 400 sims, seed 0) and `_excess`, and Q4
  differences PAIRED excess values. **ViT-B/mixup registers d_Gini:
  +0.1857 (raw, unpaired) -> -0.3570 (excess, paired) — sign inverted.**
  ViT-S/nomix saga also flips (+0.0147 -> -0.0125). SAGA's headline
  survives: S/mixup +0.0937, B/mixup +0.0655.
- **n is not 196.** The maps are spatially smooth, so the iid Spearman
  reference (sd 0.0716) is far too generous; the exact permutation null
  over 8 dihedral x 196 torus rolls (1568 transforms, deterministic, no
  seed) has sd 0.157-0.342, i.e. n_eff ~ 10-42. Every correlation now
  carries that p and null sd.
- Also fixed: Q5 extremal layers are an argmax over 12 layers -> Bonferroni
  p + the opposite-sign extreme now reported; Q5 rendered for BOTH bases
  (it was canon-only while OPEN ITEMS claimed both); Q4 deltas paired (the
  unpaired version inflated ViT-B registers 1.9x); the degenerate-map
  caveat re-diagnosed from ties (mid-rank spread is 94.8% of maximal, so
  ties were NOT the mechanism) to per-position Monte-Carlo error
  (`position_rel_se`); hand-typed prose ranges now computed (one was
  wrong: 1.79x should have been 1.71x); `verify_stored_concentration` now
  fails on a one-sided NaN, verifies `top10_positions`, and refuses
  zero-mass; unpaired SAGA runs emit explicit MISSING rows instead of
  vanishing; the CSV layer-profile pointer now names the layers/comparators
  that actually exist.

**ANSWERS (canon maps; full tables in the note):**
1. **Address exists, but it is diffuse.** H/H_uniform 0.936-0.990 (close
   to uniform) yet top-5 share is 1.71x-4.96x its uniform reference;
   excess Gini 0.143 (S/nomix) / 0.245 (S/mixup) / 0.400 (B/mixup).
2. **Seed-stable — the make-or-break number holds.** Baseline pair
   Spearman: S/mixup 0.9121 (n=6, p 0.0006-0.0019), B/mixup 0.8505
   (n=1, p 0.0013), S/nomix 0.7107 (n=1, p 0.0038). **8/8 pairs p<0.05.**
   S/mixup pairs span legacy runs AND fresh seeded reruns (0.8919-0.9365).
3. **NOT recipe-stable.** Cross-recipe mean 0.3051 canon (2/8 pairs
   p<0.05) and 0.1342 mad (0/8) vs within-recipe 0.9121 (8/8). The
   cross-recipe value sits largely inside its own spatial null.
4. **SAGA does not relocate the address, it consolidates it.** rho vs
   matched baseline 0.79-0.91 (all pairs p<0.05 in all 3 cells) while
   mass falls (-8.08 S/mixup, -3.87 B/mixup, -2.62 S/nomix) and excess
   Gini rises in the two mixup cells. Registers DOES relocate
   (S/mixup 0.4773, B/mixup 0.0063 with p=0.93) and de-concentrates.
5. **The gate tracks the address only in ViT-S/mixup, and negatively.**
   There rho = -0.518..-0.594 at layer 7/8, all negative in BOTH bases,
   4/4 repeats Bonferroni p<0.05 — the gate is suppressed exactly on the
   border ring where sinks form (visible in the figure). B/mixup: 0/2
   Bonferroni-significant and mad disagrees in sign. S/nomix: canon signs
   disagree between the two repeats. Layer-mean rho is near zero
   everywhere, partly by cancellation.
- **Geometry (extra, not in the task):** the address is a RING one patch
  inside the border — peak ring 1 for every mixup member (ring1 0.1377 vs
  ring0 0.0590 for S/mixup baseline) — and FLAT for true-nomix
  (0.0168..0.0265, peak ring 6). The border address is created by the
  mixup recipe; this is the same fact Q3 reports as a correlation.

`pytest -q`: **257 passed** (53 of them TASK-07: 26 sink_address +
27 address_analysis). Three headline numbers independently recomputed
from the source JSONs (Q2 mean/min/max, Q3 mean, Q5 s1 layer 8) — exact.

**Commits:** `17ccb2a` (this phase; supersedes `efd4f9d`).

**Pending from HPC:** nothing for TASK 07. Optional and unblocked: the
four A2 chains (registers smoke passed) — `bash scripts/submit_e2r_*.sh`,
S ~17 h / B ~29 h. Open for the human: whether to run
e2r_vitb_mixup_{baseline,saga}_s2 to take ViT-B seed-stability from 1 pair
to 3, which is the weakest number in the whole analysis.

---

## 2026-09-09 — TASK 11, PHASE A (probe set + attention dumps + localization)

Branch `task/11-teaser` off `main` at `cdae852`. Phase A writes tooling only —
**no results file was created or edited**, and no number exists yet.

**Session start was BLOCKED TWICE, both reported rather than worked around:**
- An unfinished `git pull` merge sat in the worktree (`.git/MERGE_HEAD`
  present, 223 staged TASK-09/e2r result files, `main` ahead 1 / behind 1).
  Per CLAUDE.md I did not commit, complete or stash it. After the human
  reported it done, a successful `git fetch` showed `origin/main` still at
  `d56db59` and the merge still open — the commands had not run (their shell
  is Windows PowerShell 5.1, where `&&` and the `VAR=x cmd` prefix used in my
  bash block are both invalid). Re-reported with PowerShell syntax; the human
  then landed it as `cdae852`.
- The A40 partition answer came back as an unfilled template placeholder.
  Resolved by the human pasting a working a40 header from another project:
  **`--partition=a40 --gres=gpu:a40:1`, `--cpus-per-task=4`.**

**Done (local, by Claude Code):**
- A1 `tools/build_probe_set.py` -> `results/probe/probe_set.json` (built on
  the HPC). Three groups, each frozen INDEPENDENTLY and write-once, each
  carrying a `sha256` over its own canonical serialization so a later group's
  build re-verifies the earlier ones: `curated` (12 images, 4 per stated
  criterion), `random200` (200 images, one per class over 200 seeded classes),
  `boxes20` (20 COCO-val images + GT boxes, `status: pending` until a job with
  COCO fills it). Criteria are CLASS-LEVEL priors and say so
  (`criterion_basis: class_prior`) with the synset pool and rationale recorded
  per image; `--curated-paths` gives the human per-image control
  (`criterion_basis: explicit_paths`). Relative paths + labels, never dataset
  indices. Stdlib only — deliberately no torch, so it runs on a login node
  (TASK-09 lost a sync to exactly that assumption).
- A2 `tools/dump_attention.py` -> `results/runs/<run_id>/attn/probe_<group>_<image_id>.npz`
  (git-ignored via the existing `results/**/attn/`, verified with
  `git check-ignore`). Per image: CLS->patch for ALL blocks both head-meaned
  and per-head (fp16), the prefix mass (so patch+prefix == 1 is checkable on
  the artifact), full maps for the last 4 blocks, per-block token norms,
  SAGA's `sigmoid(phi)` head-mean, and provenance (ckpt sha256, git sha,
  probe-group sha, prefix count). Prefix tokens come from the MODEL via
  `infer_num_prefix_tokens`, never assumed; TTR (TASK-10) plugs in through
  `--model-builder module:function`, so nothing about its interface is
  invented here. Idempotent on (ckpt sha, probe sha, schema); an unreadable
  file is redone, not trusted.
- A3 `tools/localization_score.py` -> `results/figures_data/F1_localization.csv`
  (exactly the mandated 5 columns; provenance in a sidecar `_meta.json`).
  boxes20: `inbox_mass`, `pointing_hit`, and `box_area_frac` — the
  uniform-attention null, written as its own row because TASK-07 established
  that these statistics are meaningless without one. random200: `ring1_mass`,
  `uniform_ring1_mass`, `attn_entropy_bits`.
  **The ring is IMPORTED, not re-derived**: ring membership is obtained by
  probing TASK-07's own `analysis.address_analysis.border_rings` with one-hot
  maps, so the two cannot drift; a test pins the agreement against an
  independent Chebyshev reference.
  **The box frame is measured, not reasoned about** (TASK-09's 3-GPU-day
  lesson): `map_boxes_to_input_frame` reproduces `build_val_transform`'s
  Resize(short side -> 256) + CenterCrop(224) including torchvision's
  truncation and rounding, and the test pushes a white rectangle through the
  REAL transform and asserts the mapped box bounds the surviving pixels.
- A6 `analysis/collect_F1.py` -> `results/figures_data/F1_teaser.npz` (runs on
  the HPC, since the dumps never leave it): thumbnails, per-model attention
  maps, gate maps at layers 7/8, TASK-07 sink-frequency maps, the ring-1 mask,
  and an index recording what is ABSENT. A model that was never dumped has no
  arrays at all — never a zero-filled panel.
  `plotting/plot_F1.py` -> the Phase-C figure; a missing TTR column renders as
  an explicit dotted "pending" panel.
- `scripts/jobs/probe_attention.sbatch` (a40, 1 GPU) + `scripts/stage_probe.sh`.
  Staging releases node-local scratch via `trap cleanup_probe EXIT`
  (TASK-09's wall-clock lesson). Both files stored LF (verified against the
  raw blobs, matching the committed scripts; `.gitattributes` untouched).
- Tests: `tests/test_task11_probe.py`, **37 tests** — probe determinism across
  two builds, seed sensitivity, disjointness, paths resolving under a fake
  root, criteria recorded, write-once refusal, `--if-missing` no-op, boxes20
  fill proven unable to perturb the frozen ImageNet groups, tampering
  detected; ring/border_rings agreement, ring-1 uniform mass == area fraction
  (44/196), entropy bounds; resized_size vs torchvision on 5 shapes, the
  white-rectangle frame measurement, boxes outside the crop vanishing;
  concentrated -> 1.0 and uniform -> box-area fraction, union-not-sum,
  pointing game; per-variant prefix handling (baseline 1 / saga 1 /
  registers 5) end to end, rows-sum-to-1 on the artifact, capture_attention
  no-op re-pinned, guard skip AND guard re-dump on a changed checkpoint,
  truncated dump redone; CSV schema + absent-never-zero; collect_F1 absent
  handling; plot_F1 rendering with a missing TTR column.

**Four defects found by my own tests and fixed before commit:**
1. **The probe groups were not disjoint by construction** — `random200` could
   (and did) draw the very image already in `curated`, which would have put
   Figure 1's illustrations inside the population they are captioned with.
   `build_random200` now excludes already-claimed paths.
2. **`np.savez_compressed(tmp, ...)` appends `.npz`**, so every dump's
   tmp+rename targeted a file that did not exist — the tool could not have
   written a single artifact on the HPC. Replaced by the repo's existing
   `tools.dense_runtime.atomic_npz_save` (which writes to a handle and
   fsyncs); `atomic_write_text` likewise for the CSV/JSON.
3. `collect_F1` hard-raised `KeyError` on a missing `criterion` — a figure
   CAPTION taking down the archive. Now tolerant.
4. `dump_attention` recomputed the gate map and shelled out to `git rev-parse`
   once PER IMAGE (~1160 subprocesses over the planned run). Both hoisted.

**Reconciliations (stated, not improvised):**
- `How to Run.md` is NOT in the repo (it sits one level above), so the
  mandated §6 note cannot be in the commit. The note is committed to
  `docs/HPC_WORKFLOW.md` (new "Partitions" section) AND written into the
  out-of-repo `How to Run.md` §6 as an uncommitted edit.
- The committed stagers extract data this task never reads
  (`stage_imagenet.sh`: 5 train shards, ~20-40 min; `stage_coco.sh`:
  train2017.zip, ~18 GB). `scripts/stage_probe.sh` provides val-only variants
  whose bodies are COPIED from those files with the train lines removed and
  the stage dir renamed, and separate per-dataset stage dirs so both can be
  staged in one job without overwriting `$STAGE_DIR`.
- `--time=06:00:00` is the only SLURM value with no source: TASK-11 estimates
  2-4 h and staging adds ~30-40 min. Flagged in the job file itself.
- A1 calls the probe set a Phase-A deliverable, but it can only be BUILT where
  the data is. The tool is Phase A; the frozen JSON is HPC step 1.
- The dumps are ~3.1 GB for the five models (computed from the npz schema)
  and must never be deleted, but they default onto the `hpc` filesystem that
  TASK-09 recorded as already over its soft quota. All three tools therefore
  take `--attn-root` (default: `--out-root`), and the job exposes it as
  `ATTN_ROOT` so bulk storage is an env var, not a code edit — the TASK-09
  `--ckpt_root` precedent. `collect_F1` keeps `--out-root` pointing at the
  repo tree regardless, because the TASK-07 `diag/*_addr.json` sink maps live
  there; a test pins that the two roots are read separately.

`pytest -q`: **297 passed** (259 pre-existing + 38 TASK-11). No trainer, eval,
diagnose or results file touched.

**Commit:** `[TASK-11] probe set + attention dumps + localization score (phase A)`

**Pending from HPC (Phase B):** run `scripts/jobs/probe_attention.sbatch`;
commit `results/probe/probe_set.json`,
`results/figures_data/F1_localization.csv` (+ `_meta.json`) and
`results/figures_data/F1_teaser.npz`. The `attn/*.npz` are never committed and
must NOT be deleted — Phase C's figure reads them. TTR is expected to be absent
unless TASK 10 has landed; that is handled, not fatal.

---

## 2026-09-09 — TASK 10, PHASE A (test-time registers: the missing 2027 baseline)

Branch `task/10-ttr` off `9887be7`. Track A: give the project an honest
answer to "why not just use test-time registers?" — Jiang, Dravid, Efros &
Gandelsman, *Vision Transformers Don't Need Trained Registers* (NeurIPS 2025
Spotlight, arXiv:2506.08010). A negative result is a fine outcome here
PROVIDED the implementation is validated, which is what Phase A builds.

**Two blockers had to be cleared before any code (both reported, neither
improvised):**
- The worktree was **mid-merge** on `main`: the human's `git pull` had staged
  223 files from `d56db59 "Task 09 half results"` without creating the merge
  commit. Per CLAUDE.md I did not commit or stash it — reported and waited.
  Human finished and pushed it (`cdae852`).
- Then the single worktree was checked out on **`task/11-teaser`** (TASK-11's
  session, created seconds after that merge). Reported per CLAUDE.md's
  one-session-per-worktree rule and waited; human confirmed TASK-11 Phase A
  was merged (`9887be7`) and its HPC job submitted, so the worktree was free.

**A1 — the vendoring instruction is unsatisfiable, and this is the reason.**
The authors' repo IS reachable (`github.com/nickjiang2378/test-time-registers`,
187 stars, official), but it has **no license of its own**: the GitHub API
reports `license: null`, and the only LICENSE in a recursive tree listing at
the pinned commit is `dinov2/LICENSE` — Meta's, covering the vendored DINOv2
subtree, not the authors' `shared/`, `custom_model/`, `clip/clip_*` or
`register_neurons.ipynb`. A1 says to vendor it "with its LICENSE file
intact"; there is no such file, and vendoring all-rights-reserved code into a
repo headed for publication is a different kind of problem from a missing
attribution. **Reconciliation: their code was READ at commit `860df43` to
implement the method exactly — no guessing about which variant — and no line
of it was copied.** `third_party/ttr/PROVENANCE.md` carries the licensing
finding, the URL/sha/date, the method as their code defines it (with the file
and function names), and a table of all eight deviations. If they add a
license later, the pinned commit makes the comparison exact.

**Done (local):**
- `saga/ttr.py` — the two documented steps.
  * `find_register_neurons`: their criterion is the DEFAULT
    (`mean_abs_act_at_outliers_v1` — per image, mean |MLP hidden activation|
    over the outlier tokens, averaged over images that have one); the
    contrast variant A2.1 suggests is selectable
    (`mean_abs_act_outliers_minus_rest_v1`). The criterion string is written
    into every artifact, so no result is ever ambiguous about which was used.
    Hook point `blocks[l].mlp.act`, the same one their
    `Dinov2HookManager.neuron_activation_component` returns. Returns the full
    descending ranking, so the `--n-neurons` sweep is a PREFIX of one scan,
    never a rescan per k. Raises loudly when no image has an outlier, as
    theirs does.
  * `apply_ttr`: zero-initialised extra tokens (their README states the
    initialisation), `scale * sign_max(...)` written into them at the selected
    neurons, those neurons zeroed on the patch tokens, CLS never touched.
  * **Token placement deviates deliberately.** They append
    (`[CLS][PATCH][REG]`); A2.2 requires patch count and ordering to be
    untouched, so extras are spliced in as PREFIX tokens
    (`[CLS][EXTRA][PATCH]`) and `num_prefix_tokens` grows. `metrics.py`,
    `diagnose.py` and `sink_address.py` therefore need no change. The two
    placements are equivalent by permutation-equivariance (the extras carry
    no positional embedding) — **measured, not asserted**: max |Δ| ≈ 2e-6 on
    tokens of norm ≈ 16, i.e. fp32 round-off.
  * **Outliers are detected among PATCH tokens only**, unlike theirs. Every τ
    in this repo is calibrated on `last_block_patch_norms`, so thresholding
    CLS with a patch-calibrated τ is a category error and would pollute the
    scores with CLS-specific activations.
  * τ comes from the base cell's canon file, not their hardcoded 30 — our
    scales differ by an order of magnitude across cells (20.85 / 127.31 /
    22.86), so one constant cannot serve.
- **A design flaw the repo's own B6 tripwire caught.** Setting
  `model.num_prefix_tokens = 1 + n_extra` trips
  `SAGAViT._interpolate_pos_embed`, which reads the same attribute to
  describe the pos-embed layout and hard-errors unless it is 1 — and it is
  right to, since it runs before block 0, i.e. before the extras exist. The
  two readers have incompatible needs during a TTR forward, so `apply_ttr`
  now yields a `TTRModel` proxy: the wrapped model is left BIT-IDENTICAL (no
  attribute of it is written at all) while every tool that asks the proxy
  gets the patched count. `.blocks` is the same object, so hook-based tools
  attach to the real blocks.
- `apply_ttr` REFUSES SAGA-gated models: `SpatialGate.forward` hard-codes
  `n_patches = N - 1`, so extra prefix tokens would be gated as patches (or
  raise a grid mismatch). TTR is a baseline-model intervention here. An empty
  selection is an EXACT no-op — nothing appended, no hook installed — because
  appending an unused zero token would NOT be one (other tokens attend to it).
- `tools/ttr_validate.py` — the A3 gate. Per sweep value: sink count under the
  base cell's canon τ and top-1 on a fixed 5 000-image subset of the frozen
  split (first 5 per class, class-balanced, no RNG), with the neuron scan run
  on a DISJOINT strided 500-image set from the other half of the split.
  A3 states the PASS criterion in words only, so both numbers are flags
  (default: ≥50 % relative sink reduction AND ≤1.00 point top-1 drop) and both
  are printed and recorded beside the verdict. Exit **0 = PASS, 3 = a measured
  FAIL** (do not run the matrix — the validated negative is the deliverable),
  anything else = the tool itself broke. Writes `neurons.json` + `sweep.csv`
  (row `n_neurons=0` is the unpatched reference, measured the same way) +
  `validate.json`.
- `tools/ttr_prepare_run.py` + `tools/ttr_derive.py` — the Phase-B pair.
  **`tools/eval.py` and `tools/diagnose.py` keep a ZERO diff**: wrapping their
  eval loops in a context manager would mean re-indenting the two tools that
  produce every paper number, so `ttr_derive` imports their exact pieces
  instead (`shard_counts`, `counts_to_metrics`, `build_val_transform`, the
  `n == 50000` assert, `compute_diagnostics`) and a test pins that its
  UNPATCHED path (`--n-neurons 0`) reproduces `tools/eval.py`'s top1/top5/loss
  to the last digit. Single-GPU by design (Phase B is one A40, eval-only), so
  no sharding and no all_reduce. npz before JSON, sha-keyed skip guards on
  both steps, so a requeue redoes only what is missing.
- **The integration point A5.3 actually depends on, found and fixed:**
  `apply_fixed_thr --version canon` resolves a diag file's cell via
  `recipe_from_run_dir`, which reads `<run_dir>/config.resolved.yaml`. A TTR
  run dir had none, so the canon backfill would have refused every TTR diag
  ("cannot resolve recipe_actual"), leaving no `sink_fixed_canon` — and then
  `sink_address.py`'s HARD cross-check would have had nothing to compare
  against. `ttr_prepare_run.py` writes that config with the **BASE** cell's
  arch/variant/recipe (plus a `ttr:` provenance block), so the existing
  backfill resolves `vit_small|mixup` and works unmodified. Pinned by a test
  that runs the real resolver and then the real `apply_fixed_thr` CLI on a
  derived TTR run dir. Related trap also guarded: a diag stem with exactly 7
  underscore-separated parts is read as a LEGACY e2 filename by
  `recipe_from_stem`, which would resolve the cell from the NAME instead of
  the config — `ttr_derive` refuses such a `--diag-name`.
- `scripts/jobs/ttr_validate.sbatch` + `ttr_matrix.sbatch`. The a40 header is
  copied from `scripts/jobs/probe_attention.sbatch` (TASK-11), which took it
  from the human's own working a40 job; the same strings are in
  `docs/HPC_WORKFLOW.md` "Partitions", so nothing was invented — a test pins
  both files against that reference header. Reuses TASK-11's committed
  val-only stager rather than adding a third one. Scratch released by
  `trap cleanup_probe EXIT` (TASK-09's lesson), canon backfill BEFORE
  `sink_address` (its cross-check needs that field), `N_BEST` required rather
  than defaulted, no wildcard-submit shapes.
- `results/README.md` documents both new trees, including WHY they are split:
  `results/ttr/<base_run_id>/` for the TTR-specific artifacts (A2.1) and
  `results/runs/ttr_<base_run_id>/` for the standard eval/diag pair (A5.3),
  the latter under `runs/` precisely so the existing collectors glob it.
- `tests/test_task10_ttr.py` — 61 tests (CPU, fake data, tiny models). The
  four A4 items are marked `[A4]`; the rest pin the mechanism (`sign_max`
  sign convention, the redirection's exact effect on extras/patches/CLS,
  `scale`, each `normal_values` mode), the guards (gated-model refusal,
  index/range validation, nesting, hook removal on exception), the selection
  helpers, the proxy, the validate units (canon τ read from the REAL
  committed file, subset carving, verdict incl. the undefined-when-zero
  case), and the two CLIs end to end through subprocess.

**Reconciliations (task file vs reality) — stated, not improvised:**
1. A1's vendoring branch is unsatisfiable (no upstream license). See above.
2. A5.2's example command omits `--data`, which the tool cannot run without
   (it needs ImageNet val). `--data` is required and appears in the printed
   block.
3. A2.1 puts `neurons.json` under `results/ttr/<base_run_id>/` while A5.3
   puts eval/diag under `results/runs/ttr_<base_run_id>/`. Both are honoured;
   the split is deliberate and documented, not a typo I picked a side on.
4. A5.3's "the legacy ViT-S baseline (recipe_actual=mixup)" is ambiguous:
   BOTH legacy ViT-S baseline checkpoints have `recipe_actual=mixup` (the
   nomix DIRECTORY is an additional repeat of the mixup cell, per the
   erratum). The matrix uses the mixup-dir one
   (`ViT-S_baseline/last.pth`, manifest sha `cd926200127b98cb…`); the sibling
   `ViT-S_baseline_nomix/last.pth` (`8d980da27a4b571a…`) is named in the job
   file as an available second repeat. **Human's call if both are wanted.**
5. A5's step 3 cannot be written concretely yet: it needs the best sweep
   value, which only exists after the gate runs. `ttr_matrix.sbatch` takes it
   as `N_BEST`.
6. Each of the four cells gets its OWN neuron scan (register neurons are
   per-model); only the chosen `n` is shared. The matrix job therefore
   re-runs the gate per cell, which also yields a per-cell sweep — cheap
   relative to a 50k eval, and better evidence. Rerunning is deterministic
   apart from the `timestamp` field.

**Per the human's ruling (separate commit):** task branches are local-only
and the HPC always runs `main`, so the merge + push comes BEFORE the HPC
block. `SAGA_Code/CLAUDE.md`'s branching section was updated in place — but
that file is tracked by NO repository (`git ls-files | grep -i claude` is
empty; it sits one level above the repo), so it cannot be committed. The
versioned copy went into `docs/HPC_WORKFLOW.md` under "Branches". The old
CLAUDE.md text was not merely incomplete but wrong: it told the HPC block to
`git checkout task/<NN>-<slug>`, a branch that is never pushed.

`pytest -q`: **358 passed** (297 pre-existing + 61 TASK-10). No results file
was written or edited by this phase; `eval.py`, `diagnose.py`,
`sink_address.py`, `apply_fixed_thr.py` and the trainers are untouched.

**Commits:** `17ce433` (phase A), `8aabdfb` (branch policy).

**Worktree hazard, recorded because it bit twice in one task.** Between my
second and third commits the shared worktree was switched out from under me:
the reflog shows `checkout: moving from task/10-ttr to main` followed by
`pull origin main: Fast-forward` to `004c152 "TASK_09 all results"` (the
human collecting HPC results — det_vitb_registers_s1 is now complete, 25
log rows, so TASK-09 has all six runs). My TASK_LOG commit therefore landed
on **`main`** instead of `task/10-ttr`, leaving `main` one unpushed commit
ahead with a task commit on it — which the next `git push origin main` would
have published out of order. Cleaned up: the commit was cherry-picked onto
`task/10-ttr` (`docs/TASK_LOG.md` was byte-identical on both sides, so no
conflict) and `main` was moved back to `origin/main` with `git branch -f`
while it was NOT checked out — a pure ref move, no `--hard`, nothing of the
human's touched. Verified afterwards: `main == origin/main == 004c152`, the
three TASK-10 commits all on `task/10-ttr`, zero file overlap between this
branch and `004c152`, `git merge-tree` clean, `pytest -q` 358 passed. The two
task branches in this one worktree remain the standing risk; CLAUDE.md's
`git worktree add ../SAGA-<NN>` is the documented way out.

**Pending from HPC (Phase B):** merge `task/10-ttr` into `main` and push,
then `sbatch scripts/jobs/ttr_validate.sbatch` → **report the PASS/FAIL line
back before anything else runs**. On PASS: `N_BEST=<n> sbatch
scripts/jobs/ttr_matrix.sbatch`. On FAIL for every n the matrix must NOT run
and Phase C writes the validated negative instead. Files expected back: the
sweep table (`results/ttr/*/{neurons.json,sweep.csv,validate.json}`) and, on
PASS, 4 × {eval, diag, addr} under `results/runs/ttr_*/`. Then Phase C
locally on `task/10-ttr-c` (T_ttr.csv + `results/notes/ttr_baseline.md`).
