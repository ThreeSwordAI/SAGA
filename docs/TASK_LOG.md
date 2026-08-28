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
