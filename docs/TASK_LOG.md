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
