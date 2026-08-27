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
