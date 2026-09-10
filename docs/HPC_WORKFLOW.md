# HPC workflow — the local ⇄ cluster round-trip

One page on how code and results move between the local machine (where
Claude Code works) and the FAU Alex cluster (where the human runs GPU jobs).
Cluster specifics — login, modules, conda env, SLURM headers, storage map —
live in the human's `How to Run.md` (one level above the repo); this page is
only the choreography.

## The loop

```
┌──────────────── LOCAL (Claude Code) ────────────────┐
│ 1. edit code / tools / tests, run pytest (CPU only) │
│ 2. git commit  (never push)                         │
└───────────────┬─────────────────────────────────────┘
                │ human: I_AM_HUMAN=1 git push origin main --tags
                ▼
┌──────────────── HPC (human) ────────────────────────┐
│ 3. git pull inside CODE_ROOT                        │
│ 4. activate env, run the printed command block      │
│    (sbatch jobs / manifest / eval / diagnose)       │
│ 5. small outputs land in results/ (JSON, CSV,       │
│    phi_e*.npz); large ones stay on vault/scratch    │
│ 6. git add results/... && git commit && push        │
└───────────────┬─────────────────────────────────────┘
                │ human (local): git pull
                ▼
┌──────────────── LOCAL (Claude Code) ────────────────┐
│ 7. read results/, build tables/figures, next task   │
└─────────────────────────────────────────────────────┘
```

Claude Code has **no HPC access**: every GPU step is delivered as an exact,
copy-pasteable command block based on `How to Run.md`, then Claude STOPS and
waits for the results to come back through git.

## Branches — task branches are local-only, the HPC runs `main`

Human's ruling, 2026-09-09. `CLAUDE.md` carries it as the standing rule, but
that file lives one level above the repo and is therefore untracked, so the
policy is recorded here too, where it is versioned.

- One branch per task, `task/<NN>-<slug>`, created off `main`. Every commit
  for that task goes there. Claude never pushes.
- **Task branches never leave this machine, and the HPC never checks one
  out.** The HPC always runs `main`.
- So the merge comes BEFORE the HPC block, not after it:

  ```
  git checkout main
  git merge --no-ff task/<NN>-<slug>
  I_AM_HUMAN=1 git push origin main
  ```

  only then can the HPC's `git checkout main && git pull` see the task's
  code. Every phase that hands work to the HPC prints the merge + push
  commands ahead of the HPC block.
- HPC result commits land on `main` and come back with a local `git pull`.

## Partitions

`How to Run.md` §6 documents only `--partition=a100 --gres=gpu:a100:N`. The
**a40** partition is also available and is addressed as
`--partition=a40 --gres=gpu:a40:1` (confirmed by the human from a working a40
job on this cluster, 2026-09-09; first used here by
`scripts/jobs/probe_attention.sbatch`). Its wall limit is not documented —
TASK-11's job asks for 6 h.

## Large files never travel through git

- Git-ignored: checkpoints (`*.pth`, `*.pt`), datasets, raw attention dumps
  (`results/**/attn/`), norm dumps (`results/**/*_norms.npz`), `results/**/ckpt/`.
- Committed: `*.json`, `*.csv`, `meta.json`, `log.csv`, gate files
  `phi_e*.npz`, `figures_data/*`.
- The full contract is in `results/README.md`. A local pre-push hook blocks
  accidental non-human pushes; humans push with `I_AM_HUMAN=1 git push ...`.

## End-of-task protocol (from CLAUDE.md — always, in this order)

1. Summary of what changed and why (short).
2. `git status` clean check + the commit(s) made.
3. **FOR THE HUMAN — LOCAL:** the push command(s).
4. **FOR THE HUMAN — HPC:** exact command block (pull, env activation per
   `How to Run.md`, the runs/scripts to execute, and which output files must
   be committed on the HPC and pushed back).
5. The list of files expected back before the next task can start.
6. Append a dated entry to `docs/TASK_LOG.md` (task id, what was done,
   commits, what is pending from HPC).
7. STOP. Do not begin the next task.
