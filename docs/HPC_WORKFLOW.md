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

**Switching partition without editing the file.** sbatch command-line flags
override the `#SBATCH` directives in the script, so when a40 is busy:

```bash
sbatch --partition=a100 --gres=gpu:a100:1 scripts/jobs/<job>.sbatch
```

Do NOT copy the job file to change its header — a duplicate drifts out of
sync, its provenance comment becomes false, and the repo's job-file tests are
keyed to the real names. (A copy named `... copy.sbatch` also cannot be
submitted unquoted at all: the space splits it into two arguments and sbatch
opens neither.)

## The conda module was renamed — `module load python`

`How to Run.md` §2/§3 documents `module load python/3.12-conda`. That
modulefile **resolved on 2026-09-08** (TASK-08's ft array completed all 16
runs through it) and was **gone by 2026-09-10**. The failure mode is nasty
rather than loud: the load fails, `source activate` then fails with
`activate: No such file or directory`, and the job runs `/usr/bin/python`,
which has no torch — so it dies deep inside Python, after staging.

`module load python` (no version) is what works, confirmed on the cluster by
the human 2026-09-10. All four LIVE env scripts —
`scripts/env_alex.sh`, `detection/`, `segmentation/` and
`classification/scripts/env_alex.sh` — now try the pinned name first, fall
back to the generic one, and print a loud WARNING if neither loads;
`tests/test_task10_ttr.py` pins that shape. The legacy launchers
(`detection/scripts/e3_eval_tinyx.sh`, `evaluation/e5_lost/…`,
`evaluation/e6_finegrained/e6_*.sh`) still name the dead module and were
left alone on purpose: they are superseded provenance and must not be made
runnable by accident.

**Job files should not depend on the module at all.** Address the env's
interpreter directly, as `env_alex.sh`'s own `TORCHRUN` line does:

```bash
PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}
```

and gate on it — `$PY -c "import torch, timm"` inside an `if !` with an
`exit 1`, placed BEFORE any data staging. An unchecked import cost TASK-10 a
job that staged 50 000 images and then died.

## `set -u` goes AFTER the env sourcing, never before

`scripts/env_alex.sh` sources `/etc/profile`, which runs the site scripts in
`/etc/profile.d/`. At least one of them dereferences an unset variable —
`debuginfod.sh` line 8, `DEBUGINFOD_URLS` — and under `set -u` bash aborts
the **calling** script. The job then dies during env setup with nothing in
the `.err` but:

```
/etc/profile.d/debuginfod.sh: line 8: DEBUGINFOD_URLS: unbound variable
```

and not one line of the job body runs. Measured on Alex 2026-09-10 by
TASK-10's validate job. So in every job file:

```bash
source .../scripts/env_alex.sh      # and the stager
cd $CODE_ROOT
set -u                              # only here
```

Every job file that had completed a real run (TASK-08 `ft_*`, TASK-09
`det_*`/`seg_*`/`dense_smoke_*`) already had this ordering; the three that
did not were fixed and `tests/test_task10_ttr.py` now pins it repo-wide over
`scripts/**/*.sbatch`. A `${VAR:?message}` guard needs no `set -u` — it
aborts on unset or empty regardless — so those can still sit at the top.

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
