# TASK 09 — handoff: dense prediction toward GATE 2

**Status: content-complete.** Nothing is pending from the HPC.
**Verdict: Gate 2 = PARTIAL.**
Written 2026-09-15. Covers TASK 09 end to end (Phase A local fixes, Phase B
HPC execution, Phase C analysis). Everything below is read from a committed
file; where a number has no file, it says MISSING.

Read this with `results/notes/gate2_report.md`, which is the generated,
authoritative statement of the verdict. This document is the wider context:
what was done, what was deliberately not done, which file holds which
number, and what the next person must not assume.

---

## 1. The one-paragraph answer

Gate 2 asked: *does SAGA's benefit show where dense practitioners need it —
AP_S and background-class IoU?* On one seed per cell: **AP_S moves against
SAGA (−0.844)** while **background-class IoU moves for it (+0.285 IoU
points)**. Exactly one of the two criteria moved in SAGA's favour, so the
frozen rule returns **PARTIAL**. SAGA does win overall detection AP
(+0.427), driven by medium and large objects; ADE20K mIoU is a wash
(single-scale −0.242, multi-scale +0.002). Every dense number in the repo
before this task was void (three separate bugs), so these are the first
valid ones.

**The caveat that must travel with all of it:** there is exactly one run per
cell. No delta here has a standard error, and the −0.844 the verdict turns
on is the same order as the within-run epoch-to-epoch spread of AP_S
(0.394–0.759).

---

## 2. Status board

| Item | State |
|---|---|
| Detection runs (baseline / registers / saga) | **done**, 25/25 epochs each |
| Segmentation runs (baseline / registers / saga) | **done**, 80/80 epochs each |
| T4 / T5 tables + per-category / per-class appendices | **done** |
| `gate2_report.md` (generated, verdict) | **done** |
| F7 qualitative figure (ADE + COCO halves) | **done** |
| Second detection seed `det_vitb_{baseline,saga}_s2` | **prepared, NOT submitted** (human's decision, 2026-09-13) |
| Segmentation seed repeats | **not done, never planned** |
| Registers backbone matched to the others | **NO** — see §7 |
| Test-time-registers dense rows | **not in scope** (became TASK 10) |
| Tests | 115 pass (`tests/test_task09_dense.py` 85, `tests/test_task09_phasec.py` 30) |

---

## 3. What the task was, and what actually got fixed

The brief named three bugs. Fixing them exposed four more that would each
have produced a confident wrong number, plus several robustness defects.

### 3.1 The three from the brief

| Bug | What it was | Fix |
|---|---|---|
| **B5** double normalization | `detection/data/transforms.py` applied the ImageNet `Normalize`, and torchvision's `GeneralizedRCNNTransform` applied it again with its own ImageNet defaults. The backbone never saw the distribution it was pretrained on. | `FasterRCNN(..., image_mean=[0,0,0], image_std=[1,1,1])` — the dataloader stays the single normalization. `ViTDetector.check_normalization()` asserts it on a real batch, **on device**, at step 0 of every job, and records the evidence in `meta.json`. |
| **B6** registers backbone | The timm `reg_tokens=4` model was wrapped in `SAGAViT`, which copies only `cls_token`/`pos_embed`. The `reg_token` parameter was silently **dropped** (it survived only because the load was `strict=False`) and the pos-embed layout was misread. | The timm model is used directly via its own `forward_intermediates()`. `SAGAViT.__init__` now **refuses** any model with `num_prefix_tokens != 1`. Loading is `strict=True` with a sha256 assert. |
| **B4** mIoU ignore leak | Per class, `union = (pred==c) OR (gt==c)` over **every** pixel, so the 255 "unlabelled" pixels (~8% of ADE20K) inflated the union whenever the model predicted `c` there — biasing mIoU downward by a variant-dependent amount. | A confusion matrix accumulated over `gt != 255` only; intersection and union both derive from it, so ignored pixels are structurally absent from both. The matrix is written out (`conf_matrix.npz`). |

### 3.2 Found while fixing them — each would have produced a wrong number

1. **COCO category-id space mismatch.** Predictions lived in the contiguous
   1–80 label space while the ground truth kept original COCO ids (1–90).
   Classes were compared against *different* classes and every annotation
   with an original id > 80 was dropped. Fixed by mapping predictions back
   through `idx_to_coco_id` and scoring against the official annotation
   file. Pinned by a test where correctly mapped predictions score AP 100.0
   and unmapped ones do not.
2. **Coordinate-frame mismatch — the most serious.** This pipeline resizes
   in the *dataloader*, so by the time torchvision's `postprocess` runs its
   "original image sizes" are already the resized ones and the rescale is a
   no-op. Predictions came out in the resized frame; COCO ground truth is in
   the original frame. **A perfect detector would have scored AP ≈ 0.**
   Found during Phase A and independently reproduced by an adversarial
   review: an oracle prediction set scores 0.0 unmapped and 100.0 mapped.
   Fixed by `boxes_to_original_frame()`; pinned by an end-to-end test that
   fails when the mapping is removed.
3. **`pycocotools.loadRes` mutates the caller's dicts**, adding a fabricated
   box-outline `segmentation` polygon, an `area`, an `id` and `iscrowd` to
   every prediction — which then got serialised into `detections_val.json`,
   inflating it ~2.2× with a mask-shaped value the model never predicted.
   Fixed with a shallow copy per entry.
4. **`fpn_indices` asymmetry.** timm's `forward_intermediates` appends one
   entry per *matching block*, so a repeated or descending index list came
   back with the wrong length, while `SAGAViT`'s honoured the request
   literally. The registers path now re-expands to the requested
   order/multiplicity. (Production uses `[3,6,9,11]`, so this never reached
   a run — it was caught by a production-resolution test.)

### 3.3 Robustness defects fixed along the way

- The "new best AP" branch wrote its checkpoint to `last.pth`, so best-AP
  state was indistinguishable from last state.
- COCOeval's `-1` "no ground truth in this slice" sentinel was being written
  as `-100.0`, i.e. a non-measurement that looks like a measurement. Now
  `null`.
- Per-rank BatchNorm buffers made the segmentation mIoU depend on which val
  images landed on which rank. Buffers are now synced from rank 0 before
  every eval.
- Staged data (~21 GB of COCO on node-local scratch) was released only on a
  *normal* exit — but a chain job being killed at the 24 h wall is the
  expected ending, up to 3× per run. Release is now an `EXIT` trap, verified
  against bash on SIGTERM, normal exit, and the count-guard's `exit 1`.
- `scripts/sync_results.sh` called bare `python` with stderr suppressed; on
  a login node without the conda env that made every *finished* run report
  as "still running", silently never shipping `detections_val.json`.
- Master ports collided with `classification/scripts/e2_nomix_alex.sh`,
  whose port is computed as `29800 + SLURM_ARRAY_TASK_ID` and so is
  invisible to a grep for a literal port. Dense ports moved to 29850+ and
  are now **pinned per run in the matrix** (deriving them from the sorted
  index renumbered already-executed runs when a new one was added).
- The three legacy evaluators (`detection/tools/{evaluate,analyze}.py`,
  `segmentation/tools/evaluate.py`) computed metrics with the old broken
  definitions and no hash check; they now refuse to run. The legacy
  launchers `e3_train_alex.sh` / `e4_train_alex.sh` carry a SUPERSEDED
  banner.

---

## 4. RESULTS

### 4.1 T4 — COCO detection

ViT-B/16, 25 epochs, best-AP epoch (24 for all three), COCO val2017, **all
5000 images**, official `instances_val2017.json` as ground truth.
Source: `results/tables/T4_coco.csv`.

| | AP | AP50 | AP75 | **AP_S** | AP_M | AP_L |
|---|---|---|---|---|---|---|
| baseline | 34.912 | 57.142 | 37.210 | **18.460** | 37.049 | 49.867 |
| registers | 35.135 | 57.115 | 37.028 | 18.391 | 37.209 | 49.636 |
| saga | 35.339 | 57.476 | 37.624 | **17.616** | 37.752 | 51.031 |
| **saga − baseline** | **+0.427** | +0.334 | +0.414 | **−0.844** | +0.703 | +1.164 |
| **registers − baseline** | +0.223 | −0.027 | −0.182 | −0.069 | +0.160 | −0.231 |

Recall (same source):

| | AR_1 | AR_10 | AR_100 | AR_S | AR_M | AR_L |
|---|---|---|---|---|---|---|
| baseline | 30.057 | 45.902 | 48.339 | 29.726 | 51.935 | 63.102 |
| registers | 30.140 | 46.440 | 48.881 | 29.863 | 52.574 | 63.391 |
| saga | 30.193 | 46.502 | 48.847 | 29.098 | 52.314 | 64.415 |
| **saga − baseline** | +0.136 | +0.600 | +0.508 | **−0.628** | +0.379 | +1.313 |

AR_S moves the same way as AP_S. The small-object deficit is consistent
across both metrics, not an artifact of one.

### 4.2 T5 — ADE20K segmentation

ViT-B/16, 80 epochs, best-val-mIoU epoch, 2000 val images, 480,230,527
labelled pixels (identical across all three runs — the B4 masking is
deterministic). Source: `results/tables/T5_ade20k.csv`.

| | mIoU ss | mIoU ms | pixel acc | mean acc | best epoch |
|---|---|---|---|---|---|
| baseline | 43.5768 | 43.9663 | 79.8750 | 55.3272 | 79 |
| registers | 42.7328 | 43.5637 | 79.5446 | 56.1654 | **29** |
| saga | 43.3350 | 43.9685 | 79.7288 | 55.3007 | 79 |
| **saga − baseline** | **−0.2418** | **+0.0022** | −0.1462 | −0.0265 | |
| **registers − baseline** | −0.8440 | −0.4026 | −0.3304 | +0.8382 | |

`ms` = multi-scale + flip TTA on the 512×512 val tensor (scales
0.5/0.75/1.0/1.25/1.5), which is **not** whole-image multi-scale testing —
the committed val pipeline resizes every image to 512². Recorded in
`miou_ms.json`.

### 4.3 The background classes (the Gate-2 second criterion)

Single-scale per-class IoU, in IoU **points**. Source:
`results/tables/T5_ade20k_per_class.csv`.

| class | index | baseline | saga | **saga − baseline** | registers |
|---|---|---|---|---|---|
| wall | 0 | 72.7150 | 72.9116 | **+0.1966** | 71.9774 |
| sky | 2 | 94.1672 | 94.2816 | **+0.1144** | 94.0429 |
| floor, flooring | 3 | 76.0690 | 76.6138 | **+0.5448** | 76.6421 |
| **mean of the three** | | | | **+0.2853** | |

### 4.4 The verdict, and the rule it follows

The rule is frozen in `analysis/build_gate2_note.py` and pinned by a
parametrised test, including the tie case:

```
moved_ap_s := (saga.AP_S - baseline.AP_S) > 0        -> -0.844  -> NO
moved_bg   := mean{wall,sky,floor} of delta IoU > 0  -> +0.2853 -> YES
PASS = both moved | PARTIAL = exactly one | FAIL = neither
```

A delta of exactly zero is **not** "in favour". → **PARTIAL**.

### 4.5 What the aggregates are made of

Neither aggregate says what it first appears to say. Source:
`results/tables/T4_coco_per_category.csv`, `T5_ade20k_per_class.csv`.

| distribution | n | SAGA wins | mean | median |
|---|---|---|---|---|
| COCO per-category AP | 80 | 57 | +0.427 | +0.408 |
| COCO per-category AP50 | 80 | 45 | +0.334 | +0.131 |
| **COCO per-category AP_S** | 80 | **43** | **−0.843** | **+0.055** |
| ADE20K per-class IoU (pts) | 150 | 69 | −0.2417 | −0.3890 |

**Read the AP_S row carefully.** SAGA is above baseline on AP_S in a
*majority* of categories (43/80) and the median delta is *positive*
(+0.055). The negative mean is carried by a few categories with very few
small instances — worst: toaster −34.10, bed −20.00, bear −17.05; best:
stop sign +11.91, train +10.83, donut +4.67. So "SAGA is worse at small
objects" is **not** what these numbers establish; "the AP_S aggregate is
lower, and it is dominated by a handful of noisy categories" is.

### 4.6 Uncertainty

One run per cell. No standard error exists for any delta, and none of the
project's usual machinery (paired SE, `significant_2xSE`, Welch — see
`analysis/build_pooled_tables.py`) can be applied. The only noise estimate
the data can supply is the within-run spread of AP_S over each run's last
three evaluation epochs:

| run | AP_S at the last three evals | spread |
|---|---|---|
| det_vitb_baseline_s1 | 17.861, 18.154, 18.460 | 0.599 |
| det_vitb_registers_s1 | 17.632, 18.159, 18.391 | 0.759 |
| det_vitb_saga_s1 | 17.472, 17.866, 17.616 | 0.394 |

The −0.844 the verdict turns on is the same order as that spread. This is a
statement about the precision of a *single* run, not a seed-level error bar.

---

## 5. WHICH FILE HAS WHICH RESULT

### 5.1 Headline results

| File | Contains |
|---|---|
| `results/notes/gate2_report.md` | **The verdict.** Generated; both criterion numbers, both tables, the distribution facts, the uncertainty section, the provenance caveats. Regenerate with `analysis/build_gate2_note.py`. |
| `results/tables/T4_coco.csv` | Detection: 6 APs + 6 ARs per variant, plus delta rows. Provenance columns (`backbone_run/source/sha256`, `git_sha`, `epoch`, `n_val_images`, `seed`). |
| `results/tables/T4_coco_per_category.csv` | 80 COCO categories × {AP, AP50, AP_S} per variant + deltas vs baseline. |
| `results/tables/T5_ade20k.csv` | Segmentation: mIoU ss/ms, pixel acc, mean acc per variant + deltas; `ms_scales`, `n_classes_scored`, `n_labelled_pixels`. |
| `results/tables/T5_ade20k_per_class.csv` | 150 ADE20K classes × IoU per variant, as **fractions and IoU points**, + deltas, with `is_background_class` flagging wall/sky/floor. |
| `results/figures/F7_dense_draft.pdf` (+ `.png`) | Qualitative: 3 ADE20K probe images × {input, GT, baseline, registers, saga}; 2 COCO small-object crops × {baseline, registers, saga}. |

### 5.2 Per-run raw artifacts

`results/detection/<run_id>/` — 5 files each, ~16 MiB (the dump dominates):

| File | Contains |
|---|---|
| `coco_eval_best.json` | All six APs + six ARs + **80-entry `per_category`** at the best-AP epoch; `n_detections`, `detections_sha256`, `val_protocol`, backbone provenance, the **B5 `normalization_check` evidence**, `git_sha`, `seed`, `smoke: false`. |
| `detections_val.json` | Raw COCO-format predictions at that epoch — official category ids, **original image coordinates**, keys exactly `{image_id, category_id, bbox, score}`. baseline 189,825 dets / 15.7 MiB; registers 187,557 / 15.5 MiB; saga 194,413 / 16.1 MiB. Enables PR curves and qualitatives with no re-inference. |
| `log.csv` | Per epoch: `epoch, lr_backbone, lr_head, train_loss, AP, AP50, AP75, AP_S, AP_M, AP_L, img_per_sec, wall_time`. AP columns are empty on non-eval epochs. |
| `meta.json` | run_registry provenance: seed, git sha, world size, hostname, GPUs, torch/timm versions, backbone run + expected vs observed sha256, `ckpt_dir`, `end_time` (the completion marker), `resumes[]`, `normalization_check`. |
| `config.resolved.yaml` | The fully resolved config the run actually used. |

`results/segmentation/<run_id>/` — 67 files each, ~1.3 MiB:

| File | Contains |
|---|---|
| `miou_ss.json` | Single-scale mIoU, pixel/mean accuracy, 150-entry `iou_per_class` (fractions), `n_classes_scored`, `n_labelled_pixels`, **`units` block**, `miou_definition` (states the B4 fix), `ignore_index`, probe-list sha, backbone provenance. |
| `miou_ms.json` | Same for multi-scale+flip TTA, plus `ms_scales` / `ms_flip`, computed once at the end from the val-selected weights. |
| `per_class_iou.csv` | 150 rows: `class_index, name, iou_frac, intersection, union, gt_pixels, pred_pixels`. Names resolved from the dataset's own `objectInfo150.txt`. Written with `csv.writer` because ADE20K names contain commas. |
| `conf_matrix.npz` | The 150×150 confusion matrix behind every number above — the strongest audit surface; a test recomputes the whole per-class table from it. |
| `preds_fixed20/` | 60 files: `<stem>_pred.png`, `<stem>_gt.png`, `<stem>_img.jpg` for the 20 frozen probe images. Mode-L PNGs hold raw class indices. |
| `log.csv`, `meta.json`, `config.resolved.yaml` | As above (mIoU column instead of AP). |

### 5.3 Frozen inputs (committed *before* the results they select over)

| File | Contains |
|---|---|
| `results/probe/ade20k_fixed20.json` | The 20 ADE20K val stems, `sorted(random.Random(0).sample(range(1,2001),20))`, with the rule recorded. Committed before the first segmentation eval. |
| `results/figures_data/F7_coco_selection.json` | The 2 COCO images F7 shows (**480275**, **233825**), chosen deterministically from committed detections: small = area < 32² px, score ≥ 0.30, ranked by \|n_small(saga) − n_small(baseline)\|, ties by image_id; 1825 images were eligible. Write-once. |
| `results/figures_data/f7_coco/` | The exported crops + `crops.json` (GT and detections in crop coordinates). 233825: 640×480 → 468×301, 19 GT (15 small). 480275: 640×471 → 461×146, 12 GT (7 small). |
| `configs/dense_matrix.yaml` | The 8 run definitions, pinned backbone sha256s, pinned ports, `ckpt_root`, chain lengths, and the `submitted: false` flags. |

---

## 6. CODE MAP

| File | Role |
|---|---|
| `detection/models/detector.py` | B5 fix + `check_normalization()`. |
| `detection/models/backbone.py` | B6 fix; the **single** dense backbone implementation. |
| `segmentation/models/backbone.py` | Thin re-export of the above (was a byte-identical copy carrying the same bug). |
| `detection/data/coco_dataset.py` | `idx_to_coco_id`, `include_empty`, `official_coco_gt()`, fixed `get_coco_api()`, `orig_size` in targets. |
| `detection/tools/train.py` | Detection trainer (rewritten): registry, append-safe log, atomic per-epoch ckpt, `--resume auto`, exact val sharding, COCO scoring, artifacts. |
| `segmentation/tools/train.py` | Segmentation trainer (rewritten): B4 metric, MS-TTA, per-class CSV, probe dumps, same hygiene. |
| `tools/dense_runtime.py` | Shared plumbing: config resolution, backbone resolution with the completion gate, seeding/RNG, atomic writes, log sanitisation, completion check, exact sharding, buffer sync. |
| `tools/dense_done.py` | Completion + backbone pre-check; used by the sbatch files to skip staging for surplus chain jobs. |
| `tools/check_smoke.py` | Phase-B smoke verdict: reads logs **and** artifacts, ~60 PASS/FAIL checks, projects measured throughput onto the production schedule. |
| `tools/export_f7_crops.py` | HPC-side: exports the 2 COCO crops straight out of the zips (login node, seconds). |
| `analysis/build_dense_tables.py` | T4/T5 + appendices. |
| `analysis/build_gate2_note.py` | The generated note; **the verdict rule lives here**. |
| `analysis/collect_F7.py` | Freezes the F7 COCO selection. |
| `plotting/plot_F7.py` | Renders F7; draws an explicit MISSING placeholder if the crops are absent. |
| `scripts/gen_dense_jobs.py` | Generates all 8 sbatch chains + 2 smokes deterministically. |
| `tests/test_task09_dense.py` (85) | Phase A: B5/B6/B4, COCO scoring, matrix contracts, launcher/port contracts, runtime helpers. |
| `tests/test_task09_phasec.py` (30) | Phase C: table↔artifact tracing, the verdict rule, note byte-identity, F7, the s2 seed, path hygiene. |

---

## 7. WHAT WE DID **NOT** DO — read before quoting any number

1. **No seed repeats.** One run per cell. `det_vitb_{baseline,saga}_s2` are
   prepared, tested and wired (ports 29856/29857) but were **not submitted**
   — the human's decision on 2026-09-13. They are marked
   `submitted: false` in the matrix so they cannot appear in a table as
   MISSING. Submitting them later is two commands and no code change.
2. **The registers rows are not backbone-matched.** The seeded e2r ViT-B
   registers run had not finished at launch, so both registers runs use the
   **legacy, unseeded** ViT-B registers checkpoint
   (`backbone_source: fallback`, sha `82365a3111…`), while baseline and saga
   use their seeded e2r s1 backbones. A registers-vs-baseline delta
   therefore mixes a backbone change with a variant change. Do not report it
   as a variant effect.
3. **The segmentation registers run peaked at epoch 29**, not 79, and its
   val mIoU declined for the remaining 50 epochs. Val-selection worked as
   designed, but that column is not read at the same point in training as
   the other two.
4. **No warmup.** `warmup_epochs` is declared in both committed base configs
   but was never implemented by the committed trainers. It stays
   unimplemented — adding it would change the schedule, i.e. the numbers.
5. **"1× schedule" is 25 epochs, not the COCO-convention 12**, because the
   brief said to keep hyperparameters as committed and
   `detection/configs/base.yaml` says 25. Flagged at the time; never
   overruled.
6. **No test-time-registers dense rows** (the brief said note, don't build).
   That became TASK 10.
7. **No ViT-S dense runs, no other architectures, no longer schedules, no
   COCO test-dev.**
8. **MS is TTA, not multi-scale testing** (§4.2).

---

## 8. HOW THE RUNS WERE PRODUCED

**Schedule** — resolved from the committed base configs, never restated in
the matrix (a test diffs the resolution against those files):

- detection: 25 epochs, global batch 8, backbone lr 1e-5 / head lr 1e-4,
  wd 0.05, grad clip 1.0, backbone frozen 1 epoch, 800/1333, fp16 AMP,
  eval every 5 epochs.
- segmentation: 80 epochs, global batch 16, same lrs/wd/clip, backbone
  frozen 2 epochs, 512 crop, scale [0.5, 2.0], eval every 10 epochs.

**Backbones** — `e2r_vitb_mixup_baseline_s1` (`316211c719c4…`) and
`e2r_vitb_mixup_saga_s1` (`a4e0e0ccd3b4…`), sha256 pinned in the matrix and
asserted at load; registers via the legacy fallback (§7.2).

**Execution** — 4× A100 per job, `--dependency=singleton,afterany` chains
(detection 4×24 h, segmentation 2×24 h). Data staged to node-local
`/scratch/iwi5359h/<dataset>_$SLURM_JOB_ID` and released by an `EXIT` trap.
Checkpoints to `/home/woody/iwi5/iwi5359h/SAGA/dense_ckpt` (hpc was over its
soft quota at 101G/100G).

**Measured cost** — all six finished inside their first chain job, **no
resumes**: detection 19.0–20.9 h of training wall per run (38–42 img/s),
segmentation 2.6–2.9 h (157–173 img/s). The pre-launch projection from the
smoke was conservative, which is why the chain lengths held.

**Smokes** — jobs 4200497/4200498, both COMPLETED 0:0, verified by
`tools/check_smoke.py`: 44/45 and 52/55 checks, the two genuine failures
being the checker's own over-broad `TaskProlog` pattern (fixed) and the
ADE20K class names (also fixed — the file is `objectInfo150.txt`, not
`.csv`).

---

## 9. HOW TO REPRODUCE / REGENERATE

Local, PowerShell, from `SAGA_Code\SAGA` — every artifact is regenerated
from committed inputs, and the committed copies reproduce byte-for-byte
(the PDF differs only in matplotlib's embedded timestamp):

```powershell
..\.venv\Scripts\python.exe analysis\build_dense_tables.py
..\.venv\Scripts\python.exe analysis\build_gate2_note.py
..\.venv\Scripts\python.exe plotting\plot_F7.py
..\.venv\Scripts\python.exe -m pytest -q tests\test_task09_dense.py tests\test_task09_phasec.py
```

To submit the second detection seed later (HPC, bash), after merging to
`main` and pulling there:

```bash
bash scripts/submit_det_vitb_baseline_s2.sh
bash scripts/submit_det_vitb_saga_s2.sh
```

Then flip `submitted: false` → `true` for those two runs in
`configs/dense_matrix.yaml` and rebuild the tables; they will join T4
automatically, and the note's uncertainty section can be replaced by a real
paired comparison.

---

## 10. GOTCHAS FOR WHOEVER PICKS THIS UP

- **Units differ between files by design.** The JSONs report mIoU /
  pixel_acc / mean_acc as **percent** and `iou_per_class` as **fractions**;
  `per_class_iou.csv`'s column is named `iou_frac` for that reason, and the
  table also carries `*_iou_pts`. Check the `units` block before comparing.
- **The note is generated.** Never hand-edit `gate2_report.md`; a test
  re-renders it from the committed tables and requires byte-identity.
- **Pair on sha256, not on filenames.** Every artifact records
  `backbone_run` / `backbone_sha256`; a run whose recorded sha does not
  match the matrix candidate it names yields MISSING, never a number.
- **`submitted: false` ≠ MISSING.** A run marked so was never started and is
  excluded from tables; a run that *is* expected and absent still yields
  MISSING. Both behaviours are pinned by tests.
- **Ports are pinned per run** in the matrix. Adding a run must not
  renumber an executed one — that is why.
- **The legacy evaluators refuse to run**, and the legacy `e3`/`e4`
  launchers are marked SUPERSEDED. Do not resurrect them; their metric
  definitions are the broken ones.
- **AP_S at the best-AP epoch.** Selection is on AP alone (single
  criterion). SAGA's AP_S happened to dip at epoch 24 (17.866 → 17.616);
  had selection been on AP_S the gap would read −0.28 rather than −0.844.
  The single-criterion protocol is the right one, but the sensitivity is
  worth knowing.

---

## 11. COMMIT HISTORY

| Commit | Phase | Content |
|---|---|---|
| `528fdae` | A | Dense fixes (B5/B6/B4 + the scoring bugs) + launchers + 62 tests |
| `bc833e8` | B | Smoke verdict tool; data confirmed; quota finding |
| `1dac02b` | B | Smoke PASSED; five review findings fixed |
| `8f5ed4b` | B | Chain lengths confirmed on measured throughput; names file located |
| `1058d41` | B | `EXIT`-trap staging release; `ckpt_root` → woody |
| `e213c0d` | C | T4/T5 tables, generated note, F7 draft, second detection seed |
| `5cd3a95` | C | F7 COCO crops (HPC export) |
| `8fd369e` | C | F7 COCO half rendered; unsubmitted seed excluded from the tables |

All are on `main`. Other tasks have since touched TASK-09 files, and the
suites still pass on current `main` (115 tests), which is the check that
matters:

- **during** TASK 09, between its Phase B and Phase C: `da4e631` (TASK-10,
  `set -u` must follow the env sourcing) and `c707021` (TASK-10, tolerate
  the conda module rename) — repo-wide fixes that also reached the dense
  launchers;
- **after** TASK 09's last commit: `94a85cc` (TASK-12, matched-init
  ablation) modified `saga/vit.py`, the file carrying TASK-09's B6 guard.
  That guard is still enforced — `test_sagavit_refuses_register_models` and
  the pos-embed tripwire tests pass — but it is the one shared file where a
  future change could silently reopen B6, so run
  `tests/test_task09_dense.py` after editing it.

---

## 12. OPEN QUESTIONS FOR THE HUMAN

1. **Does Gate 2's PARTIAL stand as the reported result**, given it rests on
   one run per cell? The cheapest thing that would change the evidence is
   the already-prepared second detection seed (~20 h per run).
2. **How should the registers column be reported**, given its backbone is
   not matched (§7.2)? Options: report with the caveat, drop it, or rerun
   once the seeded e2r registers backbone exists.
3. **Is the AP_S story "SAGA is worse on small objects" or "the AP_S
   aggregate is tail-dominated"?** §4.5 has the evidence for the second
   reading; the paper text should not silently pick the first.
