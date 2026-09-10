# results/ — the results contract

Every experimental number in the paper is read from a file in this tree.
Nothing in here is ever edited by hand. Large binaries (checkpoints, raw
attention dumps, norm dumps) are git-ignored; small artifacts (JSON, CSV,
gate files `phi_e*.npz`, `figures_data/`) are committed.

```
results/
  runs/<run_id>/            # run_id = <exp>_<arch>_<recipe>_<variant>_s<seed>
    meta.json               # written by saga/run_registry.py
    config.resolved.yaml
    log.csv                 # one row per epoch (training tasks; later)
    ckpt/                   # *.pth — git-ignored
    gates/phi_e###.npz      # SAGA gate params per epoch (small; committed)
    diag/diag_e###.json     # periodic diagnostics (later tasks)
    eval/*.json             # full-val evaluations
    attn/*.npz              # probe attention dumps — git-ignored
  detection/<run_id>/       # [TASK-09] COCO detection runs (det_vitb_*_s1)
    meta.json  config.resolved.yaml  log.csv
    coco_eval_best.json     # six APs + ARs + per-category, at the best epoch
    detections_val.json     # raw COCO-format predictions at that epoch
    ckpt/                   # *.pth — git-ignored
    eval_shards/            # transient per-rank scratch — git-ignored
  segmentation/<run_id>/    # [TASK-09] ADE20K runs (seg_vitb_*_s1)
    meta.json  config.resolved.yaml  log.csv
    miou_ss.json  miou_ms.json       # single-scale / multi-scale+flip TTA;
                            # mIoU/pixel_acc/mean_acc are PERCENT,
                            # iou_per_class are FRACTIONS (each file
                            # carries a "units" block)
    per_class_iou.csv       # iou_frac column is a 0-1 fraction
    conf_matrix.npz
    preds_fixed20/          # <stem>_{pred,gt}.png + <stem>_img.jpg for the
                            # 20 committed probe images
    ckpt/                   # *.pth — git-ignored
  ttr/<base_run_id>/        # [TASK-10] test-time registers applied to a base
                            # cell's BASELINE checkpoint (no training at all)
    neurons.json            # register-neuron ranking + the criterion string
                            # and the canon tau it was detected with
    sweep.csv               # --n-neurons sweep; row n_neurons=0 is the
                            # UNPATCHED reference measured the same way
    validate.json           # A3 gate: per-n sink/top-1 + the PASS/FAIL
                            # verdict and the thresholds it was judged on
  runs/ttr_<base_run_id>/   # [TASK-10] the PATCHED model's standard
                            # artifacts, deliberately under runs/ so the
                            # existing collectors (which glob runs/*/) find
                            # them with no change:
    config.resolved.yaml    # carries the BASE cell's arch/variant/recipe,
                            # which is how apply_fixed_thr --version canon
                            # resolves the right tau for a TTR dir
    eval/*.json  diag/*.json  diag/*_addr.json
                            # tau is always the BASE cell's canon tau —
                            # never recalibrated on the patched model
  legacy/                   # everything derived from pre-fix checkpoints
    checkpoint_manifest.csv
    eval/   diag/   attn/
  diagsplit/val_diag_split.json   # frozen 10k diagnostic images (committed)
  ftsplit/<ds>_val_split.json     # frozen fine-tune val splits, carved from
                                  # the OFFICIAL train splits (committed)
  probe/probe_set.json            # frozen probe images (committed, later task)
  probe/ade20k_fixed20.json       # frozen 20 ADE20K val images (TASK-09;
                                  # committed BEFORE the first seg eval)
  tables/*.csv              # generated tables — the paper reads only these
  figures_data/*            # one file per paper figure
```
