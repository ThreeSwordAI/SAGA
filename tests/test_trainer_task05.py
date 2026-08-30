"""TASK-05 A2: trainer equivalence, resume, atomic-checkpoint, contract."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "classification" / "tools"))
import train as trainer  # classification/tools/train.py

from saga.vit import build_saga_vit
from tools.build_diag_split import build_split

IMG = 96          # tiny grid (6x6 patches) keeps CPU runtime small
N_CLASSES = 4


# ── fixtures ─────────────────────────────────────────────────────────────────

def make_fake_imagenet(root: Path, n_per_class=2):
    rng = np.random.RandomState(0)
    for split in ("train", "val"):
        for c in range(N_CLASSES):
            d = root / split / f"n{c:08d}"
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                arr = rng.randint(0, 255, (IMG, IMG, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"im{i}_n{c:08d}.JPEG")
    return root


def make_matrix(tmp_path: Path, data_root: Path) -> Path:
    split = build_split(data_root, n_per_class=2, seed=0)
    split_file = tmp_path / "diag_split.json"
    split_file.write_text(json.dumps(split))

    overrides = {
        "model": {"img_size": IMG, "num_classes": N_CLASSES},
        "train": {"epochs": 3, "batch_size": 2, "warmup_epochs": 1,
                  "amp": False},
        "data": {"input_size": IMG, "num_workers": 0, "pin_memory": False},
        "augmentation": {"rand_aug": False, "random_erase_prob": 0.0},
        "logging": {"log_freq": 1000},
    }
    matrix = {
        "config_dir": (REPO / "classification" / "configs").as_posix(),
        "defaults": {"diag_freq": 2, "diag_split": split_file.as_posix(),
                     "diag_n_effrank": 8, "grad_log_every": 2,
                     "grad_log_epochs": 31},
        "runs": {
            "tiny_saga": {"arch": "vit_tiny_patch16_224", "recipe": "nomix",
                          "variant": "saga", "seed": 7,
                          "log_grad_phi": True, "overrides": overrides},
            "tiny_baseline": {"arch": "vit_tiny_patch16_224",
                              "recipe": "nomix", "variant": "baseline",
                              "seed": 7, "overrides": overrides},
        },
    }
    p = tmp_path / "matrix.yaml"
    p.write_text(yaml.safe_dump(matrix))
    return p


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("t05")
    data_root = make_fake_imagenet(tmp / "data")
    matrix = make_matrix(tmp, data_root)
    return {"tmp": tmp, "data": data_root, "matrix": matrix}


# ── T-eq: construction equivalence with the legacy path ─────────────────────

@pytest.mark.parametrize("variant,gate", [("tiny_baseline", False),
                                          ("tiny_saga", True)])
def test_eq_construction_and_one_step(env, variant, gate):
    cfg = trainer.resolve_run_config(env["matrix"], variant)

    trainer.set_seed(cfg["seed"])
    new = trainer.build_model(cfg)
    trainer.apply_knobs(new, cfg["knobs"])          # defaults: no-op

    trainer.set_seed(cfg["seed"])
    legacy = build_saga_vit(arch="vit_tiny_patch16_224", gate=gate,
                            img_size=IMG, patch_size=16,
                            num_classes=N_CLASSES, pretrained=False)

    sd_new, sd_old = new.state_dict(), legacy.state_dict()
    assert sd_new.keys() == sd_old.keys()
    for k in sd_new:
        assert torch.equal(sd_new[k], sd_old[k]), k

    # seeded 1-step forward+backward: identical loss and gradients
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, IMG, IMG, generator=g)
    y = torch.tensor([0, 1])
    losses, grads = [], []
    for model in (new, legacy):
        model.train()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        losses.append(loss.item())
        grads.append(model.patch_embed.proj.weight.grad.clone())
    assert abs(losses[0] - losses[1]) < 1e-6
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_knob_defaults_are_legacy():
    knobs = {"gate_init_logit": 0.0, "wd_phi_zero": False,
             "drop_path_rate": 0.0}
    model = build_saga_vit("vit_tiny_patch16_224", gate=True, img_size=IMG,
                           num_classes=N_CLASSES)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    trainer.apply_knobs(model, knobs)
    for k, v in model.state_dict().items():
        assert torch.equal(before[k], v)
    # default param group = plain model.parameters()
    params = trainer.optimizer_parameters(model, knobs)
    assert not isinstance(params, list)            # generator, single group

    # non-default knobs do act
    trainer.apply_knobs(model, {"gate_init_logit": -2.0})
    assert torch.allclose(model.blocks[0].attn.gate.phi,
                          torch.full_like(model.blocks[0].attn.gate.phi, -2.0))
    groups = trainer.optimizer_parameters(model, {"wd_phi_zero": True})
    assert isinstance(groups, list) and groups[1]["weight_decay"] == 0.0
    n_phi = len(groups[1]["params"])
    assert n_phi == len(model.blocks)


# ── T-resume + T-contract (one 3-epoch run, kill, resume to 5) ───────────────

@pytest.fixture(scope="module")
def finished_run(env):
    out_root = env["tmp"] / "runs"
    run_dir = trainer.run_training(env["matrix"], "tiny_saga",
                                   env["data"], out_root=out_root,
                                   resume="auto", max_epochs=3,
                                   device_str="cpu")
    # simulate the crash window: a log row for an epoch whose checkpoint
    # never landed (log-then-ckpt order) must be dropped on resume
    trainer.append_log_row(run_dir / "log.csv", {
        "epoch": 3, "lr": 0.1, "train_loss": 9.9, "val_top1_full": 0.0,
        "val_top5_full": 0.0, "val_loss": 9.9, "img_per_sec": 1.0,
        "wall_time": 1.0})

    run_dir2 = trainer.run_training(env["matrix"], "tiny_saga",
                                    env["data"], out_root=out_root,
                                    resume="auto", max_epochs=5,
                                    device_str="cpu")
    assert run_dir2 == run_dir
    return run_dir


def test_resume_log_contiguous(finished_run):
    rows = list(csv.DictReader(open(finished_run / "log.csv", newline="")))
    assert [int(r["epoch"]) for r in rows] == [0, 1, 2, 3, 4]
    # the poisoned duplicate epoch-3 row was dropped, the real one written
    assert float(rows[3]["train_loss"]) != 9.9
    for r in rows:
        assert all(r[k] != "" for k in trainer.LOG_FIELDS)


def test_resume_checkpoint_state(finished_run):
    ckpt = torch.load(finished_run / "ckpt" / "last.pth",
                      map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 4 and ckpt["sampler_epoch"] == 4
    for key in ("model", "optimizer", "scheduler", "scaler", "best_top1",
                "rng_states"):
        assert key in ckpt, key
    st = ckpt["rng_states"][0]
    assert {"torch", "numpy", "python"} <= set(st)

    # strict round-trip into a fresh legacy-path model
    model = build_saga_vit("vit_tiny_patch16_224", gate=True, img_size=IMG,
                           num_classes=N_CLASSES)
    model.load_state_dict(ckpt["model"], strict=True)

    best = torch.load(finished_run / "ckpt" / "best.pth",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"], strict=True)


def test_contract_files(finished_run):
    meta = json.loads((finished_run / "meta.json").read_text())
    assert meta["run_id"] == "tiny_saga" and meta["seed"] == 7
    assert meta["knobs"] == {"gate_init_logit": 0.0, "wd_phi_zero": False,
                             "drop_path_rate": 0.0}
    assert meta["resumes"], "resume must be recorded in meta"
    assert meta["end_time"] is not None
    assert (finished_run / "config.resolved.yaml").exists()

    # phi dumped EVERY epoch, including after resume
    for e in range(5):
        z = np.load(finished_run / "gates" / f"phi_e{e:03d}.npz")
        assert z["phi"].shape == (12, 3, 36)       # vit_tiny: L12 H3 N36

    # diag every diag_freq=2 epochs: (epoch+1) % 2 == 0 -> epochs 1, 3
    for e in (1, 3):
        d = json.loads(
            (finished_run / "diag" / f"diag_e{e:03d}.json").read_text())
        assert d["epoch"] == e and d["n_images"] == 8
        assert d["cls_attn_share"] is None         # --no-attn mode
        for k in ("sink_mad_k5", "oversmooth_pairwise",
                  "oversmooth_pairwise_nosink", "eff_rank"):
            assert k in d
    assert not (finished_run / "diag" / "diag_e000.json").exists()

    # grad-phi instrumentation (log_grad_phi: true on this run)
    grows = list(csv.DictReader(
        open(finished_run / "grads" / "grad_phi.csv", newline="")))
    assert grows and all(np.isfinite(float(r["grad_phi_norm"]))
                         for r in grows)
    assert {int(r["layer"]) for r in grows} == set(range(12))


def test_baseline_run_has_no_gate_artifacts(env):
    out_root = env["tmp"] / "runs_base"
    run_dir = trainer.run_training(env["matrix"], "tiny_baseline",
                                   env["data"], out_root=out_root,
                                   resume="auto", max_epochs=1,
                                   device_str="cpu")
    assert not (run_dir / "gates").exists()
    assert not (run_dir / "grads").exists()
    assert (run_dir / "ckpt" / "last.pth").exists()


# ── T-atomic: kill during checkpoint write ───────────────────────────────────

def test_atomic_checkpoint_survives_kill(tmp_path, monkeypatch):
    path = tmp_path / "last.pth"
    trainer.atomic_torch_save({"epoch": 1, "payload": torch.ones(3)}, path)

    real_save = torch.save

    def dying_save(obj, f, *a, **kw):
        # write garbage bytes to the tmp target, then die mid-write
        with open(f, "wb") as fh:
            fh.write(b"\x00garbage")
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(torch, "save", dying_save)
    with pytest.raises(RuntimeError):
        trainer.atomic_torch_save({"epoch": 2}, path)
    monkeypatch.setattr(torch, "save", real_save)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1                      # previous file intact
    assert torch.equal(ckpt["payload"], torch.ones(3))


def test_resume_refuses_steps_per_epoch_drift(env, finished_run):
    ckpt_path = finished_run / "ckpt" / "last.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["steps_per_epoch"] == 4 and ckpt["total_epochs"] == 5
    ckpt["steps_per_epoch"] = 999          # simulate a bad staged dataset
    trainer.atomic_torch_save(ckpt, ckpt_path)
    try:
        with pytest.raises(RuntimeError, match="steps_per_epoch changed"):
            trainer.run_training(env["matrix"], "tiny_saga", env["data"],
                                 out_root=finished_run.parent,
                                 resume="auto", max_epochs=6,
                                 device_str="cpu")
    finally:                               # restore for other tests
        ckpt["steps_per_epoch"] = 4
        trainer.atomic_torch_save(ckpt, ckpt_path)


def test_fresh_start_truncates_leftover_log(env, tmp_path):
    # crash-before-first-ckpt: log.csv exists, no checkpoint -> the fresh
    # start must not append duplicate epoch rows onto the leftover
    out_root = tmp_path / "runs"
    run_dir = out_root / "tiny_baseline"
    run_dir.mkdir(parents=True)
    trainer.append_log_row(run_dir / "log.csv", {
        "epoch": 0, "lr": 1.0, "train_loss": 9.9, "val_top1_full": 0.0,
        "val_top5_full": 0.0, "val_loss": 9.9, "img_per_sec": 1.0,
        "wall_time": 1.0})
    trainer.run_training(env["matrix"], "tiny_baseline", env["data"],
                         out_root=out_root, resume="auto", max_epochs=1,
                         device_str="cpu")
    rows = list(csv.DictReader(open(run_dir / "log.csv", newline="")))
    assert [int(r["epoch"]) for r in rows] == [0]
    assert float(rows[0]["train_loss"]) != 9.9      # the leftover is gone
