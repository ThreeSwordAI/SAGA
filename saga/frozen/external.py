"""
saga/frozen/external.py
=======================
The external public checkpoints — TASK A / I6 A8.

Nine models this project did not train, measured with the SAME code that
measures its own cohort: `saga/frozen/stages.py::capture_stages` for the
tokens, `saga/frozen/norms.py` for the norms and indicators,
`saga/frozen/prevalence.py` for the map, `analysis/address_analysis.py` for
the ring profile and the two references. Nothing here re-implements a
statistic.

WHAT IS DELIBERATELY ABSENT. No gate. No `SAGAViT` wrapper — it refuses
`num_prefix_tokens != 1` by design (`saga/vit.py`) and one of these models
has five. No fine-tuning, no head, no training of any kind. A public
checkpoint is loaded, put in `eval()`, and read.

THE TWO ANALOG STAGES
---------------------
`ext_s11_out` is the output of `blocks[-2]` and `ext_hist` the output of the
last block before the final norm — the same two block outputs `s11_out` and
`s12_pre_norm` name on our own models, captured by the same hooks. They
carry their own names because the models are not comparable in the way a
shared name would imply: different training data, different augmentation,
different patch size, and for DINOv2 a different native resolution (518,
run here at 224 with the position embedding interpolated). I6 tests whether
the PATTERN is present in a public model, not whether a number matches ours.

GEOMETRY IS READ FROM THE MODEL AND CROSS-CHECKED, NEVER ASSUMED
----------------------------------------------------------------
The prefix count comes from `infer_num_prefix_tokens` and the grid from the
model's own `patch_embed.grid_size`; both are then compared against the
registry and a mismatch is a hard failure, not a warning. A register model
read with a hard-coded prefix of 1 would put four register tokens into the
patch grid and every ring number after that would be wrong — quietly, and
the map would still look like a map.

No training, no optimizer, no probe fitting, no selection by any loss.
"""

import json
import os
from pathlib import Path

import numpy as np

from saga.frozen.stages import EXTERNAL_STAGES, resolve_stage
from saga.metrics import infer_num_prefix_tokens

MISSING = "MISSING"
PENDING = "PENDING"
UNAVAILABLE = "UNAVAILABLE"

#: The two analog stages, in the order tables report them.
EXT_STAGES = ("ext_s11_out", "ext_hist")

#: Registry fields every model entry must carry.
REQUIRED_FIELDS = ("model_id", "timm_name", "hub_id", "patch_size",
                   "grid_side", "n_prefix", "depth", "mixing", "group",
                   "citation")

#: The declared prediction groups. A tenth group would silently score as
#: "not applicable"; refuse it instead.
GROUPS = ("supervised_mixing", "no_mixing", "registers_exploratory")


class ExternalError(ValueError):
    """An external checkpoint that cannot be registered, built or read."""


# ─────────────────────────────────────────────────────────────────────────────
# The registry
# ─────────────────────────────────────────────────────────────────────────────

def load_registry(path="configs/frozen/I6_models.yaml") -> dict:
    """The committed registry, validated on load.

    The prediction is required and non-empty: this file exists to hold a
    claim made before the weights were fetched, and a registry without one
    is not the artifact I6 needs.
    """
    import yaml

    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("work_package", "prediction", "models", "input_size",
                "hf_home", "thresholds_cal", "threshold_split",
                "reporting_split"):
        if not doc.get(key):
            raise ExternalError(f"{path}: missing or empty key {key!r}")
    if len(doc["prediction"].strip()) < 100:
        raise ExternalError(
            f"{path}: `prediction` is {len(doc['prediction'])} characters. It "
            f"is the pre-registered claim, copied verbatim from "
            f"{doc.get('prediction_source')}, not a placeholder.")
    seen = set()
    for entry in doc["models"]:
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        if missing:
            raise ExternalError(
                f"{path}: model {entry.get('model_id')!r} is missing "
                f"{missing}")
        if entry["model_id"] in seen:
            raise ExternalError(
                f"{path}: duplicate model_id {entry['model_id']!r}")
        seen.add(entry["model_id"])
        if entry["group"] not in GROUPS:
            raise ExternalError(
                f"{path}: model {entry['model_id']!r} declares group "
                f"{entry['group']!r}; expected one of {GROUPS}")
        if str(entry["mixing"]) not in ("yes", "no"):
            raise ExternalError(
                f"{path}: model {entry['model_id']!r} declares mixing="
                f"{entry['mixing']!r}; expected the string 'yes' or 'no' "
                f"(bare yes/no is a YAML boolean)")
        side, patch = int(entry["grid_side"]), int(entry["patch_size"])
        if side * patch != int(doc["input_size"]):
            raise ExternalError(
                f"{path}: model {entry['model_id']!r} declares grid_side="
                f"{side} and patch_size={patch}, which is {side * patch} "
                f"pixels, not the declared input_size {doc['input_size']}")
    return doc


def model_entry(registry: dict, model_id: str) -> dict:
    for entry in registry["models"]:
        if entry["model_id"] == model_id:
            return entry
    raise ExternalError(
        f"{model_id!r} is not in the registry; it holds "
        f"{[m['model_id'] for m in registry['models']]}")


def timm_version() -> str:
    import timm
    return str(timm.__version__)


def resolves_in_timm(timm_name: str) -> bool:
    """True when the installed timm has BOTH the architecture and the
    pretrained tag. A name that resolves as an architecture but not as a tag
    would silently build random weights."""
    import timm

    arch = str(timm_name).split(".")[0]
    if not timm.is_model(arch):
        return False
    return str(timm_name) in set(timm.list_pretrained(arch))


def availability(registry: dict) -> dict:
    """`{model_id: "available" | "UNAVAILABLE (timm x.y.z)"}`.

    A name that does not resolve is RECORDED, never substituted with a
    different checkpoint (TASK A §12). The timm version travels with the
    verdict because that is what the verdict depends on.
    """
    version = timm_version()
    out = {}
    for entry in registry["models"]:
        out[entry["model_id"]] = (
            "available" if resolves_in_timm(entry["timm_name"])
            else f"{UNAVAILABLE} (timm {version})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Building a model, and reading its geometry
# ─────────────────────────────────────────────────────────────────────────────

def model_geometry(model) -> dict:
    """Prefix count, grid and depth READ FROM THE MODEL.

    `patch_embed.grid_size` is the model's own statement about its token
    grid, which is what the ring geometry needs; the prefix count comes from
    `infer_num_prefix_tokens`, never from a constant.
    """
    try:
        grid = tuple(int(v) for v in model.patch_embed.grid_size)
    except AttributeError as exc:
        raise ExternalError(
            f"{type(model).__name__} exposes no patch_embed.grid_size, so its "
            f"token grid cannot be read; this module refuses to guess at "
            f"one") from exc
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ExternalError(
            f"{type(model).__name__} exposes no non-empty .blocks")
    return {"grid": grid, "grid_side": grid[0], "n_patches": grid[0] * grid[1],
            "depth": int(len(blocks)),
            "n_prefix": int(infer_num_prefix_tokens(model))}


def check_geometry(geom: dict, entry: dict, *, allow_non_square=False,
                   allow_other_depth=False) -> None:
    """Fail LOUDLY when the model and the registry disagree.

    Every one of these would otherwise produce a plausible-looking map:
    a non-square grid has no ring geometry and no torus-roll null; a wrong
    prefix count slides the whole patch grid; a different depth means
    `blocks[-2]` is not the layer the registry says it is.
    """
    side_rows, side_cols = geom["grid"]
    if side_rows != side_cols and not allow_non_square:
        raise ExternalError(
            f"{entry['model_id']}: token grid is {geom['grid']}, which is not "
            f"square. The ring geometry and the dihedral x torus-roll "
            f"permutation null are both defined on a square grid only.")
    if int(entry["grid_side"]) != side_rows:
        raise ExternalError(
            f"{entry['model_id']}: the registry declares grid_side="
            f"{entry['grid_side']} but the model reports {side_rows}. One of "
            f"the two is wrong and nothing downstream could tell which.")
    if int(entry["n_prefix"]) != geom["n_prefix"]:
        raise ExternalError(
            f"{entry['model_id']}: the registry declares n_prefix="
            f"{entry['n_prefix']} but the model reports {geom['n_prefix']}. A "
            f"wrong prefix count puts non-patch tokens into the patch grid "
            f"and every ring number after it is wrong.")
    if geom["depth"] != 12 and not (allow_other_depth
                                    or int(entry["depth"]) == geom["depth"]):
        raise ExternalError(
            f"{entry['model_id']}: depth is {geom['depth']}, not 12, and the "
            f"registry does not declare it. `ext_s11_out` is blocks[-2]; on a "
            f"model of another depth that is a different layer than the one "
            f"the registry describes.")
    if int(entry["depth"]) != geom["depth"]:
        raise ExternalError(
            f"{entry['model_id']}: the registry declares depth="
            f"{entry['depth']} but the model has {geom['depth']} blocks")


def build_external(model_id: str, registry: dict, *, device="cpu",
                   pretrained=True, allow_non_square=False,
                   allow_other_depth=False):
    """`(model, info)` for one registered external checkpoint.

    Built at the registry's `input_size` — 224 for every entry — so the grid
    is 14x14 or 16x16 and the rings are defined. For the DINOv2 models,
    whose pretrained size is 518, timm interpolates the position embedding;
    `info["native_input_size"]` records that this happened and the note says
    so. No gate, no SAGAViT wrapper, no head, no fine-tuning.
    """
    import timm

    entry = model_entry(registry, model_id)
    if not resolves_in_timm(entry["timm_name"]):
        raise ExternalError(
            f"{model_id}: {entry['timm_name']!r} does not resolve in timm "
            f"{timm_version()}. It is recorded {UNAVAILABLE}; TASK A §12 "
            f"forbids substituting a different checkpoint for it.")
    size = int(registry["input_size"])
    model = timm.create_model(entry["timm_name"], pretrained=bool(pretrained),
                              img_size=size).eval().to(device)
    geom = model_geometry(model)
    check_geometry(geom, entry, allow_non_square=allow_non_square,
                   allow_other_depth=allow_other_depth)

    cfg = timm.data.resolve_data_config({}, model=model)
    info = {
        "model_id": model_id, "timm_name": entry["timm_name"],
        "hub_id": entry["hub_id"], "timm_version": timm_version(),
        "group": entry["group"], "mixing": str(entry["mixing"]),
        "citation": entry["citation"],
        "input_size": size,
        "native_input_size": int(entry.get("native_input_size", size)),
        "resolution_deviation": bool(
            int(entry.get("native_input_size", size)) != size),
        "patch_size": int(entry["patch_size"]),
        "weight_sha256": entry.get("weight_sha256", PENDING),
        "pretrained": bool(pretrained),
        # timm's OWN normalization for this tag, recorded per model
        "norm_mean": [float(v) for v in cfg["mean"]],
        "norm_std": [float(v) for v in cfg["std"]],
        "crop_pct": float(cfg["crop_pct"]),
        "interpolation": str(cfg["interpolation"]),
        **{k: geom[k] for k in ("grid_side", "n_patches", "depth",
                                "n_prefix")},
    }
    return model, info


def external_transform(model):
    """timm's own validation transform for THIS checkpoint.

    Read from the model's pretrained config, never from our cohort's
    `build_val_transform`: the four recipes in this registry disagree about
    mean, std and crop_pct, and feeding a CLIP checkpoint ImageNet
    statistics would measure the mismatch rather than the model.
    """
    import timm

    cfg = timm.data.resolve_data_config({}, model=model)
    return timm.data.create_transform(**cfg, is_training=False)


def ext_stage_alias(stage: str) -> str:
    """The internal stage an analog name resolves to, validated."""
    if stage not in EXTERNAL_STAGES:
        raise ExternalError(
            f"{stage!r} is not an external analog stage; expected one of "
            f"{tuple(EXTERNAL_STAGES)}")
    return resolve_stage(stage)


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds — calibrated on one split, applied to another (and it says so)
# ─────────────────────────────────────────────────────────────────────────────

def thresholds_doc(registry: dict, *, tau_cal: dict, sources: dict,
                   calibration_split: str, calibration_sha: str,
                   reporting_split: str, reporting_sha: str,
                   git_sha_value: str) -> dict:
    """The I6 thresholds document.

    It records BOTH shas. I1 and I2 calibrate and report on the same split,
    and `saga.frozen.diag.load_thresholds_cal` enforces that by refusing a
    file whose split sha differs from the run's. I6 deliberately does not:
    a public checkpoint has no designated baseline to borrow a threshold
    from, so it is calibrated on `calibration` and applied to `evaluation`,
    which keeps the threshold off the images it is reported on. Writing both
    shas is what stops this file from ever being read as an I1/I2-style
    same-split tau.
    """
    return {
        "work_package": registry["work_package"],
        "definition": (
            "per MODEL and per analog STAGE: tau_cal = lower median over "
            "images of (median(v_i) + k*MAD(v_i)) on the PATCH-row L2 norms "
            "at that stage, computed on the CALIBRATION split and APPLIED to "
            "the reporting split. The medians are LOWER medians via "
            "tools/compute_fixed_thr.per_image_mad_thresholds, the function "
            "that produced every canon value. Unlike I1/I2, the calibration "
            "and reporting splits DIFFER: an external checkpoint is its own "
            "reference and has no designated baseline to borrow a threshold "
            "from, so calibrating on held-out images keeps the threshold and "
            "the reported map off the same pixels."),
        "k": float(registry.get("mad_k", 5.0)),
        "calibration_split_name": calibration_split,
        "calibration_split_sha256": calibration_sha,
        "applied_to_split_name": reporting_split,
        "applied_to_split_sha256": reporting_sha,
        "stages": list(EXT_STAGES),
        "input_size": int(registry["input_size"]),
        "precision": registry.get("precision", "fp32"),
        "timm_version": timm_version(),
        "git_sha": git_sha_value,
        "generated_by": "tools/frozen_I6_maps.py",
        "tau_cal": tau_cal, "sources": sources,
    }


def load_thresholds(path, *, calibration_sha=None) -> dict:
    """The I6 thresholds, with the CALIBRATION sha checked.

    The reporting sha is deliberately NOT checked against it — see
    `thresholds_doc`. It is recorded in the file so a reader can see which
    split each half came from.
    """
    path = Path(path)
    if not path.exists():
        raise ExternalError(
            f"{path} not found — run tools/frozen_I6_maps.py --thresholds "
            f"for the calibration split before the reporting pass")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if calibration_sha is not None and \
            doc.get("calibration_split_sha256") != calibration_sha:
        raise ExternalError(
            f"{path} was calibrated on split sha "
            f"{doc.get('calibration_split_sha256')} but this run's "
            f"calibration split hashes to {calibration_sha}")
    return doc


def tau_for(doc: dict, model_id: str, stage: str):
    value = (doc.get("tau_cal", {}).get(model_id, {}) or {}).get(stage)
    return float(value) if isinstance(value, (int, float)) else MISSING


# ─────────────────────────────────────────────────────────────────────────────
# Scoring the pre-registered prediction — by code, never by hand
# ─────────────────────────────────────────────────────────────────────────────

def ring_1_peak(ring_profile) -> bool:
    """True when the border profile attains its maximum at ring 1."""
    prof = np.asarray(ring_profile, dtype=np.float64)
    if prof.size < 2:
        raise ExternalError(
            f"a {prof.size}-ring profile cannot have a ring-1 peak")
    return bool(int(np.argmax(prof)) == 1)


def prediction_outcome(group: str, *, has_ring_1_peak: bool,
                       gini_excess) -> str:
    """`met` / `not met` / `not applicable`, from the REGISTERED rule.

    The rule is in `configs/frozen/I6_models.yaml` beside the prediction it
    scores, and this function is the only thing that writes the verdict into
    a table. Nobody types it.
    """
    if group not in GROUPS:
        raise ExternalError(f"unknown prediction group {group!r}")
    if group == "registers_exploratory":
        return "not applicable"
    if group == "supervised_mixing":
        if not isinstance(gini_excess, (int, float)) or \
                not np.isfinite(gini_excess):
            return MISSING
        return "met" if (has_ring_1_peak and float(gini_excess) > 0) \
            else "not met"
    return "met" if not has_ring_1_peak else "not met"


# ─────────────────────────────────────────────────────────────────────────────
# The weight cache
# ─────────────────────────────────────────────────────────────────────────────

#: Weight files a timm hub repo may carry, tried in this order. Modern timm
#: publishes safetensors; older tags are still `pytorch_model.bin`.
WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")


def cached_weight_path(entry: dict, *, local_files_only=True):
    """The local path of this model's weight file, or None.

    `local_files_only=True` on a compute node, which has no outbound network
    on this cluster: the file must already be in the woody cache that
    `tools/frozen_I6_download.py` filled on the login node.
    """
    from huggingface_hub import hf_hub_download

    for name in WEIGHT_FILENAMES:
        try:
            return Path(hf_hub_download(repo_id=entry["hub_id"],
                                        filename=name,
                                        local_files_only=local_files_only))
        except Exception:                             # noqa: BLE001
            continue
    return None


def file_sha256(path, chunk=1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_weight_sha(entry: dict) -> str:
    """Check the cached weights against the registry BEFORE they are loaded.

    The registry's sha was written on the login node by the download tool
    and committed. Checking it here is what makes an I6 number traceable to
    an identified set of bytes, the same way `verify_checkpoint` does for
    our own cohort.
    """
    expected = str(entry.get("weight_sha256", PENDING))
    if expected in (PENDING, "", "None"):
        raise ExternalError(
            f"{entry['model_id']}: the registry records weight_sha256="
            f"{expected!r}. Run tools/frozen_I6_download.py on the login node "
            f"and commit the registry before any job reads these weights.")
    path = cached_weight_path(entry)
    if path is None:
        from huggingface_hub import constants as hf_constants
        raise ExternalError(
            f"{entry['model_id']}: no cached weight file for "
            f"{entry['hub_id']!r}.\n"
            f"  huggingface_hub looked in: {hf_constants.HF_HUB_CACHE}\n"
            f"  HF_HOME={os.environ.get('HF_HOME')!r} "
            f"SAGA_HF_HOME={os.environ.get('SAGA_HF_HOME')!r}\n"
            f"Compute nodes have no network, so the file must already be "
            f"there. If the path above is not the registry's hf_home, the "
            f"download tool wrote to a different cache and the weights need "
            f"moving — which is what happened on 2026-09-16, when "
            f"huggingface_hub had frozen its cache constant before HF_HOME "
            f"was set.")
    got = file_sha256(path)
    if got != expected:
        raise ExternalError(
            f"{entry['model_id']}: cached weights hash to {got} but the "
            f"registry records {expected}. Refusing to measure a checkpoint "
            f"that is not the one that was registered.")
    return got


def set_cache_env(registry: dict, *, create: bool = True) -> str:
    """Point HF_HOME and TORCH_HOME at the registry's woody path.

    MUST run before timm is imported anywhere that downloads. $HOME is the
    small quota on this cluster (How to Run.md §3) and a hub cache there
    fills it. `SAGA_HF_HOME` overrides the registry without editing it.

    `create=False` sets the variables without touching the filesystem — for
    a dry run, and for any machine that is not the cluster, where the
    registry's absolute POSIX path would otherwise be created relative to
    the current drive.
    """
    root = os.environ.get("SAGA_HF_HOME") or registry["hf_home"]
    if create:
        Path(root).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["TORCH_HOME"] = str(root)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(root) / "hub"))
    return str(root)


__all__ = [
    "ExternalError", "EXT_STAGES", "GROUPS", "MISSING", "PENDING",
    "UNAVAILABLE", "load_registry", "model_entry", "timm_version",
    "resolves_in_timm", "availability", "model_geometry", "check_geometry",
    "build_external", "external_transform", "ext_stage_alias",
    "thresholds_doc", "load_thresholds", "tau_for", "ring_1_peak",
    "prediction_outcome", "set_cache_env", "WEIGHT_FILENAMES",
    "cached_weight_path", "file_sha256", "verify_weight_sha",
]
