#!/usr/bin/env python3
"""
tools/frozen_I6_download.py
===========================
TASK A / I6 A8 — fetch the nine external checkpoints on the LOGIN NODE and
record what was fetched.

    python tools/frozen_I6_download.py                 # download + write shas
    python tools/frozen_I6_download.py --dry-run       # resolve only, no network

Compute nodes on this cluster have no outbound network, so the weights are
pulled here, into the woody cache the registry names, and the job that reads
them runs offline.

THIS TOOL NEVER RUNS A FORWARD PASS, and not as a matter of discipline: it
never builds a model at all. It calls `huggingface_hub.hf_hub_download`,
which fetches a FILE. There is no `timm.create_model` in this module and no
tensor is ever passed through anything — `tests/test_I6_external.py` checks
that on the parsed AST. The reason is the one that makes I6 worth doing: the
prediction in `configs/frozen/I6_models.yaml` is registered BEFORE any
external model is run, and a download step that quietly evaluated a model
would put the first look at the outcome before the commit that registers the
claim.

WHAT IT WRITES. The sha256 of each downloaded weight file, back into
`configs/frozen/I6_models.yaml`, by a LINE-TARGETED edit: the registry is
mostly prose — the pre-registered prediction, the outcome rule, the
citations, the note about DINOv2's native resolution — and a YAML round trip
through PyYAML would delete every comment in it. The edit replaces exactly
the `weight_sha256:` line under each model and leaves every other byte
alone.

Append-safe: a model that already carries a sha is left untouched, and a
re-download that disagrees with a recorded sha is a HARD ERROR rather than a
silent overwrite (the standing rule after the legacy resume-overwrite
incident).

No training, no optimizer, no probe fitting, no inference.
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REGISTRY = "configs/frozen/I6_models.yaml"


def _registry_from_argv(argv=None) -> str:
    """`--registry` read straight off the command line, before argparse.

    The cache location has to be set before ANY heavy import (see below), and
    argparse cannot run that early without also importing the module it is
    configuring. Scanning argv for one flag is the smallest thing that works.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, arg in enumerate(argv):
        if arg == "--registry" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--registry="):
            return arg.split("=", 1)[1]
    return REGISTRY


def preset_cache_env(registry_path=None) -> str:
    """Point HF_HOME / TORCH_HOME at the registry's woody path — FIRST.

    `huggingface_hub` resolves its cache directory into MODULE CONSTANTS at
    import time, and `saga.frozen.external` imports `timm`, which imports
    `huggingface_hub`. Setting the variables after that import has no effect
    at all: the download silently lands in `$HOME/.cache/huggingface`.

    That is not hypothetical. On Alex, 2026-09-16, it put ~2 GB of weights on
    /home/hpc — a 104.9 G soft quota that was already at 119.9 G — and the
    compute nodes, which DO get the variables from the job file's exports,
    then found no cached file and every I6 array task failed
    (job 4262896). Hence: before the imports, not inside main().

    Only `yaml` is imported here, which pulls in nothing that reads the cache.
    """
    import yaml

    root = os.environ.get("SAGA_HF_HOME")
    if not root:
        path = Path(registry_path or _registry_from_argv())
        root = yaml.safe_load(path.read_text(encoding="utf-8"))["hf_home"]
    os.environ["HF_HOME"] = str(root)
    os.environ["TORCH_HOME"] = str(root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(root) / "hub")
    return str(root)


# MUST stay above the imports below. `# noqa: E402` on them is not a style
# concession — the ordering is the fix.
_CACHE_ROOT = preset_cache_env()

from saga.frozen.external import (UNAVAILABLE,  # noqa: E402
                                  WEIGHT_FILENAMES, ExternalError,
                                  availability, file_sha256, load_registry,
                                  set_cache_env, timm_version)


def download_weights(hub_id: str, cache_root: str):
    """`(local_path, filename)` for one hub repo, or raise.

    Downloads a FILE. No model is constructed, so no forward pass is
    reachable from here even by accident.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    last = None
    for name in WEIGHT_FILENAMES:
        try:
            return Path(hf_hub_download(repo_id=hub_id, filename=name)), name
        except EntryNotFoundError as exc:
            last = exc
            continue
    raise ExternalError(
        f"{hub_id}: none of {WEIGHT_FILENAMES} is present in the repo "
        f"({last})")


def set_sha_in_registry(text: str, model_id: str, sha: str) -> str:
    """Replace the `weight_sha256:` line belonging to ONE model.

    Line-targeted so the registry's comments — which carry the
    pre-registered prediction, the outcome rule and every citation — survive
    untouched. A PyYAML round trip would drop all of them.
    """
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"\s*-?\s*model_id:\s*{re.escape(model_id)}\s*$",
                    line.rstrip("\n")):
            start = i
            break
    if start is None:
        raise ExternalError(
            f"no `model_id: {model_id}` line in the registry to attach a sha "
            f"to")
    for j in range(start + 1, len(lines)):
        if re.match(r"\s*-\s*model_id:", lines[j]):
            break                                   # ran into the next model
        m = re.match(r"(\s*)weight_sha256:\s*(\S.*?)\s*$",
                     lines[j].rstrip("\n"))
        if m:
            lines[j] = f"{m.group(1)}weight_sha256: {sha}\n"
            return "".join(lines)
    raise ExternalError(
        f"model {model_id!r} has no `weight_sha256:` line to write into")


def recorded_sha(registry: dict, model_id: str):
    for entry in registry["models"]:
        if entry["model_id"] == model_id:
            value = str(entry.get("weight_sha256", "PENDING"))
            return None if value in ("PENDING", "", "None") else value
    return None


def main():
    p = argparse.ArgumentParser(
        description="Download TASK A / I6's external checkpoints and record "
                    "their weight shas. Login node; no inference.")
    p.add_argument("--registry", default=REGISTRY)
    p.add_argument("--dry-run", action="store_true",
                   help="resolve names and print the plan; touch no network "
                        "and write nothing")
    p.add_argument("--only", action="append", metavar="MODEL_ID",
                   help="restrict to these model_ids (repeatable)")
    args = p.parse_args()

    path = Path(args.registry)
    registry = load_registry(path)
    # BEFORE anything downloads. A dry run creates nothing: the registry's
    # path is the cluster's, and on a laptop mkdir would plant it on the
    # current drive.
    root = set_cache_env(registry, create=not args.dry_run)
    version = timm_version()

    print(f"registry:      {path}")
    print(f"timm:          {version}")
    print(f"HF_HOME:       {root}")
    print(f"models:        {len(registry['models'])}")
    # Where huggingface_hub ACTUALLY resolved its cache, read back from the
    # library rather than from our own environment variable. The two differ
    # exactly when the preset above ran too late, which is the failure this
    # tool has already had once.
    from huggingface_hub import constants as hf_constants
    print(f"hub cache:     {hf_constants.HF_HUB_CACHE}")
    if not str(hf_constants.HF_HUB_CACHE).startswith(str(root)):
        raise SystemExit(
            f"REFUSING TO DOWNLOAD: huggingface_hub resolved its cache to\n"
            f"  {hf_constants.HF_HUB_CACHE}\n"
            f"but the registry asks for\n"
            f"  {root}\n"
            f"The weights would land on the wrong filesystem and the compute "
            f"nodes would not find them. Set SAGA_HF_HOME (and HF_HOME) in "
            f"the shell before running this tool.")
    print()

    status = availability(registry)
    wanted = set(args.only) if args.only else None
    unavailable, done, failed = [], [], []

    for entry in registry["models"]:
        model_id = entry["model_id"]
        if wanted is not None and model_id not in wanted:
            continue
        if status[model_id].startswith(UNAVAILABLE):
            # RECORDED, never substituted (TASK A §12)
            print(f"  {model_id:34s} {status[model_id]}  "
                  f"({entry['timm_name']})")
            unavailable.append(model_id)
            continue
        have = recorded_sha(registry, model_id)
        if have and not args.dry_run:
            print(f"  {model_id:34s} already recorded {have[:16]}… — skipped")
            continue
        if args.dry_run:
            print(f"  {model_id:34s} would fetch {entry['hub_id']}")
            continue
        try:
            local, filename = download_weights(entry["hub_id"], root)
            sha = file_sha256(local)
            text = path.read_text(encoding="utf-8")
            path.write_text(set_sha_in_registry(text, model_id, sha),
                            encoding="utf-8", newline="\n")
            size = local.stat().st_size
            print(f"  {model_id:34s} {sha[:16]}…  {filename}  "
                  f"{size / 1e6:.1f} MB")
            done.append(model_id)
        except Exception as exc:                      # noqa: BLE001
            print(f"  {model_id:34s} FAILED: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            failed.append(model_id)

    print()
    if args.dry_run:
        print("dry run — nothing downloaded, nothing written")
        return 1 if unavailable else 0

    print(f"downloaded {len(done)}, unavailable {len(unavailable)}, "
          f"failed {len(failed)}")
    if unavailable:
        print(f"UNAVAILABLE under timm {version}: {unavailable}")
        print("  Recorded as such. TASK A §12 forbids substituting a "
              "different checkpoint; report the gap instead.")
    if done:
        print(f"\nwrote weight shas into {path} at "
              f"{datetime.now(timezone.utc).isoformat()}")
        print("Commit the registry before submitting scripts/jobs/"
              "frozen_I6.sbatch — the job verifies every sha before load.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
