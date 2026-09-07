#!/usr/bin/env python3
"""
tools/sink_address.py
=====================
TASK-07 A1 — runs on the HPC (CPU, minutes), where the git-ignored
`_norms.npz` files live.

    python tools/sink_address.py                    # both default roots
    python tools/sink_address.py --diag-dir results/legacy/diag \
        --runs-root results/runs \
        --thresholds results/diagsplit/fixed_thresholds_canon.json

For every `<stem>_norms.npz` with a sibling `<stem>.json` carrying
`ckpt_sha256` + `arch` (legacy diag files and run-dir diag_final_* files;
per-epoch diag_e###.json have neither an npz nor a sha and are never
matched), reads `last_block_patch_norms [n, N]` and writes a small
committable sibling `<stem>_addr.json`:

- freq_canon[N]: fraction of images in which position p is STRICTLY above
  the cell's canon tau (key "<arch>|<recipe_actual>" resolved exactly as
  tools/apply_fixed_thr.py --version canon does, legacy remap included).
- freq_mad[N]:   same with the per-image median + k*MAD threshold
  (k=5, LOWER medians — matching saga.metrics.sink_counts_mad).
- mean_norm[N], p99_norm[N] (percentile over images, linear interpolation).
- concentration stats per freq map: entropy in bits vs the uniform
  reference log2(N) (+ normalized), Gini (0 = uniform, (N-1)/N = single
  position), top-5/top-20 positions' share of total sink mass, top-10
  position indices (ties broken toward the lower index).
- provenance: ckpt_sha256, n_images, tau used, canon key, k, git sha.

Cross-checks against the sibling diag JSON (same npz -> same numbers):
`mass_canon` (= sum of freq_canon = mean per-image count) must equal the
sibling's `sink_fixed_canon` when that was computed with the same tau —
a mismatch is a hard ERROR and nothing is written. `mass_mad` vs the
sibling's `sink_mad_k5` is recorded but only warned about: the JSON value
was computed from fp32 activations in-model, the npz stores fp16, so
borderline tokens may differ slightly.

Idempotent / requeue-safe: an existing output is skipped iff its
ckpt_sha256, tau_canon and k_mad all match the current resolution (so a
thresholds-file change rewrites the affected files); writes are atomic
(tmp + os.replace). A file whose canon key cannot be resolved still gets
freq_mad/mean/p99 with freq_canon = null and the reason recorded
(status PARTIAL); it self-heals on rerun once the tau exists.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.run_registry import git_sha
from tools.apply_fixed_thr import resolve_key

K_MAD = 5.0
CANON_CROSSCHECK_TOL = 1e-8   # same arrays, same op -> only summation-order noise
MAD_WARN_TOL = 0.5            # fp16 npz vs fp32 in-model: small drift is expected


def _lower_median(a: np.ndarray, axis: int) -> np.ndarray:
    """Lower median — matches torch.median (see tools/compute_fixed_thr.py)."""
    idx = (a.shape[axis] - 1) // 2
    return np.take(np.sort(a, axis=axis), idx, axis=axis)


def freq_above_tau(norms: np.ndarray, tau: float) -> np.ndarray:
    """norms [n, N] -> freq [N]: fraction of images with v[i,p] > tau."""
    v = norms.astype(np.float64)
    return (v > tau).mean(axis=0)


def freq_above_mad(norms: np.ndarray, k: float = K_MAD) -> np.ndarray:
    """norms [n, N] -> freq [N]: fraction of images with
    v[i,p] > median_i + k*MAD_i (per-image lower-median threshold,
    the sink_counts_mad convention)."""
    v = norms.astype(np.float64)
    med = _lower_median(v, axis=1)
    mad = _lower_median(np.abs(v - med[:, None]), axis=1)
    thr = med + k * mad
    return (v > thr[:, None]).mean(axis=0)


def gini(x: np.ndarray):
    """Gini coefficient of a non-negative vector (0 = uniform,
    (n-1)/n = all mass on one element). None on zero total mass."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    total = x.sum()
    if total == 0:
        return None
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(2.0 * (i * x).sum() / (n * total) - (n + 1) / n)


def concentration(freq: np.ndarray) -> dict:
    """Concentration stats of one freq map [N]. Entropy is over the
    normalized mass distribution q = freq/sum(freq); uniform reference is
    log2(N). Zero-mass maps get null stats and an empty top-10."""
    f = np.asarray(freq, dtype=np.float64)
    n = f.size
    mass = float(f.sum())
    uniform_bits = float(np.log2(n))
    out = {
        "total_mass": mass,
        "entropy_uniform_bits": uniform_bits,
        "entropy_bits": None,
        "entropy_normalized": None,
        "gini": None,
        "top5_share": None,
        "top20_share": None,
        "top10_positions": [],
    }
    if mass == 0:
        return out
    q = f / mass
    nz = q[q > 0]
    h = float(-(nz * np.log2(nz)).sum())
    desc = np.sort(q)[::-1]
    order = np.argsort(-f, kind="stable")
    out.update({
        "entropy_bits": h,
        "entropy_normalized": h / uniform_bits,
        "gini": gini(f),
        "top5_share": float(desc[:5].sum()),
        "top20_share": float(desc[:20].sum()),
        "top10_positions": [int(p) for p in order[: min(10, n)]],
    })
    return out


def build_addr(norms: np.ndarray, sibling: dict, tau, canon_key,
               canon_skip_reason, npz_name: str) -> dict:
    """Assemble the full _addr.json payload (no I/O). tau may be None."""
    v = norms.astype(np.float64)
    n_images, n_positions = v.shape

    f_mad = freq_above_mad(v, K_MAD)
    # mean over images of the per-image count — the exact op of
    # saga.metrics.sink_counts_mad / apply_fixed_thr.fixed_count_mean,
    # so it is comparable digit-for-digit with the diag JSON fields
    med = _lower_median(v, axis=1)
    mad = _lower_median(np.abs(v - med[:, None]), axis=1)
    mass_mad = float((v > (med + K_MAD * mad)[:, None]).sum(axis=1).mean())

    result = {
        "schema": "sink_address_v1",
        "source_npz": npz_name,
        "ckpt_sha256": sibling["ckpt_sha256"],
        "arch": sibling["arch"],
        "variant": sibling.get("variant"),
        "git_sha": git_sha(),
        "n_images": int(n_images),
        "n_positions": int(n_positions),
        "canon_key": canon_key,
        "canon_skip_reason": canon_skip_reason,
        "tau_canon": tau,
        "k_mad": K_MAD,
        "freq_canon": None,
        "mass_canon": None,
        "concentration_canon": None,
        "freq_mad": f_mad.tolist(),
        "mass_mad": mass_mad,
        "concentration_mad": concentration(f_mad),
        "mean_norm": v.mean(axis=0).tolist(),
        "p99_norm": np.percentile(v, 99, axis=0).tolist(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if tau is not None:
        f_canon = freq_above_tau(v, tau)
        result.update({
            "freq_canon": f_canon.tolist(),
            "mass_canon": float((v > tau).sum(axis=1).mean()),
            "concentration_canon": concentration(f_canon),
        })
    return result


def process_one(npz_path: Path, thresholds: dict) -> str:
    """Compute + write <stem>_addr.json for one npz. Returns a status line
    starting with 'wrote', 'PARTIAL', 'SKIP' or 'ERROR'."""
    sibling_path = npz_path.with_name(npz_path.name.replace("_norms.npz", ".json"))
    if not sibling_path.exists():
        return "SKIP (no sibling diag JSON)"
    try:
        sibling = json.load(open(sibling_path))
    except ValueError:
        return f"ERROR (unreadable sibling JSON {sibling_path.name})"
    sha = sibling.get("ckpt_sha256")
    arch = sibling.get("arch")
    if not sha:
        return "SKIP (sibling has no ckpt_sha256)"
    if not arch:
        return "SKIP (sibling has no arch)"

    canon_key, canon_skip_reason = resolve_key(sibling_path, arch, "canon")
    tau = None
    if canon_key is not None:
        if canon_key in thresholds:
            tau = float(thresholds[canon_key])
        else:
            canon_skip_reason = f"key {canon_key!r} not in thresholds"

    # the hard cross-check below engages only when the sibling carries a
    # comparable sink_fixed_canon; recorded here so the skip guard can
    # re-run a file whose sibling gained the fields after the first pass
    crosscheckable = (tau is not None
                      and sibling.get("sink_fixed_canon") is not None
                      and sibling.get("canon_thr_value") == tau)

    out_path = npz_path.with_name(npz_path.name.replace("_norms.npz", "_addr.json"))
    if out_path.exists():
        try:
            old = json.load(open(out_path))
        except ValueError:
            old = {}
        if (old.get("ckpt_sha256") == sha
                and old.get("tau_canon") == tau
                and old.get("k_mad") == K_MAD
                and old.get("n_images") == sibling.get("n_images")
                and not (crosscheckable
                         and not isinstance(old.get("crosscheck_canon"), dict))):
            return "SKIP (up to date)"

    try:
        with np.load(npz_path) as z:
            norms = z["last_block_patch_norms"]
    except Exception as exc:
        return f"ERROR (cannot read npz: {exc})"
    if norms.ndim != 2 or norms.shape[0] == 0 or norms.shape[1] < 2:
        return f"ERROR (bad norms shape {norms.shape})"
    if "n_images" in sibling and sibling["n_images"] != norms.shape[0]:
        return (f"ERROR (n_images mismatch: npz {norms.shape[0]} vs "
                f"sibling {sibling['n_images']})")

    result = build_addr(norms, sibling, tau, canon_key,
                        canon_skip_reason, npz_path.name)

    # hard cross-check: sibling sink_fixed_canon came from THIS npz via the
    # same strict-> mean (apply_fixed_thr), so with the same tau the values
    # can only differ by summation noise — anything more means the wrong
    # array, tau or file, and nothing is written
    if crosscheckable:
        diff = abs(result["mass_canon"] - sibling["sink_fixed_canon"])
        result["crosscheck_canon"] = {
            "sibling_sink_fixed_canon": sibling["sink_fixed_canon"],
            "abs_diff": diff,
        }
        if diff > CANON_CROSSCHECK_TOL:
            return (f"ERROR (mass_canon {result['mass_canon']:.6f} != sibling "
                    f"sink_fixed_canon {sibling['sink_fixed_canon']:.6f})")
    else:
        result["crosscheck_canon"] = "not comparable (no sibling value or different tau)"

    # soft cross-check: sink_mad_k5 was computed in-model from fp32
    # activations, the npz is fp16 — record the drift, warn if large
    warn = ""
    if sibling.get("sink_mad_k5") is not None:
        diff = result["mass_mad"] - sibling["sink_mad_k5"]
        result["crosscheck_mad"] = {
            "sibling_sink_mad_k5": sibling["sink_mad_k5"],
            "diff": diff,
        }
        if abs(diff) > MAD_WARN_TOL:
            warn = (f" [WARN mad mass {result['mass_mad']:.4f} vs sibling "
                    f"{sibling['sink_mad_k5']:.4f}]")
    else:
        result["crosscheck_mad"] = "not comparable (no sibling sink_mad_k5)"

    # atomic write — the addr JSON is its own completion marker; compact
    # like the sibling _normstats.json (the arrays make indent bulky)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, out_path)

    if tau is None:
        return f"PARTIAL {out_path.name} (no canon tau: {canon_skip_reason}){warn}"
    return f"wrote {out_path.name} (tau={tau}){warn}"


def collect_npz(diag_dir: Path, runs_root) -> list:
    targets = sorted(diag_dir.glob("*_norms.npz")) if diag_dir else []
    if runs_root:
        targets += sorted(Path(runs_root).glob("*/diag/*_norms.npz"))
    return targets


def main():
    parser = argparse.ArgumentParser(
        description="Per-position sink-address maps from _norms.npz files "
                    "(runs on the HPC beside the npz).")
    parser.add_argument("--diag-dir", default="results/legacy/diag")
    parser.add_argument("--runs-root", default="results/runs",
                        help="also process every <root>/*/diag/*_norms.npz")
    parser.add_argument("--thresholds",
                        default="results/diagsplit/fixed_thresholds_canon.json")
    args = parser.parse_args()

    with open(args.thresholds) as f:
        thresholds = json.load(f)

    targets = collect_npz(Path(args.diag_dir), args.runs_root)
    if not targets:
        sys.exit(f"no *_norms.npz under {args.diag_dir} or {args.runs_root}")

    counts = {"wrote": 0, "PARTIAL": 0, "SKIP": 0, "ERROR": 0}
    for npz_path in targets:
        status = process_one(npz_path, thresholds)
        counts[status.split(" ", 1)[0]] += 1
        print(f"  {npz_path.name}: {status}")
    print(f"done: {counts['wrote']} written, {counts['PARTIAL']} partial, "
          f"{counts['SKIP']} skipped, {counts['ERROR']} errors")
    if counts["ERROR"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
