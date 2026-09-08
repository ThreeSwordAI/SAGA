"""TASK-07 Phase C: address analysis (concentration re-derivation, border
geometry, correlations, sha gating, gate pooling order) and the generated
note. Includes the acceptance brute-force check against REAL addr files."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from scipy.stats import spearmanr

from analysis.address_analysis import (border_rings, build, build_perm,
                                       concentration_null,
                                       concentration_recompute, gini_pairwise,
                                       head_mean_gate, load_manifest,
                                       load_member, position_rel_se, rho,
                                       rho_spatial, save_npz,
                                       verify_stored_concentration, zrank)
from analysis.build_sink_address_note import Table, build_note
from tools.sink_address import concentration as tool_concentration

REPO = Path(__file__).resolve().parents[1]
N_POS = 196


# ── fixtures ─────────────────────────────────────────────────────────────────

def addr_payload(freq_canon, freq_mad=None, sha="sha1", arch="vit_small",
                 variant="baseline", n_images=100):
    freq_mad = freq_canon if freq_mad is None else freq_mad
    payload = {
        "schema": "sink_address_v1", "ckpt_sha256": sha, "arch": arch,
        "variant": variant, "n_images": n_images,
        "n_positions": len(freq_canon), "canon_key": f"{arch}|mixup",
        "canon_skip_reason": None, "tau_canon": 20.0, "k_mad": 5.0,
        "freq_canon": list(map(float, freq_canon)),
        "mass_canon": float(np.sum(freq_canon)),
        "concentration_canon": tool_concentration(np.asarray(freq_canon)),
        "freq_mad": list(map(float, freq_mad)),
        "mass_mad": float(np.sum(freq_mad)),
        "concentration_mad": tool_concentration(np.asarray(freq_mad)),
        "mean_norm": [1.0] * len(freq_canon),
        "p99_norm": [2.0] * len(freq_canon),
    }
    return payload


def t_has(rows, **kw):
    return any(all(r.get(k) == v for k, v in kw.items()) for r in rows)


def ring_map(peak_ring, high=0.5, low=0.01, side=14):
    """Frequency map whose mass sits on one ring from the border."""
    idx = np.arange(side)
    dist = np.minimum(np.minimum(idx[:, None], side - 1 - idx[:, None]),
                      np.minimum(idx[None, :], side - 1 - idx[None, :]))
    return np.where(dist == peak_ring, high, low).ravel()


def make_tree(root: Path, seed=0):
    """Repo-shaped tree mirroring the real cell membership: the ViT-S legacy
    mixup+nomix dirs, the ViT-B legacy nomix dir (the VOID ViT-B mixup-dir
    trio deliberately absent), registers as legacy-only, plus one seeded
    ViT-S run. A complete tree must yield zero gaps."""
    rng = np.random.RandomState(seed)
    (root / "results/legacy/diag").mkdir(parents=True)
    (root / "results/legacy/gates").mkdir(parents=True)

    man_rows = []
    for arch, dirname, n_heads in (("vit_small", "mixup", 6),
                                   ("vit_small", "nomix", 6),
                                   ("vit_base", "nomix", 12)):
        # registers exists as LEGACY repeats only — the real repo's shape
        for variant in ("baseline", "saga", "registers"):
            sha = f"sha_{arch}_{dirname}_{variant}"
            stem = f"e2_{arch}_{dirname}_{variant}_rlast_last"
            freq = ring_map(1) + rng.rand(N_POS) * 0.01
            (root / f"results/legacy/diag/{stem}_addr.json").write_text(
                json.dumps(addr_payload(freq, sha=sha, arch=arch,
                                        variant=variant)))
            man_rows.append({"path": f"/x/{stem}.pth", "filename": "last.pth",
                             "size_bytes": "1", "mtime_iso": "", "sha256": sha,
                             "exp": "e2", "arch": arch,
                             "recipe": dirname, "recipe_actual": "mixup",
                             "variant": variant, "seed": "rlast"})
            if variant == "saga":
                phi = rng.randn(12, n_heads, N_POS) * 0.1
                phi[11] = 0.0                      # constant final layer
                np.savez(root / f"results/legacy/gates/"
                                f"e2_{arch}_{dirname}_saga_rlast_last_phi.npz",
                         phi=phi)

    # DECOY: the VOID legacy ViT-B mixup-dir trio (never finished training).
    # Files exist on disk and are in the manifest, but the cell membership
    # imported from build_pooled_tables must exclude them entirely.
    for variant in ("baseline", "saga", "registers"):
        sha = f"sha_VOID_{variant}"
        stem = f"e2_vit_base_mixup_{variant}_rlast_last"
        (root / f"results/legacy/diag/{stem}_addr.json").write_text(
            json.dumps(addr_payload(ring_map(3), sha=sha, arch="vit_base",
                                    variant=variant)))
        man_rows.append({"path": f"/x/{stem}.pth", "filename": "last.pth",
                         "size_bytes": "1", "mtime_iso": "", "sha256": sha,
                         "exp": "e2", "arch": "vit_base", "recipe": "mixup",
                         "recipe_actual": "mixup", "variant": variant,
                         "seed": "rlast"})

    man = root / "results/legacy/checkpoint_manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man_rows[0]))
        w.writeheader()
        w.writerows(man_rows)

    # seeded runs: the mixup cell (s1) and the true-nomix cell (s1), the
    # latter so Q3's cross-recipe comparison has both sides
    for recipe, peak in (("mixup", 1), ("nomix", 4)):
        for variant in ("baseline", "saga"):
            run = root / f"results/runs/e2r_vits_{recipe}_{variant}_s1"
            (run / "diag").mkdir(parents=True)
            (run / "gates").mkdir(parents=True)
            (run / "config.resolved.yaml").write_text(
                yaml.safe_dump({"recipe": recipe, "variant": variant}))
            sha = f"sha_run_{recipe}_{variant}"
            freq = ring_map(peak) + rng.rand(N_POS) * 0.01
            (run / "diag/diag_final_last_addr.json").write_text(
                json.dumps(addr_payload(freq, sha=sha, variant=variant)))
            (run / "diag/diag_final_last.json").write_text(
                json.dumps({"arch": "vit_small", "ckpt_sha256": sha}))
            if variant == "saga":
                phi = rng.randn(12, 6, N_POS) * 0.1
                phi[11] = 0.0
                np.savez(run / "gates/phi_e299.npz", phi=phi)
    return man


# ── independent concentration re-derivation ──────────────────────────────────

def test_concentration_recompute_uniform_and_onehot():
    u = concentration_recompute(np.full(N_POS, 0.25))
    assert u["entropy_bits"] == pytest.approx(math.log2(N_POS))
    assert u["entropy_normalized"] == pytest.approx(1.0)
    assert u["gini"] == pytest.approx(0.0, abs=1e-12)
    assert u["top5_share"] == pytest.approx(5 / N_POS)

    f = np.zeros(N_POS)
    f[7] = 0.9
    o = concentration_recompute(f)
    assert o["entropy_bits"] == pytest.approx(0.0)
    assert o["gini"] == pytest.approx((N_POS - 1) / N_POS)
    assert o["top5_share"] == pytest.approx(1.0)


def test_gini_pairwise_matches_sorted_formula_but_is_independent():
    rng = np.random.RandomState(3)
    for _ in range(5):
        x = rng.gamma(0.4, 1.0, size=64)
        # the tool's sorted-cumulative implementation vs this pairwise one
        assert gini_pairwise(x) == pytest.approx(
            tool_concentration(x)["gini"], abs=1e-12)
    assert np.isnan(gini_pairwise(np.zeros(5)))


def test_verify_stored_concentration_catches_tampering():
    freq = ring_map(1)
    payload = addr_payload(freq)
    verify_stored_concentration(payload, "canon", "ok")   # passes untouched
    payload["concentration_canon"]["gini"] += 0.01
    with pytest.raises(SystemExit, match="disagrees"):
        verify_stored_concentration(payload, "canon", "tampered")


def test_verify_rejects_one_sided_nan():
    """`abs(nan - x) > tol` is False, so a naive check waves a NaN through."""
    payload = addr_payload(ring_map(1))
    payload["concentration_canon"]["gini"] = float("nan")
    with pytest.raises(SystemExit, match="finiteness"):
        verify_stored_concentration(payload, "canon", "nan")


def test_verify_checks_top10_positions():
    payload = addr_payload(ring_map(1))
    payload["concentration_canon"]["top10_positions"] = [999] * 10
    with pytest.raises(SystemExit, match="top10_positions"):
        verify_stored_concentration(payload, "canon", "bad-top10")


def test_concentration_recompute_rejects_zero_mass():
    with pytest.raises(ValueError, match="zero-mass"):
        concentration_recompute(np.zeros(N_POS))


# ── the finite-sample null (the F1 mechanism) ────────────────────────────────

def test_concentration_null_floor_moves_with_mass():
    """A spatially UNIFORM ground truth does not give Gini 0 when estimated
    from a finite number of images: the floor is ~0.017 at mass 20 but
    ~0.5 at mass 0.02, which is why raw Gini is not comparable across
    members and Q4 must difference EXCESS values."""
    high = concentration_null(20.0, 10000, N_POS)["gini"][0]
    low = concentration_null(0.02, 10000, N_POS)["gini"][0]
    assert high < 0.05
    assert low > 0.4
    assert low > 8 * high
    # the null entropy is near-uniform at high mass, well below it at low
    assert concentration_null(20.0, 10000, N_POS)["entropy_normalized"][0] > 0.999
    assert concentration_null(0.02, 10000, N_POS)["entropy_normalized"][0] < 0.95


def test_concentration_null_is_deterministic():
    a = concentration_null(5.0, 10000, N_POS)
    b = concentration_null(5.0, 10000, N_POS, seed=0)
    assert a["gini"] == b["gini"]


def test_position_rel_se_known_values():
    # p = mass/196; rel SE = sqrt((1-p)/(n*p))
    assert position_rel_se(19.7, 10000, N_POS) == pytest.approx(0.0299, abs=1e-3)
    assert position_rel_se(0.0233, 10000, N_POS) == pytest.approx(0.917, abs=1e-2)
    assert position_rel_se(0.0, 10000, N_POS) is None


# ── the exact spatial permutation null (the F2 mechanism) ────────────────────

def test_build_perm_is_a_complete_transform_set():
    perm = build_perm(14)
    assert perm.shape == (8 * 196, 196)
    assert any((perm[i] == np.arange(196)).all() for i in range(len(perm)))
    for i in range(0, len(perm), 61):
        assert len(np.unique(perm[i])) == 196


def test_rho_matches_scipy_including_heavy_ties():
    rng = np.random.RandomState(0)
    for trial in range(6):
        a, b = rng.rand(N_POS), rng.rand(N_POS)
        if trial > 2:
            a, b = np.round(a * 3), np.round(b * 2)      # heavy tie groups
        assert rho(a, b) == pytest.approx(spearmanr(a, b).statistic,
                                          abs=1e-12)
    assert zrank(np.ones(N_POS)) is None


def test_rho_spatial_p_is_exact_and_bounded():
    rng = np.random.RandomState(1)
    a = rng.rand(N_POS)
    r, p, sd, n = rho_spatial(a, a)
    assert r == pytest.approx(1.0)
    assert n == 8 * 196
    assert p == pytest.approx(1.0 / n)          # floor: identity is in the set
    assert p > 0                                 # never reported as zero

    # genuinely SPATIALLY SMOOTH maps (low-frequency, so neighbouring
    # positions are correlated): the spatial null is then far wider than
    # the iid reference 1/sqrt(195) = 0.0716, which is the whole point
    idx = np.arange(14)
    yy, xx = np.meshgrid(idx, idx, indexing="ij")
    smooth_a = (np.sin(2 * np.pi * yy / 14) + np.cos(2 * np.pi * xx / 14)).ravel()
    smooth_b = (np.cos(2 * np.pi * yy / 14)
                + np.sin(4 * np.pi * xx / 14)).ravel()
    _, _, sd_spatial, _ = rho_spatial(smooth_a, smooth_b)
    assert sd_spatial > 3 * (1 / (N_POS - 1) ** 0.5), sd_spatial

    # ... whereas white noise really does sit near the iid reference
    _, _, sd_iid, _ = rho_spatial(rng.rand(N_POS), rng.rand(N_POS))
    assert sd_iid < 2 * (1 / (N_POS - 1) ** 0.5)

    assert rho_spatial(a, np.ones(N_POS)) is None


@pytest.mark.parametrize("addr_file", [
    "results/legacy/diag/e2_vit_small_mixup_baseline_rlast_last_addr.json",
    "results/runs/e2r_vits_mixup_saga_s1/diag/diag_final_last_addr.json",
])
def test_real_addr_concentration_brute_force(addr_file):
    """Acceptance: concentration stats of REAL files verified against a
    third, loop-based brute force (no numpy vectorisation at all)."""
    path = REPO / addr_file
    if not path.exists():
        pytest.skip(f"{addr_file} not present (HPC output)")
    addr = json.loads(path.read_text())
    for basis in ("canon", "mad"):
        freq = addr[f"freq_{basis}"]
        stored = addr[f"concentration_{basis}"]
        total = 0.0
        for v in freq:
            total += v
        entropy = 0.0
        for v in freq:
            p = v / total
            if p > 0:
                entropy -= p * math.log(p) / math.log(2.0)
        num = 0.0
        for a in freq:
            for b in freq:
                num += abs(a - b)
        gini = num / (2.0 * len(freq) * len(freq) * (total / len(freq)))
        ordered = sorted(freq, reverse=True)
        top5 = sum(ordered[:5]) / total
        top20 = sum(ordered[:20]) / total

        assert stored["total_mass"] == pytest.approx(total, abs=1e-9)
        assert stored["entropy_bits"] == pytest.approx(entropy, abs=1e-9)
        assert stored["gini"] == pytest.approx(gini, abs=1e-9)
        assert stored["top5_share"] == pytest.approx(top5, abs=1e-9)
        assert stored["top20_share"] == pytest.approx(top20, abs=1e-9)
        assert stored["entropy_uniform_bits"] == pytest.approx(
            math.log2(len(freq)), abs=1e-12)


# ── geometry ─────────────────────────────────────────────────────────────────

def test_border_rings_known_geometry():
    prof = border_rings(ring_map(0, high=1.0, low=0.0))
    assert prof[0] == pytest.approx(1.0)
    assert all(v == pytest.approx(0.0) for v in prof[1:])

    prof = border_rings(ring_map(1, high=1.0, low=0.0))
    assert prof[1] == pytest.approx(1.0)
    assert prof[0] == pytest.approx(0.0)
    assert int(np.argmax(prof)) == 1

    flat = border_rings(np.full(N_POS, 0.3))
    assert all(v == pytest.approx(0.3) for v in flat)
    assert len(flat) == 7


# ── correlation helpers ──────────────────────────────────────────────────────

def test_rho_constant_maps_are_undefined():
    rng = np.random.RandomState(0)
    v = rng.rand(N_POS)
    assert rho(v, np.ones(N_POS)) is None
    assert rho(np.ones(N_POS), v) is None
    assert rho(v, v) == pytest.approx(1.0)
    assert rho(v, -v) == pytest.approx(-1.0)


def test_head_mean_gate_is_sigmoid_then_mean():
    """Regression guard for the TASK-06B operator-precedence defect: the
    pooling MUST be mean_h sigmoid(phi), never a mean of phi (or any
    harmonic-style pooling), which differs whenever heads disagree."""
    phi = np.array([[[2.0, -2.0], [-2.0, 2.0]]])          # [1, 2, 2]
    got = head_mean_gate(phi)
    expect = np.array([[(1 / (1 + math.exp(-2.0)) + 1 / (1 + math.exp(2.0))) / 2] * 2])
    np.testing.assert_allclose(got, expect)
    assert got[0, 0] == pytest.approx(0.5)

    # sigmoid(mean(phi)) would also give 0.5 here, so use an asymmetric case
    phi2 = np.array([[[3.0], [1.0]]])
    sig_then_mean = (1 / (1 + math.exp(-3.0)) + 1 / (1 + math.exp(-1.0))) / 2
    mean_then_sig = 1 / (1 + math.exp(-2.0))
    assert head_mean_gate(phi2)[0, 0] == pytest.approx(sig_then_mean)
    assert head_mean_gate(phi2)[0, 0] != pytest.approx(mean_then_sig)

    # and the head mean must be arithmetic, not harmonic
    harmonic = 2 / (1 / (1 / (1 + math.exp(-3.0))) + 1 / (1 / (1 + math.exp(-1.0))))
    assert head_mean_gate(phi2)[0, 0] != pytest.approx(harmonic)


# ── sha gating ───────────────────────────────────────────────────────────────

def test_manifest_sha_mismatch_is_a_gap(tmp_path, monkeypatch):
    man = make_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    manifest = load_manifest(man)
    key = ("vit_small", "mixup", "baseline", "rlast", "last.pth")
    path = Path("results/legacy/diag/"
                "e2_vit_small_mixup_baseline_rlast_last_addr.json")

    addr, reason = load_member(path, key, manifest, "ok")
    assert addr is not None and reason is None

    bad = dict(manifest)
    bad[key] = "different-sha"
    addr, reason = load_member(path, key, bad, "bad")
    assert addr is None and "mismatch vs manifest" in reason

    addr, reason = load_member(path, ("nope",) * 5, manifest, "missing")
    assert addr is None and "no manifest row" in reason


def test_run_dir_sha_checked_against_sibling(tmp_path, monkeypatch):
    make_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    path = Path("results/runs/e2r_vits_mixup_saga_s1/diag/"
                "diag_final_last_addr.json")
    addr, reason = load_member(path, None, {}, "ok")
    assert addr is not None and reason is None

    sib = path.with_name("diag_final_last.json")
    sib.write_text(json.dumps({"arch": "vit_small",
                               "ckpt_sha256": "other"}))
    addr, reason = load_member(path, None, {}, "bad")
    assert addr is None and "mismatch vs sibling diag" in reason

    sib.unlink()
    addr, reason = load_member(path, None, {}, "gone")
    assert addr is None and "no sibling" in reason


def test_missing_addr_file_is_a_gap(tmp_path, monkeypatch):
    make_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    addr, reason = load_member(Path("results/legacy/diag/nope_addr.json"),
                               None, {}, "x")
    assert addr is None and reason.startswith("no ")


# ── end-to-end on the synthetic tree ─────────────────────────────────────────

def test_build_end_to_end(tmp_path, monkeypatch):
    man = make_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    rows, members, profiles, gaps = build(man)
    assert gaps == []
    # vit_small|mixup gets both legacy dirs (erratum remap) + the s1 run;
    # registers is legacy-only, so it has just the two legacy repeats
    assert len(members[("vit_small", "mixup", "baseline")]) == 3
    assert len(members[("vit_small", "mixup", "saga")]) == 3
    assert len(members[("vit_small", "mixup", "registers")]) == 2
    # ViT-B/mixup takes ONLY the legacy nomix-dir repeat: the planted VOID
    # mixup-dir decoys exist on disk and in the manifest, yet appear nowhere
    for variant in ("baseline", "saga", "registers"):
        assert [tag for tag, _, _ in members[("vit_base", "mixup", variant)]] \
            == ["legacy-nomixdir"]
    assert (tmp_path / "results/legacy/diag"
            / "e2_vit_base_mixup_baseline_rlast_last_addr.json").exists()
    assert not t_has(rows, arch="vit_base", subject="legacy-mixupdir")
    # registers is compared to baseline in Q4 and never appears in Q5
    assert t_has(rows, question="Q4_relocation", variant="registers")
    assert not t_has(rows, question="Q5_gate_address", variant="registers")

    t = Table(rows)
    q2 = t.one(question="Q2_seed_stability", arch="vit_small",
               recipe_actual="mixup", map_basis="canon", variant="baseline",
               subject="all-pairs", statistic="spearman_mean")
    assert q2 is not None and int(q2["n"]) == 3          # 3 repeats -> 3 pairs
    # planted ring-1 maps must correlate strongly across repeats
    assert float(q2["value"]) > 0.5

    geo = t.find(question="Q1b_geometry", arch="vit_small",
                 recipe_actual="mixup", map_basis="canon",
                 variant="baseline", statistic="peak_ring")
    # planted maps peak on ring 1 (value is an int in memory, a string once
    # the row has been through the CSV writer)
    assert geo and all(int(r["value"]) == 1 for r in geo)

    # the constant final layer is excluded and counted, never averaged as NaN
    mean_rows = t.find(question="Q5_gate_address", arch="vit_small",
                       recipe_actual="mixup", map_basis="canon",
                       statistic="spearman_layer_mean")
    assert mean_rows
    for r in mean_rows:
        assert r["value"] != "MISSING"
        assert "n_layers_defined=11 of 12" in r["note"]
    assert all(np.isnan(p[11]) for k, p in profiles.items())

    # gate spatial std is emitted once (basis-independent)
    stds = t.find(question="Q5_gate_address", map_basis="-",
                  statistic="gate_spatial_std_layer11")
    assert stds and all(float(r["value"]) == 0.0 for r in stds)

    npz_path = tmp_path / "out.npz"
    n = save_npz(npz_path, members, profiles)
    z = np.load(npz_path)
    assert n == len(z["member_keys"])
    assert any(k.startswith("gate__") for k in z.files)
    assert any(k.startswith("profile__") for k in z.files)


# ── note generation ──────────────────────────────────────────────────────────

def test_q4_deltas_use_excess_not_raw_and_are_paired(tmp_path, monkeypatch):
    """REGRESSION (adversarial review F1+F5): a sparse variant map looks
    highly concentrated purely from Monte-Carlo noise, so differencing RAW
    concentration reports the wrong SIGN. Q4 must difference EXCESS values,
    paired by provenance.

    Fixture: the baseline is genuinely concentrated at high mass; the
    variant is spatially UNIFORM but very sparse. Raw Gini says the variant
    is MORE concentrated; excess Gini must say LESS.
    """
    rng = np.random.RandomState(5)
    (tmp_path / "results/legacy/diag").mkdir(parents=True)
    n_img = 10000

    # baseline: real structure, high mass (~20 sinks/image)
    base_freq = ring_map(1, high=0.30, low=0.05)
    # variant: NO structure at all, tiny mass (~0.02 sinks/image), drawn
    # from the uniform null itself
    var_freq = rng.binomial(n_img, 0.02 / N_POS, size=N_POS) / n_img

    raw_base = tool_concentration(base_freq)["gini"]
    raw_var = tool_concentration(var_freq)["gini"]
    assert raw_var > raw_base, "fixture must show the raw-Gini illusion"

    man_rows = []
    for variant, freq in (("baseline", base_freq), ("saga", var_freq)):
        sha = f"sha_{variant}"
        stem = f"e2_vit_small_mixup_{variant}_rlast_last"
        (tmp_path / f"results/legacy/diag/{stem}_addr.json").write_text(
            json.dumps(addr_payload(freq, sha=sha, variant=variant,
                                    n_images=n_img)))
        man_rows.append({"path": f"/x/{stem}.pth", "filename": "last.pth",
                         "size_bytes": "1", "mtime_iso": "", "sha256": sha,
                         "exp": "e2", "arch": "vit_small", "recipe": "mixup",
                         "recipe_actual": "mixup", "variant": variant,
                         "seed": "rlast"})
    man = tmp_path / "results/legacy/checkpoint_manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man_rows[0]))
        w.writeheader()
        w.writerows(man_rows)

    monkeypatch.chdir(tmp_path)
    rows, members, _, _ = build(man)
    t = Table(rows)

    delta = t.num(question="Q4_relocation", arch="vit_small",
                  recipe_actual="mixup", map_basis="canon", variant="saga",
                  subject="MEAN", comparator="baseline-matched",
                  statistic="delta_gini_excess")
    assert delta is not None
    assert delta < 0, (f"excess-Gini delta must be NEGATIVE (the sparse "
                       f"variant is less concentrated than its own null), "
                       f"got {delta}; raw would have said "
                       f"{raw_var - raw_base:+.4f}")

    # and no RAW delta may be emitted, so nobody can read the wrong sign
    assert not t.find(question="Q4_relocation", statistic="delta_gini")
    # the paired delta is over provenance-matched repeats only
    mean_row = t.one(question="Q4_relocation", arch="vit_small",
                     recipe_actual="mixup", map_basis="canon",
                     variant="saga", subject="MEAN",
                     statistic="delta_gini_excess")
    assert int(mean_row["n"]) == 1
    # the sparse map must be noise-flagged
    rel_se = t.num(question="Q1_concentration", arch="vit_small",
                   recipe_actual="mixup", map_basis="canon", variant="saga",
                   subject="legacy-mixupdir", statistic="position_rel_se")
    assert rel_se is not None and rel_se > 0.5


def test_correlations_carry_a_spatial_p_value(tmp_path, monkeypatch):
    """Every reported correlation must be accompanied by its exact
    permutation p and the null's sd — n=196 alone overstates the evidence."""
    man = make_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    rows, _, _, _ = build(man)
    t = Table(rows)
    for question in ("Q2_seed_stability", "Q3_recipe_stability",
                     "Q4_relocation"):
        corr = [r for r in t.find(question=question, statistic="spearman")
                if r["value"] != "MISSING"]
        assert corr, question
        for r in corr:
            assert t.num(question=question, arch=r["arch"],
                         recipe_actual=r["recipe_actual"],
                         map_basis=r["map_basis"], variant=r["variant"],
                         subject=r["subject"],
                         statistic="spearman_p_spatial") is not None
            assert "spatial-permutation p=" in r["note"]
    # Q5 extremal layers carry a Bonferroni correction for the 12-layer argmax
    absmax = [r for r in t.find(question="Q5_gate_address",
                                statistic="spearman_layer_absmax")
              if r["value"] != "MISSING"]
    assert absmax
    for r in absmax:
        assert "Bonferroni p=" in r["note"] and "layer=" in r["note"]


def test_q5_emits_missing_when_no_matched_baseline(tmp_path, monkeypatch):
    """A SAGA run without a same-provenance baseline must leave a MISSING
    row, never silently vanish from Q5 (which would let the headline count
    'all N repeats' over a subset)."""
    man = make_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    # remove the ViT-S mixup-dir BASELINE, keeping its saga partner
    (tmp_path / "results/legacy/diag"
     / "e2_vit_small_mixup_baseline_rlast_last_addr.json").unlink()
    rows, _, _, gaps = build(man)
    t = Table(rows)
    assert any("baseline|legacy-mixupdir" in g for g in gaps)
    miss = t.find(question="Q5_gate_address", arch="vit_small",
                  recipe_actual="mixup", subject="legacy-mixupdir",
                  comparator="baseline-matched",
                  statistic="spearman_layer_absmax")
    assert miss and all(r["value"] == "MISSING" for r in miss)
    assert all("no same-provenance baseline" in r["note"] for r in miss)
    q4miss = t.find(question="Q4_relocation", arch="vit_small",
                    recipe_actual="mixup", subject="legacy-mixupdir",
                    statistic="spearman")
    assert q4miss and all("EXCLUDED" in r["note"] for r in q4miss)


def test_note_numbers_come_from_the_csv(tmp_path, monkeypatch):
    man = make_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    rows, members, profiles, gaps = build(man)
    note = build_note(rows, Path("t.csv"), Path("t.npz"))
    t = Table(rows)
    conc = t.num(question="Q1_concentration", arch="vit_small",
                 recipe_actual="mixup", map_basis="canon", variant="baseline",
                 subject="MEAN", statistic="entropy_normalized")
    assert f"{conc:.4f}" in note
    assert "GENERATED by" in note
    for header in ("## Q1 ", "## Q1b ", "## Q2 ", "## Q3 ", "## Q4 ",
                   "## Q5 ", "## OPEN ITEMS"):
        assert header in note


def test_note_reports_sign_disagreement_instead_of_a_range():
    """A cell whose repeats disagree in sign must say so — never hide it
    behind a min..max range that straddles zero."""
    def gate_rows(values):
        rows = []
        for tag, layer, val in values:
            rows.append({
                "question": "Q5_gate_address", "arch": "vit_small",
                "recipe_actual": "nomix", "map_basis": "canon",
                "variant": "saga", "subject": tag,
                "comparator": "baseline-matched",
                "statistic": "spearman_layer_absmax", "value": str(val),
                "n": "196", "reference": "", "note": f"layer={layer}"})
            rows.append({
                "question": "Q5_gate_address", "arch": "vit_small",
                "recipe_actual": "nomix", "map_basis": "-",
                "variant": "saga", "subject": tag, "comparator": "-",
                "statistic": f"gate_spatial_std_layer{layer:02d}",
                "value": "0.01", "n": "196", "reference": "", "note": ""})
        return rows

    disagree = build_note(gate_rows([("s1", 3, 0.492), ("s2", 7, -0.641)]),
                          Path("t.csv"), Path("t.npz"))
    assert "SIGNS DISAGREE" in disagree
    assert "1 negative of 2" in disagree

    agree = build_note(gate_rows([("s1", 7, -0.5), ("s2", 8, -0.6)]),
                       Path("t.csv"), Path("t.npz"))
    assert "SIGNS DISAGREE" not in agree
    assert "all negative" in agree
    assert "layer(s) 7, 8" in agree


# ── the committed artifacts stay in sync ─────────────────────────────────────

def test_committed_note_matches_committed_table():
    """The note in results/notes must be the one the committed CSV renders
    (it is generated, so a stale note means someone edited it by hand)."""
    table = REPO / "results/tables/sink_address.csv"
    note = REPO / "results/notes/sink_address.md"
    if not (table.exists() and note.exists()):
        pytest.skip("phase-C artifacts not present")
    with open(table, newline="") as f:
        rows = list(csv.DictReader(f))
    rendered = build_note(rows, Path("results/tables/sink_address.csv"),
                          Path("results/figures_data/Faddr.npz"))
    assert note.read_text(encoding="utf-8").strip() == rendered.strip()
