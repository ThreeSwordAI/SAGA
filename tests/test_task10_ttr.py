"""
tests/test_task10_ttr.py
========================
TASK-10 PHASE A tests for test-time registers (saga/ttr.py,
tools/ttr_validate.py). CPU, fake data, tiny models.

The four A4 items are marked [A4] in their docstrings; the rest pin the
mechanism, the guards, and the validation tool's contract.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.metrics import infer_num_prefix_tokens
from saga.ttr import (CRITERION_CONTRAST, CRITERION_MEAN_ABS, NEURONS_JSON_KEYS,
                      NEURONS_JSON_SCHEMA, TTRModel, apply_ttr,
                      find_register_neurons, load_neurons_json,
                      mlp_act_module, mlp_hidden_dim, neurons_to_dict,
                      top_k, write_neurons_json, _signed_max_per_image)
from saga.vit import build_saga_vit
from tools.build_diag_split import DiagSplitDataset
from tools.ttr_validate import (carve_subsets, infer_base_run_id, measure,
                                resolve_canon_tau, verdict)

TINY = "vit_tiny_patch16_224"


def tiny_model(gate=False, num_classes=10):
    torch.manual_seed(0)
    return build_saga_vit(TINY, gate=gate, num_classes=num_classes).eval()


def fake_batch(n=4):
    torch.manual_seed(1)
    return torch.randn(n, 3, 224, 224)


def fake_loader(n_images=4, batch=2, num_classes=10):
    torch.manual_seed(2)
    xs = torch.randn(n_images, 3, 224, 224)
    ys = torch.arange(n_images) % num_classes
    return DataLoader(list(zip(xs, ys)), batch_size=batch, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# A4.1 — the patch-token contract
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_tokens_count_and_order_intact():
    """[A4] The splice preserves patch count AND raster order, and the extra
    tokens are zeros sitting between CLS and the patches."""
    model = tiny_model()
    x = fake_batch(3)
    n_extra = 2

    seen = {}

    def grab(key):
        return lambda mod, args: seen.__setitem__(key, args[0].detach().clone())

    h = model.blocks[0].register_forward_pre_hook(grab("plain"))
    with torch.no_grad():
        model(x)
    h.remove()

    neurons = [(4, 3, 1.0), (6, 11, 0.5)]
    with apply_ttr(model, neurons, n_extra_tokens=n_extra) as patched:
        h = patched.blocks[0].register_forward_pre_hook(grab("ttr"))
        with torch.no_grad():
            patched(x)
        h.remove()
        P = infer_num_prefix_tokens(patched)

    plain, ttr = seen["plain"], seen["ttr"]
    assert plain.shape[1] == 1 + 196
    assert ttr.shape[1] == 1 + n_extra + 196
    assert P == 1 + n_extra
    # CLS unchanged, extras are exactly zero, patches bit-identical IN ORDER
    assert torch.equal(ttr[:, :1, :], plain[:, :1, :])
    assert torch.equal(ttr[:, 1:1 + n_extra, :],
                       torch.zeros_like(ttr[:, 1:1 + n_extra, :]))
    assert torch.equal(ttr[:, 1 + n_extra:, :], plain[:, 1:, :])


def test_patch_slice_shape_through_infer_num_prefix_tokens():
    """[A4] The downstream slice x[:, P:, :] yields the same 196 patches."""
    model = tiny_model()
    feats = {}
    with apply_ttr(model, [(5, 1, 1.0)], n_extra_tokens=1) as patched:
        P = infer_num_prefix_tokens(patched)
        h = patched.blocks[-1].register_forward_hook(
            lambda m, i, o: feats.__setitem__("x", o.detach()))
        with torch.no_grad():
            patched(fake_batch(2))
        h.remove()
    assert feats["x"].shape[1] == 1 + 1 + 196
    assert feats["x"][:, P:, :].shape[1] == 196


def test_num_prefix_tokens_restored_and_model_untouched():
    """The wrapped model is never written to; exit leaves no trace."""
    model = tiny_model()
    x = fake_batch(2)
    with torch.no_grad():
        before = model(x)
    assert infer_num_prefix_tokens(model) == 1
    with apply_ttr(model, [(3, 2, 1.0)], n_extra_tokens=1) as patched:
        assert infer_num_prefix_tokens(patched) == 2
        assert infer_num_prefix_tokens(model) == 1   # inner untouched DURING
    assert infer_num_prefix_tokens(model) == 1
    with torch.no_grad():
        after = model(x)
    assert torch.equal(before, after)


# ─────────────────────────────────────────────────────────────────────────────
# A4.2 — the no-op sanity check
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_neuron_list_is_a_noop():
    """[A4] Empty selection => patched logits == unpatched (atol 1e-4)."""
    model = tiny_model()
    x = fake_batch(3)
    with torch.no_grad():
        plain = model(x)
    with apply_ttr(model, []) as patched:
        with torch.no_grad():
            out = patched(x)
        assert infer_num_prefix_tokens(patched) == 1
    assert torch.allclose(out, plain, atol=1e-4)
    assert torch.equal(out, plain), "empty selection must be an EXACT no-op"


def test_empty_dict_and_empty_layers_are_noops():
    model = tiny_model()
    x = fake_batch(2)
    with torch.no_grad():
        plain = model(x)
    for empty in ({}, {4: []}, []):
        with apply_ttr(model, empty) as patched:
            with torch.no_grad():
                assert torch.equal(patched(x), plain)


# ─────────────────────────────────────────────────────────────────────────────
# A4.3 — determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_find_register_neurons_deterministic():
    """[A4] Two calls, same seed, same loader order => identical ranking."""
    model = tiny_model()
    a = find_register_neurons(model, fake_loader(), 20, None,
                              outlier_tau=1.0, seed=0, max_images=4)
    b = find_register_neurons(model, fake_loader(), 20, None,
                              outlier_tau=1.0, seed=0, max_images=4)
    assert a == b
    assert len(a) == 20
    # descending by score
    assert all(a[i][2] >= a[i + 1][2] for i in range(len(a) - 1))


def test_find_register_neurons_scan_stats_and_layer_range():
    model = tiny_model()
    hidden = mlp_hidden_dim(model.blocks[0])
    r = find_register_neurons(model, fake_loader(), None, (4, 7),
                              outlier_tau=1.0, seed=0, max_images=4)
    stats = find_register_neurons.last_scan_stats
    assert stats["layer_range"] == [4, 7]
    assert stats["num_layers_scanned"] == 3
    assert stats["num_neurons_per_layer"] == hidden
    assert stats["num_prefix_tokens"] == 1
    assert stats["n_images_seen"] == 4
    assert len(r) == 3 * hidden
    assert {l for l, _, _ in r} <= {4, 5, 6}


def test_find_register_neurons_max_images_caps_the_scan():
    model = tiny_model()
    find_register_neurons(model, fake_loader(n_images=8, batch=3), 5, None,
                          outlier_tau=1.0, seed=0, max_images=5)
    assert find_register_neurons.last_scan_stats["n_images_seen"] == 5


def test_find_register_neurons_raises_without_outliers():
    """An unreachable tau must fail loudly, as the authors' version does."""
    model = tiny_model()
    with pytest.raises(RuntimeError, match="no image produced an outlier"):
        find_register_neurons(model, fake_loader(), 4, None,
                              outlier_tau=1e9, seed=0, max_images=4)


def test_contrast_criterion_differs_from_authors_criterion():
    """The two documented criteria are genuinely different rankings.

    tau 9.0 sits near the median of this model's patch norms (7.5..10.5 on
    the fake batch), so both outlier and non-outlier tokens exist — the
    contrast criterion needs both sides to be defined.
    """
    model = tiny_model()
    kw = dict(outlier_tau=9.0, seed=0, max_images=4)
    a = find_register_neurons(model, fake_loader(), 30, None,
                              criterion=CRITERION_MEAN_ABS, **kw)
    b = find_register_neurons(model, fake_loader(), 30, None,
                              criterion=CRITERION_CONTRAST, **kw)
    assert [x[:2] for x in a] != [x[:2] for x in b]


def test_unknown_criterion_refused():
    model = tiny_model()
    with pytest.raises(ValueError, match="criterion must be one of"):
        find_register_neurons(model, fake_loader(), 4, None,
                              outlier_tau=1.0, criterion="nope")


# ─────────────────────────────────────────────────────────────────────────────
# A4.4 — neurons.json round-trip
# ─────────────────────────────────────────────────────────────────────────────

def _write_sample_json(tmp_path, neurons=None, top_n=3):
    neurons = neurons if neurons is not None else [
        (5, 12, 9.5), (5, 77, 8.25), (7, 3, 7.125), (8, 100, 6.0625)]
    return write_neurons_json(
        tmp_path / "neurons.json", neurons,
        criterion=CRITERION_MEAN_ABS, outlier_tau=20.8515625,
        tau_key="vit_small|mixup",
        tau_source="results/diagsplit/fixed_thresholds_canon.json",
        seed=0, arch="vit_small", variant="baseline",
        ckpt="/x/last.pth", ckpt_sha256="ab" * 32,
        scan_stats=dict(n_images_seen=500, n_images_scored=487,
                        layer_range=[0, 12], num_layers_scanned=12,
                        num_neurons_per_layer=1536, num_prefix_tokens=1,
                        detect_outliers_layer=-1),
        top_n_stored=top_n)


def test_neurons_json_round_trip(tmp_path):
    """[A4] Schema round-trip: every mandated key survives, values intact."""
    payload = _write_sample_json(tmp_path, top_n=4)
    back = load_neurons_json(tmp_path / "neurons.json")
    for key in NEURONS_JSON_KEYS:
        assert key in back, key
    assert back["schema"] == NEURONS_JSON_SCHEMA
    assert back["criterion"] == CRITERION_MEAN_ABS
    assert back["criterion_description"] == payload["criterion_description"]
    assert back["outlier_tau"] == 20.8515625
    assert back["tau_key"] == "vit_small|mixup"
    assert back["neurons"] == [(5, 12, 9.5), (5, 77, 8.25),
                              (7, 3, 7.125), (8, 100, 6.0625)]
    assert back["n_images_scored"] == 487
    # the ranking round-trips straight back into a selection
    assert neurons_to_dict(top_k(back["neurons"], 3)) == {5: [12, 77], 7: [3]}


def test_neurons_json_truncates_to_top_n_and_records_it(tmp_path):
    _write_sample_json(tmp_path, top_n=2)
    back = load_neurons_json(tmp_path / "neurons.json")
    assert back["top_n_stored"] == 2
    assert len(back["neurons"]) == 2


def test_neurons_json_rejects_bad_schema_and_missing_keys(tmp_path):
    _write_sample_json(tmp_path)
    path = tmp_path / "neurons.json"
    good = json.loads(path.read_text())

    bad = dict(good, schema="something/else")
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="schema"):
        load_neurons_json(path)

    stripped = {k: v for k, v in good.items() if k != "criterion"}
    path.write_text(json.dumps(stripped))
    with pytest.raises(ValueError, match="missing keys"):
        load_neurons_json(path)


def test_neurons_json_write_is_atomic_no_tmp_left(tmp_path):
    _write_sample_json(tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


# ─────────────────────────────────────────────────────────────────────────────
# The mechanism itself
# ─────────────────────────────────────────────────────────────────────────────

def test_signed_max_keeps_the_sign_of_the_largest_magnitude():
    """The authors' sign_max: largest |value|, sign preserved, per image."""
    t = torch.tensor([[[1.0, -5.0], [2.0, 3.0]],       # image 0: -5 wins
                      [[7.0, -2.0], [0.5, 1.0]]])      # image 1: +7 wins
    assert torch.equal(_signed_max_per_image(t), torch.tensor([-5.0, 7.0]))


def test_intervention_writes_signed_max_into_extras_and_zeros_patches():
    """The redirection does exactly what it claims, at the hooked layer."""
    model = tiny_model()
    layer, idxs, n_extra = 5, [3, 9, 17], 2
    captured = {}

    with apply_ttr(model, [(layer, i, 1.0) for i in idxs],
                   n_extra_tokens=n_extra) as patched:
        h = mlp_act_module(patched.blocks[layer]).register_forward_hook(
            lambda m, i, o: captured.__setitem__("post", o.detach().clone()))
        with torch.no_grad():
            patched(fake_batch(3))
        h.remove()

    post = captured["post"]
    assert post.shape[1] == 1 + n_extra + 196
    # patches zeroed at the selected neurons, untouched elsewhere
    assert torch.equal(post[:, 1 + n_extra:, idxs],
                       torch.zeros_like(post[:, 1 + n_extra:, idxs]))
    other = [i for i in range(post.shape[2]) if i not in idxs]
    assert post[:, 1 + n_extra:, other].abs().sum() > 0
    # every extra token carries the SAME per-image scalar at every selected
    # neuron (the authors broadcast one sign_max per image/layer)
    fill = post[:, 1:1 + n_extra, idxs]
    for b in range(fill.shape[0]):
        assert torch.allclose(fill[b], fill[b].flatten()[0].expand_as(fill[b]))
    # CLS is never touched by the intervention
    assert post[:, 0, idxs].abs().sum() > 0


def test_scale_multiplies_the_redirected_value():
    model = tiny_model()
    layer, idxs = 5, [3]
    got = {}

    def run(scale):
        with apply_ttr(model, [(layer, i, 1.0) for i in idxs],
                       n_extra_tokens=1, scale=scale) as patched:
            h = mlp_act_module(patched.blocks[layer]).register_forward_hook(
                lambda m, i, o: got.__setitem__(scale, o.detach().clone()))
            with torch.no_grad():
                patched(fake_batch(2))
            h.remove()
        return got[scale][:, 1, idxs]

    one, half = run(1.0), run(0.5)
    assert torch.allclose(half, one * 0.5, atol=1e-6)


def test_normal_values_same_leaves_patch_activations_alone():
    model = tiny_model()
    layer, idxs = 5, [3, 9]
    got = {}
    with apply_ttr(model, [(layer, i, 1.0) for i in idxs],
                   n_extra_tokens=1, normal_values="same") as patched:
        h = mlp_act_module(patched.blocks[layer]).register_forward_hook(
            lambda m, i, o: got.__setitem__("x", o.detach().clone()))
        with torch.no_grad():
            patched(fake_batch(2))
        h.remove()
    assert got["x"][:, 2:, idxs].abs().sum() > 0


def test_normal_values_mean_makes_patches_uniform():
    model = tiny_model()
    layer, idxs = 5, [3, 9]
    got = {}
    with apply_ttr(model, [(layer, i, 1.0) for i in idxs],
                   n_extra_tokens=1, normal_values="mean") as patched:
        h = mlp_act_module(patched.blocks[layer]).register_forward_hook(
            lambda m, i, o: got.__setitem__("x", o.detach().clone()))
        with torch.no_grad():
            patched(fake_batch(2))
        h.remove()
    patch = got["x"][:, 2:, idxs]
    assert torch.allclose(patch, patch[:, :1, :].expand_as(patch), atol=1e-6)


def test_prefix_placement_matches_appended_placement():
    """Our prefix layout == the authors' appended layout, up to fp32 noise.

    The docstring of saga/ttr.py claims the two placements are equivalent by
    permutation-equivariance; this measures it instead of asserting it.
    """
    model = tiny_model()
    x = fake_batch(2)
    n_extra = 1
    grouped = neurons_to_dict([(5, 12, 9.0), (7, 3, 7.0)])

    def run(mode):
        handles = []

        def insert(mod, args):
            t = args[0]
            e = t.new_zeros(t.shape[0], n_extra, t.shape[2])
            t2 = (torch.cat([t[:, :1], e, t[:, 1:]], 1) if mode == "prefix"
                  else torch.cat([t, e], 1))
            return (t2,) + args[1:]

        def make(idx):
            def hook(mod, inp, out):
                o = out.clone()
                fill = _signed_max_per_image(o[:, :, idx])
                if mode == "prefix":
                    o[:, 1:1 + n_extra, idx] = fill[:, None, None]
                    o[:, 1 + n_extra:, idx] = 0
                else:
                    o[:, -n_extra:, idx] = fill[:, None, None]
                    o[:, 1:-n_extra, idx] = 0
                return o
            return hook

        handles.append(model.blocks[0].register_forward_pre_hook(insert))
        for layer, idxs in grouped.items():
            handles.append(mlp_act_module(model.blocks[layer])
                           .register_forward_hook(make(list(idxs))))
        seen = {}
        h = model.blocks[-1].register_forward_hook(
            lambda m, i, o: seen.__setitem__("x", o.detach()))
        with torch.no_grad():
            logits = model(x)
        h.remove()
        for hh in handles:
            hh.remove()
        t = seen["x"]
        patch = (t[:, 1 + n_extra:, :] if mode == "prefix"
                 else t[:, 1:-n_extra, :])
        return logits, patch

    lo_p, pa_p = run("prefix")
    lo_a, pa_a = run("append")
    assert torch.allclose(lo_p, lo_a, atol=1e-4)
    assert torch.allclose(pa_p, pa_a, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Guards
# ─────────────────────────────────────────────────────────────────────────────

def test_saga_gated_model_refused_by_both_entry_points():
    """The SpatialGate hard-codes n_patches = N-1; extra tokens break it.
    A loud refusal beats a misaligned gate map."""
    gated = tiny_model(gate=True)
    with pytest.raises(ValueError, match="SpatialGate"):
        with apply_ttr(gated, [(5, 1, 1.0)]):
            pass
    with pytest.raises(ValueError, match="SpatialGate"):
        find_register_neurons(gated, fake_loader(), 4, None, outlier_tau=1.0)


def test_out_of_range_layer_and_neuron_refused():
    model = tiny_model()
    hidden = mlp_hidden_dim(model.blocks[0])
    with pytest.raises(ValueError, match="layer 99 out of range"):
        with apply_ttr(model, [(99, 0, 1.0)]):
            pass
    with pytest.raises(ValueError, match="neuron indices"):
        with apply_ttr(model, [(0, hidden, 1.0)]):
            pass


def test_bad_layer_range_refused():
    model = tiny_model()
    for rng in [(-1, 4), (5, 5), (0, 99), (7, 3)]:
        with pytest.raises(ValueError, match="not a valid half-open range"):
            find_register_neurons(model, fake_loader(), 4, rng,
                                  outlier_tau=1.0)


def test_nesting_refused():
    model = tiny_model()
    with apply_ttr(model, [(5, 1, 1.0)]):
        with pytest.raises(RuntimeError, match="already active"):
            with apply_ttr(model, [(6, 2, 1.0)]):
                pass


def test_zero_extra_tokens_refused_for_nonempty_selection():
    model = tiny_model()
    with pytest.raises(ValueError, match="n_extra_tokens must be >= 1"):
        with apply_ttr(model, [(5, 1, 1.0)], n_extra_tokens=0):
            pass


def test_bad_normal_values_refused():
    model = tiny_model()
    with pytest.raises(ValueError, match="normal_values must be one of"):
        with apply_ttr(model, [(5, 1, 1.0)], normal_values="nope"):
            pass


def test_hooks_removed_even_when_body_raises():
    model = tiny_model()
    x = fake_batch(2)
    with torch.no_grad():
        plain = model(x)
    with pytest.raises(ZeroDivisionError):
        with apply_ttr(model, [(5, 1, 1.0)]):
            1 / 0
    with torch.no_grad():
        assert torch.equal(model(x), plain)
    assert infer_num_prefix_tokens(model) == 1


def test_block_without_mlp_act_refused():
    class NoMlp(torch.nn.Module):
        pass
    from saga.ttr import mlp_act_module as f
    with pytest.raises(TypeError, match=r"no \.mlp\.act"):
        f(NoMlp())


# ─────────────────────────────────────────────────────────────────────────────
# Selection helpers and the proxy
# ─────────────────────────────────────────────────────────────────────────────

def test_neurons_to_dict_normalises_all_accepted_shapes():
    assert neurons_to_dict([(1, 5, 0.9), (1, 6, 0.8), (2, 7, 0.7)]) == \
        {1: [5, 6], 2: [7]}
    assert neurons_to_dict([(1, 5), (1, 5), (1, 6)]) == {1: [5, 6]}
    assert neurons_to_dict({3: [1, 2], 4: []}) == {3: [1, 2]}
    assert neurons_to_dict([]) == {}


def test_top_k_is_a_prefix_of_one_ranking():
    ranking = [(0, i, 10.0 - i) for i in range(10)]
    assert top_k(ranking, 3) == ranking[:3]
    assert top_k(ranking, 0) == []
    assert top_k(ranking, 99) == ranking
    with pytest.raises(ValueError):
        top_k(ranking, -1)


def test_ttrmodel_proxy_delegates_and_exposes_inner():
    model = tiny_model()
    proxy = TTRModel(model, 3)
    assert proxy.num_prefix_tokens == 3
    assert infer_num_prefix_tokens(proxy) == 3
    assert proxy.blocks is model.blocks           # same object => hooks work
    assert proxy.ttr_inner is model
    assert proxy.eval() is proxy and proxy.to("cpu") is proxy
    assert next(proxy.parameters()) is next(model.parameters())
    with torch.no_grad():
        assert torch.equal(proxy(fake_batch(2)), model(fake_batch(2)))
    assert "num_prefix_tokens=3" in repr(proxy)


# ─────────────────────────────────────────────────────────────────────────────
# tools/ttr_validate.py — units
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_canon_tau_reads_the_committed_file():
    """The real committed file, so a key rename would be caught here."""
    tau, key = resolve_canon_tau(
        Path("results/diagsplit/fixed_thresholds_canon.json"),
        "vit_small", "mixup")
    assert key == "vit_small|mixup"
    assert tau == 20.8515625
    tau_b, _ = resolve_canon_tau(
        Path("results/diagsplit/fixed_thresholds_canon.json"),
        "vit_base", "mixup")
    assert tau_b == 127.3125


def test_resolve_canon_tau_refuses_a_missing_cell(tmp_path):
    f = tmp_path / "thr.json"
    f.write_text(json.dumps({"k": 5.0, "vit_small|mixup": 1.0}))
    with pytest.raises(KeyError, match="refusing to invent"):
        resolve_canon_tau(f, "vit_base", "nomix")


def test_infer_base_run_id():
    assert infer_base_run_id(
        Path("results/runs/e2r_vits_mixup_baseline_s1/ckpt/last.pth")
    ) == "e2r_vits_mixup_baseline_s1"
    with pytest.raises(ValueError, match="cannot infer a base run id"):
        infer_base_run_id(Path("results/legacy/ckpt/last.pth"))


class _Items:
    def __init__(self, items):
        self.items = items


def test_carve_subsets_is_disjoint_balanced_and_deterministic():
    items = [(f"val/c{c}/{i}.JPEG", c) for c in range(10) for i in range(10)]
    ev, fi = carve_subsets(_Items(items), eval_per_class=5, n_find=20)
    assert len(ev) == 50
    assert not (set(ev) & set(fi))
    labels = [items[i][1] for i in ev]
    assert {labels.count(c) for c in range(10)} == {5}
    # the find set spans many classes, not just the first ones
    assert len({items[i][1] for i in fi}) >= 8
    assert carve_subsets(_Items(items), 5, 20) == (ev, fi)


def test_carve_subsets_refuses_to_consume_the_whole_split():
    items = [(f"val/c{c}/{i}.JPEG", c) for c in range(3) for i in range(4)]
    with pytest.raises(ValueError, match="consumes the whole split"):
        carve_subsets(_Items(items), eval_per_class=4, n_find=5)


def test_verdict_pass_fail_and_undefined():
    unp = {"sink_fixed_canon": 10.0, "top1": 70.0}
    rows = [{"n_neurons": 4, "sink_fixed_canon": 9.0, "top1": 69.9},
            {"n_neurons": 8, "sink_fixed_canon": 2.0, "top1": 69.5},
            {"n_neurons": 16, "sink_fixed_canon": 0.5, "top1": 50.0}]
    v = verdict(unp, rows, 0.50, 1.00)
    assert v["passed"] and v["best_n_neurons"] == 8
    assert rows[0]["passes"] is False and rows[0]["meets_top1"] is True
    assert rows[1]["passes"] is True
    assert rows[2]["meets_sink"] is True and rows[2]["meets_top1"] is False
    assert rows[1]["sink_reduction_frac"] == pytest.approx(0.8)

    # accuracy preserved but sinks barely move => FAIL, and it says so
    v2 = verdict(unp, [{"n_neurons": 4, "sink_fixed_canon": 9.9,
                        "top1": 70.0}], 0.50, 1.00)
    assert not v2["passed"] and v2["best_n_neurons"] is None

    # no sinks to begin with => relative reduction undefined
    v3 = verdict({"sink_fixed_canon": 0.0, "top1": 70.0},
                 [{"n_neurons": 4, "sink_fixed_canon": 0.0, "top1": 70.0}],
                 0.50, 1.00)
    assert not v3["passed"] and "undefined" in v3["undefined_reason"]


def test_measure_reports_patched_prefix_and_constant_patch_count():
    model = tiny_model()
    loader = fake_loader(n_images=4, batch=2)
    plain = measure(model, loader, tau=1.0, device=torch.device("cpu"))
    assert plain["num_prefix_tokens"] == 1
    assert plain["n_images"] == 4
    assert 0.0 <= plain["top1"] <= 100.0
    assert 0.0 <= plain["sink_fixed_canon"] <= 196.0
    assert -1.0 <= plain["oversmooth_pairwise"] <= 1.0
    with apply_ttr(model, [(5, 3, 1.0)], n_extra_tokens=1) as patched:
        got = measure(patched, loader, tau=1.0, device=torch.device("cpu"))
    assert got["num_prefix_tokens"] == 2
    assert got["n_images"] == 4


# ─────────────────────────────────────────────────────────────────────────────
# tools/ttr_validate.py — end to end through the real CLI
# ─────────────────────────────────────────────────────────────────────────────

def _fake_imagenet(root, n_classes=2, per_class=3):
    items = []
    for c in range(n_classes):
        d = root / "val" / f"n{c:08d}"
        d.mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (16, 16),
                      (c * 40 % 256, i * 60 % 256, 90)).save(
                          d / f"img{i:02d}.JPEG")
            items.append([f"val/n{c:08d}/img{i:02d}.JPEG", c])
    return items


def test_ttr_validate_cli_end_to_end(tmp_path):
    """The real CLI on fake data: artifact set, schema, tau provenance, and
    an exit code that distinguishes PASS (0) from a measured FAIL (3)."""
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    items = _fake_imagenet(data)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(
        {"seed": 0, "n": len(items), "n_per_class": 3, "items": items}))

    thr = tmp_path / "thr.json"
    thr.write_text(json.dumps({"definition": "test", "k": 5.0,
                               "vit_small|mixup": 8.0}))

    from tools.model_factory import build_model
    torch.manual_seed(0)
    ckpt = tmp_path / "last.pth"
    torch.save({"model": build_model("vit_small", "baseline").state_dict()},
               ckpt)

    out_root = tmp_path / "ttr"
    proc = subprocess.run(
        [sys.executable, "tools/ttr_validate.py",
         "--ckpt", str(ckpt), "--arch", "vit_small", "--recipe", "mixup",
         "--n-neurons", "2,4", "--data", str(data),
         "--split-file", str(split), "--thr-file", str(thr),
         "--base-run-id", "e2r_vits_mixup_baseline_s1",
         "--out-root", str(out_root), "--eval-per-class", "2",
         "--n-find-images", "2", "--batch-size", "2", "--find-batch-size", "1",
         "--num-workers", "0", "--device", "cpu", "--top-n-stored", "8"],
        cwd=repo, capture_output=True, text=True)

    assert proc.returncode in (0, 3), proc.stdout + proc.stderr
    assert "TTR VALIDATION" in proc.stdout, proc.stdout + proc.stderr

    run_dir = out_root / "e2r_vits_mixup_baseline_s1"   # A2.1 layout
    neurons = load_neurons_json(run_dir / "neurons.json")
    assert neurons["outlier_tau"] == 8.0
    assert neurons["tau_key"] == "vit_small|mixup"
    assert neurons["arch"] == "vit_small" and neurons["variant"] == "baseline"
    assert len(neurons["neurons"]) <= 8
    assert neurons["ckpt_sha256"] and len(neurons["ckpt_sha256"]) == 64

    summary = json.loads((run_dir / "validate.json").read_text())
    assert summary["schema"] == "saga.ttr.validate/v1"
    # tau came from the base cell's file and was NOT recalibrated (acceptance)
    assert summary["canon_tau"] == 8.0
    assert summary["tau_recalibrated_on_patched_model"] is False
    assert summary["tau_source"] == str(thr)
    assert summary["n_eval_images"] == 4
    assert [r["n_neurons"] for r in summary["sweep"]] == [2, 4]
    assert summary["unpatched"]["num_prefix_tokens"] == 1
    assert all(r["num_prefix_tokens"] == 2 for r in summary["sweep"])
    assert summary["verdict"]["passed"] == (proc.returncode == 0)
    assert summary["criterion"] == CRITERION_MEAN_ABS

    rows = (run_dir / "sweep.csv").read_text().splitlines()
    assert rows[0].split(",")[0] == "n_neurons"
    assert [r.split(",")[0] for r in rows[1:]] == ["0", "2", "4"]


def test_ttr_validate_rejects_bad_sweep_values(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "tools/ttr_validate.py",
         "--ckpt", "x", "--arch", "vit_small", "--recipe", "mixup",
         "--n-neurons", "0,4", "--data", str(tmp_path)],
        cwd=repo, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "must be positive integers" in proc.stdout + proc.stderr


# ─────────────────────────────────────────────────────────────────────────────
# tools/ttr_prepare_run.py + tools/ttr_derive.py
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_run(tmp_path, neurons_json, n_neurons=2, arch="vit_small",
                 recipe="mixup", base="e2r_vits_mixup_baseline_s1"):
    repo = Path(__file__).resolve().parents[1]
    runs = tmp_path / "runs"
    proc = subprocess.run(
        [sys.executable, "tools/ttr_prepare_run.py",
         "--base-run-id", base, "--arch", arch, "--recipe", recipe,
         "--neurons-file", str(neurons_json), "--n-neurons", str(n_neurons),
         "--runs-root", str(runs)],
        cwd=repo, capture_output=True, text=True)
    return proc, runs / f"ttr_{base}"


def test_prepare_run_writes_a_config_the_canon_resolver_accepts(tmp_path):
    """The integration point A5.3 depends on: apply_fixed_thr --version canon
    must resolve a TTR diag file to the BASE cell's tau key."""
    _write_sample_json(tmp_path, top_n=8)
    proc, run_dir = _prepare_run(tmp_path, tmp_path / "neurons.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    import yaml
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    assert cfg["recipe"] == "mixup"        # the BASE cell's recipe_actual
    assert cfg["arch"] == "vit_small" and cfg["variant"] == "baseline"
    assert cfg["ttr"]["n_neurons"] == 2
    assert cfg["ttr"]["tau_recalibrated_on_patched_model"] is False
    assert cfg["ttr"]["base_run_id"] == "e2r_vits_mixup_baseline_s1"
    assert cfg["ttr"]["layers_touched"] == [5]
    assert len(cfg["ttr"]["neurons_file_sha256"]) == 64
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["run_id"] == "ttr_e2r_vits_mixup_baseline_s1"

    # the real resolver, on a diag path inside that run dir
    from tools.apply_fixed_thr import resolve_key
    key, reason = resolve_key(run_dir / "diag" / "diag_last.json",
                              "vit_small", "canon")
    assert key == "vit_small|mixup", reason


def test_prepare_run_refuses_arch_mismatch_and_oversized_n(tmp_path):
    _write_sample_json(tmp_path, top_n=2)
    proc, _ = _prepare_run(tmp_path, tmp_path / "neurons.json",
                           n_neurons=2, arch="vit_base")
    assert proc.returncode != 0
    assert "disagrees" in proc.stdout + proc.stderr

    proc, _ = _prepare_run(tmp_path, tmp_path / "neurons.json", n_neurons=99)
    assert proc.returncode != 0
    assert "exceeds" in proc.stdout + proc.stderr


def test_derive_refuses_a_legacy_shaped_diag_name(tmp_path):
    """A 7-part stem is read as a legacy e2 filename by
    apply_fixed_thr.recipe_from_stem, which would resolve the cell from the
    NAME instead of this run's config — refuse rather than mislabel."""
    _write_sample_json(tmp_path, top_n=4)
    _, run_dir = _prepare_run(tmp_path, tmp_path / "neurons.json")
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "tools/ttr_derive.py", "--ckpt", "x",
         "--arch", "vit_small", "--recipe", "mixup",
         "--neurons-file", str(tmp_path / "neurons.json"), "--n-neurons", "2",
         "--data", str(tmp_path), "--run-dir", str(run_dir),
         "--diag-name", "e2_vit_small_nomix_saga_rlast_last"],
        cwd=repo, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "7 underscore-separated" in proc.stdout + proc.stderr


def test_derive_refuses_a_run_dir_without_config(tmp_path):
    _write_sample_json(tmp_path, top_n=4)
    bare = tmp_path / "runs" / "ttr_nope"
    bare.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "tools/ttr_derive.py", "--ckpt", "x",
         "--arch", "vit_small", "--recipe", "mixup",
         "--neurons-file", str(tmp_path / "neurons.json"), "--n-neurons", "2",
         "--data", str(tmp_path), "--run-dir", str(bare)],
        cwd=repo, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "config.resolved.yaml is missing" in proc.stdout + proc.stderr


def _real_neurons_json(tmp_path, model, loader, tau=9.0, top_n=8):
    ranking = find_register_neurons(model, loader, None, None,
                                    outlier_tau=tau, seed=0, max_images=4)
    write_neurons_json(
        tmp_path / "neurons.json", ranking, criterion=CRITERION_MEAN_ABS,
        outlier_tau=tau, tau_key="vit_small|mixup", tau_source="test",
        seed=0, arch="vit_small", variant="baseline", ckpt="/x/last.pth",
        ckpt_sha256="cd" * 32,
        scan_stats=find_register_neurons.last_scan_stats, top_n_stored=top_n)
    return tmp_path / "neurons.json"


def test_derive_unpatched_path_reproduces_tools_eval(tmp_path, monkeypatch):
    """[claim] ttr_derive's eval loop == tools/eval.py's, so no arithmetic
    was forked. --n-neurons 0 makes apply_ttr a no-op, giving the unpatched
    number both tools must agree on to the last digit.

    Both modules' 50 000-image guard is lowered to the fake set's size; that
    assert is the only thing standing between them and a small dataset.
    """
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    items = _fake_imagenet(data, n_classes=2, per_class=3)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(
        {"seed": 0, "n": len(items), "n_per_class": 3, "items": items}))

    from tools.model_factory import build_model
    torch.manual_seed(0)
    ckpt = tmp_path / "last.pth"
    torch.save({"model": build_model("vit_small", "baseline").state_dict()},
               ckpt)

    neurons = _real_neurons_json(tmp_path, tiny_model(),
                                 fake_loader(n_images=4, batch=2))
    _, run_dir = _prepare_run(tmp_path, neurons, n_neurons=0)

    shared = ["--ckpt", str(ckpt), "--arch", "vit_small",
              "--data", str(data), "--device", "cpu", "--num-workers", "0",
              "--batch-size", "2"]

    import tools.eval as eval_mod
    import tools.ttr_derive as derive_mod
    monkeypatch.setattr(eval_mod, "IMAGENET_VAL_SIZE", len(items))
    monkeypatch.setattr(derive_mod, "IMAGENET_VAL_SIZE", len(items))
    monkeypatch.chdir(repo)

    ref_out = tmp_path / "ref.json"
    monkeypatch.setattr(sys, "argv", ["eval.py", *shared,
                                      "--variant", "baseline",
                                      "--out", str(ref_out)])
    eval_mod.main()

    monkeypatch.setattr(sys, "argv", [
        "ttr_derive.py", *shared, "--recipe", "mixup",
        "--neurons-file", str(neurons), "--n-neurons", "0",
        "--split-file", str(split), "--run-dir", str(run_dir),
        "--no-attn", "--n-effrank", "2", "--diag-batch-size", "2"])
    derive_mod.main()

    ref = json.loads(ref_out.read_text())
    got = json.loads((run_dir / "eval" / "eval_last.json").read_text())
    assert got["top1"] == ref["top1"]
    assert got["top5"] == ref["top5"]
    assert got["loss"] == ref["loss"]
    assert got["n_images"] == ref["n_images"] == len(items)
    assert got["ckpt_sha256"] == ref["ckpt_sha256"]
    # an empty selection is a no-op, so the patched prefix count stays 1
    diag = json.loads((run_dir / "diag" / "diag_last.json").read_text())
    assert diag["num_prefix_tokens"] == 1
    assert diag["ttr"]["n_neurons"] == 0
    assert diag["ttr"]["tau_recalibrated_on_patched_model"] is False


def test_derive_patched_writes_widened_prefix_and_canon_backfills(
        tmp_path, monkeypatch):
    """The full A5.3 chain: derive a PATCHED model, then let the EXISTING
    apply_fixed_thr --version canon put the BASE cell's tau on it."""
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    items = _fake_imagenet(data, n_classes=2, per_class=3)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(
        {"seed": 0, "n": len(items), "n_per_class": 3, "items": items}))

    from tools.model_factory import build_model
    torch.manual_seed(0)
    ckpt = tmp_path / "last.pth"
    torch.save({"model": build_model("vit_small", "baseline").state_dict()},
               ckpt)

    neurons = _real_neurons_json(tmp_path, tiny_model(),
                                 fake_loader(n_images=4, batch=2))
    _, run_dir = _prepare_run(tmp_path, neurons, n_neurons=3)

    import tools.ttr_derive as derive_mod
    monkeypatch.setattr(derive_mod, "IMAGENET_VAL_SIZE", len(items))
    monkeypatch.chdir(repo)
    monkeypatch.setattr(sys, "argv", [
        "ttr_derive.py", "--ckpt", str(ckpt), "--arch", "vit_small",
        "--recipe", "mixup", "--neurons-file", str(neurons),
        "--n-neurons", "3", "--data", str(data), "--split-file", str(split),
        "--run-dir", str(run_dir), "--device", "cpu", "--num-workers", "0",
        "--batch-size", "2", "--diag-batch-size", "2", "--no-attn",
        "--n-effrank", "2"])
    derive_mod.main()

    diag_path = run_dir / "diag" / "diag_last.json"
    diag = json.loads(diag_path.read_text())
    assert diag["num_prefix_tokens"] == 2         # 1 CLS + 1 extra token
    assert diag["ttr"]["n_neurons"] == 3
    assert diag["sink_fixed_thr"] is None         # no v1 field written
    assert "sink_fixed_canon" not in diag         # that is the backfill's job
    npz = run_dir / "diag" / "diag_last_norms.npz"
    assert npz.exists()
    with np.load(npz) as z:
        assert z["last_block_patch_norms"].shape == (len(items), 196)

    # idempotency: a requeue redoes nothing
    before = diag_path.read_text()
    derive_mod.main()
    assert diag_path.read_text() == before

    # now the EXISTING canon backfill, unmodified, on the TTR run dir
    thr = tmp_path / "canon.json"
    thr.write_text(json.dumps({"definition": "test", "k": 5.0,
                               "vit_small|mixup": 9.0}))
    empty = tmp_path / "empty_diag"
    empty.mkdir()
    proc = subprocess.run(
        [sys.executable, "tools/apply_fixed_thr.py", "--version", "canon",
         "--thr-file", str(thr), "--diag-dir", str(empty),
         "--runs-root", str(run_dir.parent)],
        cwd=repo, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    after = json.loads(diag_path.read_text())
    assert after["canon_thr_value"] == 9.0
    assert 0.0 <= after["sink_fixed_canon"] <= 196.0
    assert after["num_prefix_tokens"] == 2        # untouched by the backfill
    assert after["ttr"]["n_neurons"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# scripts/jobs/ttr_*.sbatch — the Phase-B launchers
# ─────────────────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parents[1]
JOBS = REPO / "scripts" / "jobs"
TTR_JOBS = ["ttr_validate.sbatch", "ttr_matrix.sbatch"]


def _job(name):
    return (JOBS / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", TTR_JOBS)
def test_job_header_matches_the_established_a40_pattern(name):
    """The a40 strings are the human's, via TASK-11's committed job file —
    not invented here. If that file's header changes, this catches the drift."""
    ref = _job("probe_attention.sbatch")
    src = _job(name)
    for line in ["#SBATCH --partition=a40",
                 "#SBATCH --gres=gpu:a40:1",
                 "#SBATCH --cpus-per-task=4"]:
        assert line in ref, f"{line} vanished from probe_attention.sbatch"
        assert line in src, f"{line} missing from {name}"
    assert "#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/" in src
    assert "#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/" in src


@pytest.mark.parametrize("name", TTR_JOBS)
def test_job_is_lf_only_and_sets_u(name):
    """A CRLF sbatch fails on the HPC with cryptic errors (.gitattributes
    forces LF; this asserts the working copy too)."""
    raw = (JOBS / name).read_bytes()
    assert b"\r\n" not in raw, f"{name} has CRLF line endings"
    assert raw.startswith(b"#!/bin/bash")
    assert b"\nset -u\n" in raw


@pytest.mark.parametrize("name", TTR_JOBS)
def test_job_releases_scratch_on_every_exit_path(name):
    """TASK-09's lesson: a plain cleanup call at the end only runs on a
    normal exit, so a wall-clock kill leaves the extracted copy behind."""
    src = _job(name)
    assert "trap cleanup_probe EXIT" in src
    assert src.index("trap cleanup_probe EXIT") < \
        src.index("stage_probe_imagenet_val\n"), \
        "the trap must be installed before staging returns"
    # exactly once, so it cannot double-run (the trap is the ONLY caller;
    # the function itself is defined in scripts/stage_probe.sh)
    assert src.count("cleanup_probe") == 1


@pytest.mark.parametrize("name", TTR_JOBS)
def test_job_sources_env_and_stager_by_absolute_path(name):
    src = _job(name)
    assert ("source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh"
            in src)
    assert "source $CODE_ROOT/scripts/stage_probe.sh" in src
    assert "--data $PROBE_IN_DIR" in src


def test_matrix_job_requires_n_best():
    """Submitting without N_BEST must fail loudly, not silently pick a value."""
    src = _job("ttr_matrix.sbatch")
    assert ': "${N_BEST:?' in src
    assert "N_BEST=16 sbatch" in src        # the usage line is spelled out


def test_matrix_job_backfills_canon_before_sink_address():
    """sink_address.py hard-checks its canon mass against the sibling diag's
    sink_fixed_canon, which apply_fixed_thr writes — so the order is load
    bearing, not cosmetic."""
    src = _job("ttr_matrix.sbatch")
    i_apply = src.index("tools/apply_fixed_thr.py")
    i_addr = src.index("tools/sink_address.py")
    assert i_apply < i_addr
    assert "--version canon" in src
    assert "fixed_thresholds_canon.json" in src
    # never a tau recalibrated on the patched model
    assert "compute_fixed_thr" not in src


def test_matrix_job_covers_the_four_A5_3_cells_with_manifest_paths():
    """The four base cells of TASK-10 A5.3, and the legacy path checked
    against the committed manifest rather than typed from memory."""
    src = _job("ttr_matrix.sbatch")
    for run_id, arch, recipe in [
            ("e2r_vits_mixup_baseline_s1", "vit_small", "mixup"),
            ("e2r_vitb_mixup_baseline_s1", "vit_base", "mixup"),
            ("e2r_vits_nomix_baseline_s1", "vit_small", "nomix"),
            ("legacy_vits_baseline", "vit_small", "mixup")]:
        assert f'"{run_id}|{arch}|{recipe}|' in src, run_id

    import csv
    legacy = ("/home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints/"
              "ViT-S_baseline/last.pth")
    assert legacy in src
    with open(REPO / "results" / "legacy" / "checkpoint_manifest.csv") as f:
        rows = list(csv.DictReader(f))
    row = [r for r in rows if r["path"] == legacy]
    assert len(row) == 1, f"{legacy} is not a unique manifest row"
    assert row[0]["arch"] == "vit_small"
    assert row[0]["variant"] == "baseline"
    # recipe_actual is what decides the canon tau key, and the job says mixup
    assert row[0]["recipe_actual"] == "mixup"


def test_validate_job_runs_the_gate_and_says_not_to_proceed_on_fail():
    src = _job("ttr_validate.sbatch")
    assert "tools/ttr_validate.py" in src
    # the grid is now a variable; its default is pinned by
    # test_validate_job_sweep_grid_is_overridable_with_a_safe_default
    assert '--n-neurons "$N_NEURONS"' in src
    assert "e2r_vits_mixup_baseline_s1" in src
    assert "--arch vit_small --recipe mixup" in src
    assert "Do NOT submit ttr_matrix.sbatch unless this said PASS." in src
    assert "exit $RC" in src                 # the PASS/FAIL code propagates
    # the gate job must NOT derive anything itself
    assert "tools/ttr_derive.py" not in src


@pytest.mark.parametrize("name", TTR_JOBS)
def test_job_has_no_wildcard_or_brace_submit_shape(name):
    """TASK-09 verified that a globbed/braced submit runs only the FIRST
    script and passes the rest as ignored positional arguments."""
    src = _job(name)
    for bad in ["sbatch scripts/jobs/ttr_*", "sbatch scripts/jobs/*.sbatch",
                "submit_ttr_*", "ttr_{"]:
        assert bad not in src, f"{name} contains the shape {bad!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Repo-wide: `set -u` must never precede the env sourcing
# ─────────────────────────────────────────────────────────────────────────────

def _all_job_files():
    """Every committed sbatch under scripts/, however it is named."""
    return sorted(REPO.glob("scripts/**/*.sbatch"))


def test_there_are_job_files_to_check():
    """Guard against the glob silently matching nothing."""
    files = _all_job_files()
    assert len(files) >= 20, f"only found {len(files)} sbatch files"


@pytest.mark.parametrize(
    "job", _all_job_files(), ids=lambda p: p.name)
def test_set_u_never_precedes_the_env_sourcing(job):
    """`set -u` above `source .../env_alex.sh` kills the job during env setup.

    env_alex.sh sources /etc/profile, which runs the site scripts in
    /etc/profile.d/. At least one of them dereferences an unset variable
    (debuginfod.sh line 8: DEBUGINFOD_URLS), and under `set -u` bash aborts
    the CALLING script — so not one line of the job body runs and the only
    thing in the .err is that single message.

    Measured on Alex 2026-09-10 by TASK-10's ttr_validate job. The bug was
    inherited by copying probe_attention.sbatch's header; every job file that
    had actually completed a run (TASK-08 ft_*, TASK-09 det_*/seg_*/
    dense_smoke_*) already had the safe ordering. This test exists so the
    next job file cannot repeat it.
    """
    lines = job.read_text(encoding="utf-8").splitlines()

    def first(pred):
        return next((i for i, l in enumerate(lines) if pred(l)), None)

    i_setu = first(lambda l: l.strip() == "set -u")
    i_src = first(lambda l: l.strip().startswith("source ")
                  and "env_alex.sh" in l)
    if i_setu is None or i_src is None:
        pytest.skip(f"{job.name}: set -u={i_setu}, env source={i_src}")
    assert i_setu > i_src, (
        f"{job.name}: `set -u` is on line {i_setu + 1}, above the env "
        f"sourcing on line {i_src + 1}. Move it below — /etc/profile's site "
        f"scripts reference unset variables and abort the job under set -u.")


def test_no_job_file_name_contains_a_space():
    """A space in an sbatch name breaks unquoted submission: `sbatch
    scripts/jobs/ttr_validate copy.sbatch` splits into two arguments, sbatch
    opens neither, and no job is ever queued. It also breaks plain shell
    loops over the job files."""
    bad = [p.name for p in _all_job_files() if " " in p.name]
    assert not bad, f"sbatch names with spaces: {bad}"


# ─────────────────────────────────────────────────────────────────────────────
# The interpreter preflight must be absolute, fatal, and before staging
# ─────────────────────────────────────────────────────────────────────────────

CONDA_PY = "/home/vault/iwi5/iwi5359h/envs/saga/bin/python"


@pytest.mark.parametrize("name", TTR_JOBS)
def test_job_uses_the_env_interpreter_by_absolute_path(name):
    """`which python` silently falls back to /usr/bin/python when
    env_alex.sh's `module load` fails — measured on Alex 2026-09-10, job
    4211911, which then had no torch. The absolute env path needs no module;
    it is the repo's own committed pattern (env_alex.sh's TORCHRUN line)."""
    src = _job(name)
    assert "PY=$(which python)" not in src, (
        f"{name}: `which python` resolves to /usr/bin/python when the conda "
        f"module is unavailable")
    assert f"PY=${{SAGA_PY:-{CONDA_PY}}}" in src
    # the path must match the one env_alex.sh itself uses for TORCHRUN
    env_alex = (REPO / "scripts" / "env_alex.sh").read_text(encoding="utf-8")
    assert CONDA_PY in env_alex, (
        "scripts/env_alex.sh no longer references this interpreter path — "
        "the two must not drift apart")


@pytest.mark.parametrize("name", TTR_JOBS)
def test_torch_preflight_is_fatal_and_precedes_staging(name):
    """An unchecked `$PY -c "import torch"` printed its own
    ModuleNotFoundError and let the job stage 50 000 images anyway. The
    check must gate, and it must gate BEFORE staging."""
    lines = _job(name).splitlines()

    def first(pred):
        return next((i for i, l in enumerate(lines) if pred(l)), None)

    i_check = first(lambda l: l.startswith("if ! $PY -c")
                    and "import torch" in l)
    i_exec = first(lambda l: l.strip() == "-x \"$PY\" ]" or
                   l.strip().startswith("if [ ! -x \"$PY\" ]"))
    i_stage = first(lambda l: l.strip() == "stage_probe_imagenet_val")
    assert i_check is not None, f"{name}: the torch import is not gated by `if !`"
    assert i_exec is not None, f"{name}: no executable check on $PY"
    assert i_stage is not None, f"{name}: staging call not found"
    assert i_exec < i_check < i_stage, (
        f"{name}: order must be -x check ({i_exec}) -> torch check "
        f"({i_check}) -> staging ({i_stage})")
    # each guard must actually abort
    for i in (i_exec, i_check):
        block = "\n".join(lines[i:i + 8])
        assert "exit 1" in block, (
            f"{name}: the guard at line {i + 1} does not exit")


@pytest.mark.parametrize("name", TTR_JOBS)
def test_preflight_runs_before_the_cleanup_trap_is_armed(name):
    """Exiting from the preflight must not fire a cleanup for a stage dir
    that was never created."""
    lines = _job(name).splitlines()
    i_check = next(i for i, l in enumerate(lines)
                   if l.startswith("if ! $PY -c") and "import torch" in l)
    i_trap = next(i for i, l in enumerate(lines)
                  if l.strip() == "trap cleanup_probe EXIT")
    assert i_check < i_trap


# ─────────────────────────────────────────────────────────────────────────────
# The conda module name churned mid-project; live env scripts must tolerate it
# ─────────────────────────────────────────────────────────────────────────────

#: The env scripts every CURRENT job file sources. The legacy launchers
#: (detection/scripts/e3_eval_tinyx.sh, evaluation/e5_lost/…,
#: evaluation/e6_finegrained/e6_*.sh) still name the dead module and are
#: deliberately NOT fixed: they are superseded provenance (TASK-08 replaced
#: e6, TASK-09 banner-marked e3/e4) and must not be silently made runnable.
LIVE_ENV_SCRIPTS = [
    "scripts/env_alex.sh",
    "detection/scripts/env_alex.sh",
    "segmentation/scripts/env_alex.sh",
    "classification/scripts/env_alex.sh",
]


@pytest.mark.parametrize("rel", LIVE_ENV_SCRIPTS)
def test_live_env_script_tolerates_the_module_rename(rel):
    """'python/3.12-conda' resolved on 2026-09-08 and was gone by 2026-09-10,
    which broke every job in the repo: the load failed, `source activate`
    failed, and jobs ran /usr/bin/python with no torch. A single hardcoded
    module name is therefore a known failure mode, not a hypothetical."""
    src = (REPO / rel).read_text(encoding="utf-8")
    assert "module load python/3.12-conda" in src, (
        f"{rel}: the pinned name should still be TRIED first so a restored "
        f"modulefile is preferred")
    assert "elif module load python 2>/dev/null" in src, (
        f"{rel}: no fallback to the generic 'module load python', which is "
        f"what works on the cluster as of 2026-09-10")
    # and it must not fail silently
    assert "WARNING - no python module loaded" in src, (
        f"{rel}: a failed module load must be announced, not swallowed")
    assert "/home/vault/iwi5/iwi5359h/envs/saga" in src


@pytest.mark.parametrize("rel", LIVE_ENV_SCRIPTS)
def test_live_env_script_is_syntactically_valid(rel):
    """These are sourced by every job; a syntax error here kills all of them."""
    import shutil
    import subprocess
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash available")
    proc = subprocess.run([bash, "-n", str(REPO / rel)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"{rel}: {proc.stderr}"


def test_sync_results_hint_does_not_tell_the_human_a_dead_command():
    """sync_results.sh prints an 'activate the env first' hint when it cannot
    determine a run's state; that hint named the removed module."""
    src = (REPO / "scripts" / "sync_results.sh").read_text(encoding="utf-8")
    assert "module load python &&" in src
    assert "module load python/3.12-conda &&" not in src


def test_validate_job_sweep_grid_is_overridable_with_a_safe_default():
    """The A2.3 default grid is a factor-of-2 ladder, and job 4212808 showed
    the sink/accuracy transition falling inside the 8->16 gap. Refining the
    GRID must be possible without editing the file (a measurement fix);
    changing the PASS thresholds must not be, so that a FAIL cannot be turned
    into a PASS by a command-line flag in the launcher."""
    src = _job("ttr_validate.sbatch")
    assert "N_NEURONS=${N_NEURONS:-4,8,16,32,64}" in src
    assert '--n-neurons "$N_NEURONS"' in src
    # The verdict thresholds stay at the tool's defaults. Check the EXECUTED
    # lines, not the whole text — the comment above legitimately names both
    # flags while explaining why they are not set.
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "--min-sink-reduction" not in code
    assert "--max-top1-drop" not in code
    # and a rerun clobbers the previous sweep, so the file says so
    assert "COMMIT the previous sweep" in src
