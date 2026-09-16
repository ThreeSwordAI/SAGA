"""
tests/test_I0_manifest.py
=========================
TASK I0 D7 — the cohort manifest.

Two kinds of test here, and the difference matters:

  * tests against the COMMITTED manifest, which pin the facts TASK I0 §3
    says the manifest must reproduce (19 completed 300-epoch conditions,
    8 baselines / 8 SAGA / 3 registers, the VOID trio excluded, prefix
    counts 1 and 5);
  * tests against synthetic run directories in tmp_path, which pin the
    BEHAVIOUR (a mislabelled recipe is refused, a diagnostics seed is never
    read, an incomplete run is invalid).

`eligibility.md` is re-rendered and compared byte for byte — the project
convention for a generated note.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import analysis.build_I0_manifest as bm
from analysis.build_pooled_tables import E2R_MEMBERS, LEGACY_MEMBERS, VARIANTS
from saga.frozen.stages import HIST_STAGE

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "frozen" / "I0_manifest"
MANIFEST_CSV = OUT / "manifest.csv"
MANIFEST_JSON = OUT / "manifest.json"
ELIGIBILITY = OUT / "eligibility.md"
HASHES = OUT / "ckpt_hashes.json"
MISSING = "MISSING"


@pytest.fixture(scope="module")
def rows():
    with open(MANIFEST_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def cohort(rows):
    """The 300-epoch classification cohort: one row per completed condition."""
    return [r for r in rows
            if r["family"] in ("e2r_300ep", "legacy_300ep")
            and r["ckpt_kind"] == "last"
            and r["status"] in ("eligible", "eligible_legacy")]


# ── the pinned facts of TASK I0 §3 ──────────────────────────────────────────

def test_the_cohort_has_nineteen_completed_conditions(cohort):
    assert len(cohort) == 19


def test_the_cohort_breakdown_is_eight_eight_three(cohort):
    """8 baselines (4 ViT-S mixup, 2 ViT-S true-nomix, 2 ViT-B mixup),
    8 matching SAGA, 3 registers (2 ViT-S mixup, 1 ViT-B mixup)."""
    by_variant = {}
    for r in cohort:
        by_variant.setdefault(r["variant"], []).append(r)
    assert {k: len(v) for k, v in sorted(by_variant.items())} == {
        "baseline": 8, "registers": 3, "saga": 8}

    def cells(variant):
        out = {}
        for r in by_variant[variant]:
            key = (r["arch"], r["recipe_actual"])
            out[key] = out.get(key, 0) + 1
        return out

    expect_8 = {("vit_small", "mixup"): 4, ("vit_small", "nomix"): 2,
                ("vit_base", "mixup"): 2}
    assert cells("baseline") == expect_8
    assert cells("saga") == expect_8
    assert cells("registers") == {("vit_small", "mixup"): 2,
                                  ("vit_base", "mixup"): 1}


def test_every_register_row_is_legacy(cohort):
    regs = [r for r in cohort if r["variant"] == "registers"]
    assert regs and all(r["family"] == "legacy_300ep" for r in regs)


def test_fresh_pairs_are_seed_controlled_and_legacy_repeats_are_not(cohort):
    for r in cohort:
        if r["family"] == "e2r_300ep":
            assert r["seed_controlled"] == "1", r["run_id"]
            assert r["seed"] in ("1", "2"), r["run_id"]
        else:
            assert r["seed_controlled"] == "0", r["run_id"]
            assert r["seed"] == MISSING, r["run_id"]
            assert "recorded no seed" in r["seed_source"]


def test_the_void_vitb_mixupdir_trio_is_invalid_and_never_eligible(rows):
    """The legacy ViT-B mixup-DIRECTORY trio never finished training. It must
    appear (so it is accounted for) and must never be eligible."""
    void = [r for r in rows
            if r["family"] == "legacy_300ep" and r["arch"] == "vit_base"
            and r["recipe_dirname"] == "mixup"]
    assert len(void) == 6                       # 3 conditions x {last, best}
    assert {r["variant"] for r in void} == set(VARIANTS)
    for r in void:
        assert r["status"] == "invalid", r["run_id"]
        assert "VOID" in r["status_reason"]
        assert r["recipe_actual"] == MISSING     # member of no cell
    assert not any(r["status"].startswith("eligible") for r in void)


def test_the_incomplete_vitb_s2_runs_are_invalid(rows):
    bad = [r for r in rows if r["run_id"].endswith("_s2")
           and r["arch"] == "vit_base" and r["family"] == "e2r_300ep"]
    assert len(bad) == 4                        # 2 runs x {last, best}
    for r in bad:
        assert r["status"] == "invalid"
        assert "training incomplete" in r["status_reason"]


def test_ablation_is_a_separate_family_never_mixed_with_300ep(rows):
    """The 6 arms, 100 epochs, seed 0, checkpoints on the woody abl_ckpt
    root recorded in each meta.json."""
    abl = [r for r in rows if r["family"] == "ablation_100ep"
           and r["ckpt_kind"] == "last"]
    assert len(abl) == 6
    for r in abl:
        assert r["status"] == "eligible"
        assert r["epochs_completed"] == "100"
        assert r["seed"] == "0"
        assert "abl_ckpt" in r["ckpt_path"], r["run_id"]
        assert r["ckpt_dir_source"] == 'meta.json["ckpt_dir"]'
        assert "separate stratum" in r["status_reason"].lower()
    assert {r["gate_mode"] for r in abl} == {
        "none", "const", "headscalar", "layerscale", "spatial"}
    # arm F is the +4 init; its gate_init_logit comes from meta.json knobs
    assert {r["gate_init_logit"] for r in abl} == {"0.0", "4.0"}


def test_finetune_family_has_24_runs_and_best_is_canonical(rows):
    ft = [r for r in rows if r["family"] == "finetune"]
    best = [r for r in ft if r["ckpt_kind"] == "best"]
    last = [r for r in ft if r["ckpt_kind"] == "last"]
    assert len(best) == 24 and len(last) == 24
    for r in best:
        assert r["status"] == "eligible"
        assert "selects on the frozen val split" in r["status_reason"]
        assert r["base_run_id"].startswith("e2r_")
        assert len(r["base_ckpt_sha256"]) == 64
    for r in last:
        assert r["status"] == "superseded"


def test_every_other_family_uses_last_and_records_best_as_superseded(rows):
    for family in ("e2r_300ep", "legacy_300ep", "ablation_100ep",
                   "dense_det", "dense_seg"):
        best = [r for r in rows if r["family"] == family
                and r["ckpt_kind"] == "best"]
        assert best, family
        assert not any(r["status"].startswith("eligible") for r in best), family


def test_dense_rows_name_the_files_the_trainers_actually_write(rows):
    """TASK I0 §3: "record what weight files actually exist per run". The
    Phase-B hash pass found six `best.pth` paths that never existed, so the
    filenames are now read from the trainers and pinned here against them.

      detection    saves ckpt/last.pth ONLY — its best-AP state is the JSON
                   pair coco_eval_best.json + detections_val.json, which is
                   exactly the case §3 warned about.
      segmentation saves ckpt/last.pth and ckpt/best_model.pth.
    """
    det_src = (REPO / "detection" / "tools" / "train.py").read_text(
        encoding="utf-8")
    seg_src = (REPO / "segmentation" / "tools" / "train.py").read_text(
        encoding="utf-8")
    assert 'ckpt_dir / "best_model.pth"' in seg_src
    assert 'ckpt_dir / "best.pth"' not in seg_src
    assert 'ckpt_dir / "best.pth"' not in det_src
    assert 'ckpt_dir / "best_model.pth"' not in det_src
    assert bm.DENSE_CKPT_FILENAME == {
        "dense_det": {"last": "last.pth", "best": None},
        "dense_seg": {"last": "last.pth", "best": "best_model.pth"}}

    det = [r for r in rows if r["family"] == "dense_det"]
    seg = [r for r in rows if r["family"] == "dense_seg"]
    assert len(det) == 6 and len(seg) == 6
    for r in det:
        # detection writes no best file; its `best` row either resolves to
        # last.pth (when the epochs coincide) or names nothing at all
        assert r["ckpt_path"].endswith("/ckpt/last.pth") \
            or r["ckpt_path"] == MISSING, r["run_id"]
        assert "no best checkpoint exists" in r["status_reason"] \
            or r["ckpt_kind"] == "last"
    for r in seg:
        want = "best_model.pth" if r["ckpt_kind"] == "best" else "last.pth"
        assert r["ckpt_path"].endswith(f"/ckpt/{want}"), r["run_id"]
    # the filename that never existed appears nowhere
    assert not any(r["ckpt_path"].endswith("/ckpt/best.pth")
                   for r in det + seg)


def test_detection_best_rows_resolve_to_last_pth_only_when_the_epochs_agree(
        rows):
    """Detection saves no best checkpoint. The human's call (2026-09-16) is
    that a `best` row RESOLVES to `ckpt/last.pth` rather than staying empty —
    but only where the run's own files prove that is the same thing.

    The proof is per run and re-derived on every build: `coco_eval_best.json`
    gives the best-AP epoch and `log.csv` the last one. Where they coincide,
    last.pth IS the best-AP weights. Where they would not, the row must NOT
    resolve — the best weights were genuinely never saved. TASK-09's defect
    was that one could not tell which case applied; here it is computed.

    The cost of resolving is a shared sha256 across two ckpt_kinds, so the
    row declares `resolves_to_ckpt_kind` and any count must de-duplicate.
    """
    det_best = [r for r in rows
                if r["family"] == "dense_det" and r["ckpt_kind"] == "best"]
    det_last = {r["run_id"]: r for r in rows
                if r["family"] == "dense_det" and r["ckpt_kind"] == "last"}
    assert len(det_best) == 3 and len(det_last) == 3

    for r in det_best:
        p = json.loads(r["derived_params"])
        run = REPO / "results" / "detection" / r["run_id"]
        best = json.loads((run / "coco_eval_best.json").read_text(
            encoding="utf-8"))
        with open(run / "log.csv", newline="", encoding="utf-8") as f:
            last_ep = max(int(x["epoch"]) for x in csv.DictReader(f)
                          if x.get("AP"))
        # the epochs are READ from the run's own files, never typed
        assert p["best_ap_epoch"] == best.get("epoch", best.get("best_epoch"))
        assert p["last_epoch"] == last_ep
        assert p["best_weights_are_last_pth"] == (p["best_ap_epoch"] == last_ep)
        assert str(p["best_ap_epoch"]) in r["status_reason"]

        sibling = det_last[r["run_id"]]
        if p["best_weights_are_last_pth"]:
            # resolved: SAME file, SAME sha, and it says so
            assert p["resolves_to_ckpt_kind"] == "last", r["run_id"]
            assert r["ckpt_path"] == sibling["ckpt_path"], r["run_id"]
            assert r["ckpt_sha256"] == sibling["ckpt_sha256"], r["run_id"]
            assert len(r["ckpt_sha256"]) == 64, r["run_id"]
            assert "RESOLVED TO ckpt/last.pth" in r["status_reason"]
            assert "de-duplicate on sha256" in r["status_reason"]
        else:
            # not resolvable: the best weights were never written
            assert p["resolves_to_ckpt_kind"] is None, r["run_id"]
            assert r["ckpt_path"] == MISSING, r["run_id"]
            assert r["ckpt_sha256"] == MISSING, r["run_id"]
            assert "were never saved" in r["status_reason"]

    # only ONE of the two rows is eligible, so a count of eligible dense
    # checkpoints cannot double-count a run
    for r in det_best:
        assert r["status"] == "superseded", r["run_id"]
    for r in det_last.values():
        assert r["status"] == "eligible", r["run_id"]
    assert len({r["ckpt_sha256"] for r in det_last.values()}) == 3


def test_dense_rows_record_their_backbone(rows):
    dense = [r for r in rows if r["family"] in ("dense_det", "dense_seg")]
    assert dense
    for r in dense:
        # every dense run names the backbone it was built on. The registers
        # runs name the LEGACY fallback, not an e2r run: configs/
        # dense_matrix.yaml records that e2r_vitb_mixup_registers_s1 was never
        # trained, so `resolve_backbone` took the documented fallback — that
        # is a recorded fact about the cohort, not a gap.
        assert r["base_run_id"] != MISSING, r["run_id"]
        assert (r["base_run_id"].startswith("e2r_")
                or r["variant"] == "registers"), r["run_id"]
        assert len(r["base_ckpt_sha256"]) == 64, r["run_id"]
    fallback = [r for r in dense if r["variant"] == "registers"]
    assert fallback
    assert all(r["base_run_id"].startswith("legacy_") for r in fallback)


def test_ttr_rows_are_derived_edits_not_models(rows):
    ttr = [r for r in rows if r["family"] == "ttr_edit"]
    assert len(ttr) == 4
    for r in ttr:
        assert r["status"] == "derived"
        assert r["ckpt_kind"] == "derived"
        assert r["ckpt_path"] == MISSING
        assert r["base_run_id"] != MISSING
        params = json.loads(r["derived_params"])
        assert params["n_neurons"]
        assert params["layer_range"]
        assert len(params["neurons_file_sha256"]) == 64
        # the TTR seed is the EDIT's, never a training seed
        assert "NOT a training seed" in r["seed_source"]
        assert r["seed_controlled"] == "0"


def test_prefix_counts_are_one_everywhere_except_registers(rows):
    for r in rows:
        if r["n_prefix"] == MISSING:
            continue
        if r["variant"] == "registers":
            assert r["n_prefix"] == "5", r["run_id"]
        elif r["family"] == "ttr_edit":
            # TTR prepends n_extra_tokens to the base model's single CLS
            params = json.loads(r["derived_params"])
            assert int(r["n_prefix"]) == 1 + int(params["n_extra_tokens"])
        else:
            assert r["n_prefix"] == "1", r["run_id"]


def test_gate_param_counts_agree_with_the_ablation_table(rows):
    """The manifest builds its own model; TASK-12's table builds its own.
    They must not drift, so the two are compared directly."""
    from analysis.build_ablation_tables import gate_param_counts
    seen = 0
    for r in rows:
        if r["family"] != "ablation_100ep" or r["ckpt_kind"] != "last":
            continue
        arch_full = {"vit_small": "vit_small_patch16_224",
                     "vit_base": "vit_base_patch16_224"}[r["arch"]]
        reg, tr = gate_param_counts(arch_full, r["gate_mode"])
        assert int(r["gate_params_registered"]) == reg, r["run_id"]
        assert int(r["gate_params_trainable"]) == tr, r["run_id"]
        seen += 1
    assert seen == 6


def test_hist_stage_is_the_identified_stage_and_is_set_where_diag_exists(rows):
    with_diag = [r for r in rows if r["hist_stage"] != MISSING]
    assert with_diag
    assert {r["hist_stage"] for r in with_diag} == {HIST_STAGE}
    # every eligible 300-epoch condition has historical diagnostics
    for r in rows:
        if (r["family"] in ("e2r_300ep", "legacy_300ep")
                and r["ckpt_kind"] == "last"
                and r["status"].startswith("eligible")):
            assert r["hist_stage"] == HIST_STAGE, r["run_id"]
            assert len(r["hist_split_sha"]) == 64, r["run_id"]
            assert r["hist_eval_precision"] == "off", r["run_id"]


def test_recipe_actual_is_never_the_directory_name(rows):
    """The erratum in one assertion: every legacy row in the cohort whose
    DIRECTORY says 'nomix' has recipe_actual 'mixup'."""
    eligible = [r for r in rows if r["status"] == "eligible_legacy"]
    assert len(eligible) == 9
    by_dirname = {}
    for r in eligible:
        by_dirname.setdefault(r["recipe_dirname"], []).append(r)
    # ViT-S nomix-dir (3) + ViT-B nomix-dir (3); ViT-S mixup-dir (3).
    # The ViT-B mixup-dir trio is the VOID one and appears in neither.
    assert {k: len(v) for k, v in sorted(by_dirname.items())} == {
        "mixup": 3, "nomix": 6}
    for r in by_dirname["nomix"]:
        assert r["recipe_actual"] == "mixup", r["run_id"]
        assert "erratum" in r["recipe_source"] or \
               "build_pooled_tables" in r["recipe_source"]
    assert {r["arch"] for r in by_dirname["mixup"]} == {"vit_small"}
    # ... and the only true-nomix cell is the seeded e2r one
    nomix = [r for r in rows if r["recipe_actual"] == "nomix"
             and r["ckpt_kind"] == "last" and r["status"] == "eligible"]
    assert {r["run_id"] for r in nomix} == {
        f"e2r_vits_nomix_{v}_s{s}" for v in ("baseline", "saga")
        for s in (1, 2)}


def test_cell_membership_is_imported_not_redefined():
    """A source-level guard: the manifest must not carry its own copy of the
    cell tables."""
    src = (REPO / "analysis" / "build_I0_manifest.py").read_text(
        encoding="utf-8")
    assert "from analysis.build_pooled_tables import" in src
    for name in ("LEGACY_MEMBERS", "E2R_MEMBERS", "VARIANTS"):
        assert f"{name} = " not in src, f"{name} is redefined in the manifest"
    assert LEGACY_MEMBERS and E2R_MEMBERS          # imported objects are real


def test_every_row_has_a_status_reason(rows):
    for r in rows:
        assert r["status"] in bm.STATUSES, r["run_id"]
        assert r["family"] in bm.FAMILIES, r["run_id"]
        assert r["status_reason"].strip(), r["run_id"]


def test_run_id_and_ckpt_kind_are_a_unique_key(rows):
    keys = [(r["run_id"], r["ckpt_kind"]) for r in rows]
    assert len(keys) == len(set(keys))


def test_manifest_json_agrees_with_the_csv(rows):
    doc = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    assert doc["n_rows"] == len(rows)
    assert doc["hist_stage"] == HIST_STAGE
    assert doc["columns"] == bm.COLUMNS
    assert doc["n_cohort_300ep"] == 19
    assert [r["run_id"] for r in doc["rows"]] == [r["run_id"] for r in rows]


# ── the generated note ──────────────────────────────────────────────────────

def test_eligibility_note_is_generated_not_hand_typed(tmp_path):
    """Re-render from the committed run metadata; byte-identical or the note
    has drifted from its inputs.

    The committed manifest carries the Phase-B checkpoint hashes, so the
    re-render must be given the same `ckpt_hashes.json` — a rebuild without
    it would legitimately differ and would be testing the wrong thing. The
    merge is new-values-only, so this also re-checks that every committed
    sha still agrees with the hash pass.
    """
    args = [sys.executable, str(REPO / "analysis" / "build_I0_manifest.py"),
            "--out-dir", str(tmp_path)]
    if HASHES.exists():
        args += ["--hashes", str(HASHES)]
    rc = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    assert (tmp_path / "eligibility.md").read_text(encoding="utf-8") == \
        ELIGIBILITY.read_text(encoding="utf-8")
    assert (tmp_path / "manifest.csv").read_text(encoding="utf-8") == \
        MANIFEST_CSV.read_text(encoding="utf-8")


def test_eligibility_note_states_the_hist_stage_with_its_citation():
    text = ELIGIBILITY.read_text(encoding="utf-8")
    assert HIST_STAGE in text
    assert "saga/metrics.py" in text
    assert "before" in text.lower() and "final LayerNorm" in text
    assert "19 eligible conditions" in text


def test_eligibility_note_reports_the_counts_it_was_built_from(rows, cohort):
    text = ELIGIBILITY.read_text(encoding="utf-8")
    assert f"**{len(rows)}**" in text
    assert f"**{len(cohort)} eligible conditions**" in text


# ── behaviour, on synthetic run dirs ────────────────────────────────────────

def _write_run(root, run_id, *, recipe, mixup, cutmix, epochs=300,
               final_epoch=299, end_time="2026-01-01T00:00:00+00:00",
               seed=1, variant="baseline", arch="vit_small_patch16_224"):
    d = root / "results" / "runs" / run_id
    (d / "diag").mkdir(parents=True)
    (d / "eval").mkdir(parents=True)
    yaml.safe_dump({
        "model": {"arch": arch, "gate": variant == "saga", "registers": 0},
        "train": {"epochs": epochs},
        "augmentation": {"mixup_alpha": mixup, "cutmix_alpha": cutmix},
        "variant": variant, "recipe": recipe, "seed": seed,
    }, (d / "config.resolved.yaml").open("w"))
    (d / "meta.json").write_text(json.dumps({
        "run_id": run_id, "seed": seed, "end_time": end_time,
        "git_sha": "deadbeef", "git_dirty": True,
        "knobs": {"gate_init_logit": 0.0}}))
    with (d / "log.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"])
        for e in range(final_epoch + 1):
            w.writerow([e])
    (d / "eval" / "imagenet_val_last.json").write_text(
        json.dumps({"top1": 70.0, "amp": "off"}))
    (d / "diag" / "diag_final_last.json").write_text(json.dumps({"seed": 0}))
    return d


def test_a_recipe_key_contradicting_its_own_augmentation_is_refused(tmp_path):
    """The recipe erratum, as a guard: a config that SAYS nomix while its
    resolved augmentation block applies mixup is the exact failure mode, and
    it must be a hard error rather than a silent mislabel."""
    _write_run(tmp_path, "e2r_vits_nomix_baseline_s1", recipe="nomix",
               mixup=0.8, cutmix=1.0)
    with pytest.raises(SystemExit, match="resolves to 'mixup'"):
        bm.classification_rows(tmp_path / "results" / "runs", "e2r_*",
                               "e2r_300ep")


def test_recipe_actual_comes_from_the_augmentation_block(tmp_path):
    _write_run(tmp_path, "e2r_vits_nomix_baseline_s1", recipe="nomix",
               mixup=0.0, cutmix=0.0)
    _write_run(tmp_path, "e2r_vits_mixup_baseline_s1", recipe="mixup",
               mixup=0.8, cutmix=1.0)
    got = {r["run_id"]: r["recipe_actual"]
           for r in bm.classification_rows(tmp_path / "results" / "runs",
                                           "e2r_*", "e2r_300ep")}
    assert got["e2r_vits_nomix_baseline_s1"] == "nomix"
    assert got["e2r_vits_mixup_baseline_s1"] == "mixup"


def test_a_disagreeing_seed_is_refused_rather_than_picked(tmp_path):
    d = _write_run(tmp_path, "e2r_vits_mixup_baseline_s1", recipe="mixup",
                   mixup=0.8, cutmix=1.0, seed=1)
    meta = json.loads((d / "meta.json").read_text())
    meta["seed"] = 7
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="refusing to pick one"):
        bm.classification_rows(tmp_path / "results" / "runs", "e2r_*",
                               "e2r_300ep")


def test_the_diagnostics_seed_is_never_read(tmp_path):
    """tools/diagnose.py writes its own --seed default of 0 into every diag
    JSON. A run trained with seed 2 must still report 2."""
    d = _write_run(tmp_path, "e2r_vits_mixup_baseline_s2", recipe="mixup",
                   mixup=0.8, cutmix=1.0, seed=2)
    (d / "diag" / "diag_final_last.json").write_text(
        json.dumps({"seed": 0, "sink_mad_k5": 1.0}))
    rows = bm.classification_rows(tmp_path / "results" / "runs", "e2r_*",
                                  "e2r_300ep")
    assert {r["seed"] for r in rows} == {2}
    assert all("meta.json" in r["seed_source"] for r in rows)


def test_an_unfinished_run_is_invalid_for_both_checkpoint_kinds(tmp_path):
    _write_run(tmp_path, "e2r_vitb_mixup_saga_s2", recipe="mixup", mixup=0.8,
               cutmix=1.0, final_epoch=0, end_time=None, variant="saga",
               arch="vit_base_patch16_224", seed=2)
    rows = bm.classification_rows(tmp_path / "results" / "runs", "e2r_*",
                                  "e2r_300ep")
    assert len(rows) == 2
    assert {r["status"] for r in rows} == {"invalid"}
    assert all("training incomplete" in r["status_reason"] for r in rows)


def test_merge_hashes_refuses_to_overwrite_a_recorded_value(tmp_path, rows):
    """New values only — a committed sha must AGREE with the hash pass, not
    be replaced by it (TASK I0 §2.6)."""
    legacy = next(r for r in rows if r["family"] == "legacy_300ep"
                  and r["ckpt_sha256"] != MISSING)
    hashes = tmp_path / "h.json"
    hashes.write_text(json.dumps({"sha256_by_path": {
        legacy["ckpt_path"]: "0" * 64}}))
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        bm.merge_hashes([dict(legacy)], hashes)

    hashes.write_text(json.dumps({"sha256_by_path": {
        legacy["ckpt_path"]: legacy["ckpt_sha256"]}}))
    filled, agreed = bm.merge_hashes([dict(legacy)], hashes)
    assert (filled, agreed) == (0, 1)


def test_phase_b_hashes_are_complete_and_agree_with_recorded_provenance(rows):
    """The Phase-B hash pass is cross-checked against two files that recorded
    checkpoint shas INDEPENDENTLY, long before this task existed. If the
    hashing were wrong, these would disagree."""
    if not HASHES.exists():
        pytest.skip("ckpt_hashes.json not present (Phase B not run)")
    doc = json.loads(HASHES.read_text(encoding="utf-8"))
    assert doc["n_hashed"] >= 114

    canon = json.loads(
        (REPO / "results" / "diagsplit"
         / "fixed_thresholds_canon.json").read_text(encoding="utf-8"))
    for run, key in (("e2r_vits_mixup_baseline_s1", "vit_small|mixup"),
                     ("e2r_vitb_mixup_baseline_s1", "vit_base|mixup"),
                     ("e2r_vits_nomix_baseline_s1", "vit_small|nomix")):
        got = next(r["ckpt_sha256"] for r in rows
                   if r["run_id"] == run and r["ckpt_kind"] == "last")
        assert got == canon["source_ckpt_sha256"][key], run

    legacy = {}
    with open(REPO / "results" / "legacy" / "checkpoint_manifest.csv",
              newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            legacy[r["path"]] = r["sha256"]
    checked = 0
    for r in rows:
        if r["ckpt_path"] in legacy:
            assert r["ckpt_sha256"] == legacy[r["ckpt_path"]], r["run_id"]
            checked += 1
    assert checked == 24


def test_every_remaining_missing_hash_is_accounted_for(rows):
    """After Phase B, a MISSING ckpt_sha256 must fall into one of exactly
    three named buckets. Anything else is an unexplained gap.

    The third bucket is the segmentation `best_model.pth` rows, whose paths
    were CORRECTED after the Phase-B hash pass ran against the wrong
    filename. They stay MISSING until the hash tool is re-run; the assertion
    is written so it keeps holding once they are filled, rather than failing
    when things improve.
    """
    if not HASHES.exists():
        pytest.skip("ckpt_hashes.json not present (Phase B not run)")
    derived, no_file, unexplained = [], [], []
    for r in rows:
        if r["ckpt_sha256"] != MISSING:
            continue
        if r["ckpt_kind"] == "derived":
            derived.append(r["run_id"])
        elif r["ckpt_path"] == MISSING:
            # a checkpoint kind this family never writes AND could not be
            # resolved to another row's file
            no_file.append(r["run_id"])
        else:
            unexplained.append((r["run_id"], r["ckpt_kind"], r["ckpt_path"]))

    assert not unexplained, unexplained
    assert len(derived) == 4                  # the four TTR edits
    # every detection `best` row resolved to last.pth, so nothing is left
    assert no_file == [], no_file
    assert len(derived) + len(no_file) == 4

    # and every EVALUABLE row — anything an intervention could actually be
    # run on — is fully identified
    for r in rows:
        if r["status"] in ("eligible", "eligible_legacy"):
            assert len(r["ckpt_sha256"]) == 64, (r["run_id"], r["ckpt_kind"])


def test_a_renamed_path_does_not_keep_reading_as_a_missing_checkpoint(
        tmp_path):
    """A manifest correction that renames a checkpoint path must not leave
    the old path reported as an absent checkpoint for ever.

    This is the situation the dense `best.pth` -> `best_model.pth` correction
    created: the first hash pass recorded six absent `best.pth` paths, and
    after the correction the tool still printed all six, so a run in which
    every target hashed successfully read as "6 are missing". Absences are
    now scoped to paths the manifest currently names; the rest move to
    `no_longer_in_manifest`.
    """
    import subprocess as sp

    cols = ["run_id", "ckpt_kind", "ckpt_path", "ckpt_sha256"]
    real = tmp_path / "last.pth"
    real.write_bytes(b"weights")
    man = tmp_path / "manifest.csv"
    out = tmp_path / "ckpt_hashes.json"

    def write_manifest(best_name):
        with open(man, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows([
                {"run_id": "seg_a", "ckpt_kind": "last",
                 "ckpt_path": str(real), "ckpt_sha256": MISSING},
                {"run_id": "seg_a", "ckpt_kind": "best",
                 "ckpt_path": str(tmp_path / best_name),
                 "ckpt_sha256": MISSING}])

    def run():
        rc = sp.run([sys.executable,
                     str(REPO / "tools" / "frozen_manifest_hashes.py"),
                     "--manifest", str(man), "--out", str(out)],
                    cwd=REPO, capture_output=True, text=True)
        assert rc.returncode == 0, rc.stderr
        return json.loads(out.read_text(encoding="utf-8")), rc.stdout

    # pass 1: the manifest names a best.pth that does not exist
    write_manifest("best.pth")
    doc, _ = run()
    assert doc["n_absent"] == 1

    # pass 2: the manifest is corrected to the file that really exists
    (tmp_path / "best_model.pth").write_bytes(b"best weights")
    write_manifest("best_model.pth")
    doc, stdout = run()

    assert doc["n_absent"] == 0, "the corrected path must not read as absent"
    assert doc["n_remaining"] == 0
    assert doc["complete"] is True
    assert doc["n_hashed"] == 2
    assert list(doc["no_longer_in_manifest"]) == [str(tmp_path / "best.pth")]
    # the stale path is out of the hash map too, so a consumer cannot resolve
    # a sha for a path the manifest no longer names
    assert str(tmp_path / "best.pth") not in doc["sha256_by_path"]
    assert "0 absent" in stdout
    assert "NOT missing data" in stdout


def test_merge_hashes_fills_a_missing_value(tmp_path):
    """merge_hashes fills a MISSING sha for a row that names a path.

    The row is CONSTRUCTED rather than found in the committed manifest: once
    Phase B completed, no committed row has both a real ckpt_path and a
    MISSING sha, and a test of merge_hashes' behaviour should not depend on
    the manifest being in a transient half-hashed state.
    """
    row = {"run_id": "synthetic", "ckpt_kind": "last",
           "ckpt_path": "results/runs/synthetic/ckpt/last.pth",
           "ckpt_sha256": MISSING}
    hashes = tmp_path / "h.json"
    hashes.write_text(json.dumps({"sha256_by_path": {
        row["ckpt_path"]: "a" * 64}}))
    filled, agreed = bm.merge_hashes([row], hashes)
    assert (filled, agreed) == (1, 0)
    assert row["ckpt_sha256"] == "a" * 64

    # a path the hash pass does not know stays MISSING, never guessed
    other = {"run_id": "synthetic2", "ckpt_kind": "last",
             "ckpt_path": "results/runs/synthetic2/ckpt/last.pth",
             "ckpt_sha256": MISSING}
    assert bm.merge_hashes([other], hashes) == (0, 0)
    assert other["ckpt_sha256"] == MISSING
