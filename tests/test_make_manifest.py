"""make_manifest: field guesses come only from below the scanned root.

Regression: on the HPC every checkpoint lives under /home/.../SAGA/..., and
matching the absolute path labeled EVERY row variant=saga (baselines and
registers included). Guesses must use the root-relative path with bounded
'saga' matching.
"""

from tools.make_manifest import guess_fields, scan


def test_ancestor_saga_dir_does_not_set_variant():
    # relative paths (as scan() now passes them) with no variant token
    assert guess_fields("e1/V01_A/last.pth")["variant"] == ""
    # the enclosing project dir must not leak in — scan() strips it, and even
    # a literal 'sagacity' token inside a name must not match
    assert guess_fields("sagacity_run/last.pth")["variant"] == ""


def test_variant_recipe_arch_guesses():
    g = guess_fields("e2/checkpoints/ViT-B_SAGA/best.pth")
    assert (g["exp"], g["arch"], g["variant"]) == ("e2", "vit_base", "saga")

    g = guess_fields("e2/checkpoints/ViT-S_registers_nomix/last.pth")
    assert (g["arch"], g["variant"], g["recipe"]) == \
        ("vit_small", "registers", "nomix")

    g = guess_fields("e2/checkpoints/ViT-B_baseline/best.pth")
    assert (g["arch"], g["variant"]) == ("vit_base", "baseline")
    # 'vit_base' must not be misread as variant baseline
    assert guess_fields("vit_base_patch16_224/last.pth")["variant"] == ""

    assert guess_fields("runs/x_seed2/best.pth")["seed"] == "2"
    assert guess_fields("runs/x_s1/best.pth")["seed"] == "1"


def test_scan_strips_the_root_before_guessing(tmp_path):
    # checkpoint tree deliberately nested under a directory named SAGA
    root = tmp_path / "SAGA" / "checkpoints"
    for run in ["ViT-B_baseline", "ViT-S_registers_nomix", "ViT-B_SAGA"]:
        d = root / "e2" / run
        d.mkdir(parents=True)
        (d / "best.pth").write_bytes(b"\x00" * 8)

    rows = scan([root], do_hash=False)
    by_run = {r["path"].rsplit("/", 2)[1]: r for r in rows}
    assert by_run["ViT-B_baseline"]["variant"] == "baseline"
    assert by_run["ViT-S_registers_nomix"]["variant"] == "registers"
    assert by_run["ViT-B_SAGA"]["variant"] == "saga"
    assert all(r["exp"] == "e2" for r in rows)
    assert all(r["sha256"] == "" for r in rows)
