"""T7 — build_diag_split determinism and DiagSplitDataset loading."""

import json

import torch
from PIL import Image
from torchvision import transforms

from tools.build_diag_split import DiagSplitDataset, build_split

N_CLASSES = 20
N_IMAGES = 12


def make_fake_imagenet(root):
    for c in range(N_CLASSES):
        cls_dir = root / "val" / f"n{c:08d}"
        cls_dir.mkdir(parents=True)
        for i in range(N_IMAGES):
            img = Image.new("RGB", (8, 8), (c * 10 % 256, i * 20 % 256, 50))
            img.save(cls_dir / f"img{i:02d}_n{c:08d}.JPEG")
    return root


def test_split_is_10_per_class_and_deterministic(tmp_path):
    root = make_fake_imagenet(tmp_path)

    split_a = build_split(root, n_per_class=10, seed=0)
    split_b = build_split(root, n_per_class=10, seed=0)
    assert split_a == split_b

    assert split_a["n"] == N_CLASSES * 10
    assert len(split_a["items"]) == N_CLASSES * 10

    per_class = {}
    for rel_path, class_id in split_a["items"]:
        assert rel_path.startswith("val/")
        assert not rel_path.startswith("/")
        per_class[class_id] = per_class.get(class_id, 0) + 1
    assert per_class == {c: 10 for c in range(N_CLASSES)}

    # a different seed picks a different subset
    split_c = build_split(root, n_per_class=10, seed=1)
    assert split_c != split_a


def test_class_ids_follow_sorted_dir_order(tmp_path):
    root = make_fake_imagenet(tmp_path)
    split = build_split(root, n_per_class=2, seed=0)
    classes = sorted({rel.split("/")[1] for rel, _ in split["items"]})
    for rel_path, class_id in split["items"]:
        assert rel_path.split("/")[1] == classes[class_id]


def test_dataset_loads_the_frozen_split(tmp_path):
    root = make_fake_imagenet(tmp_path)
    split = build_split(root, n_per_class=10, seed=0)
    split_file = tmp_path / "val_diag_split.json"
    split_file.write_text(json.dumps(split))

    ds = DiagSplitDataset(root, split_file, transform=transforms.ToTensor())
    assert len(ds) == N_CLASSES * 10

    img, label = ds[0]
    assert isinstance(img, torch.Tensor) and img.shape == (3, 8, 8)
    assert label == 0
    img_last, label_last = ds[len(ds) - 1]
    assert label_last == N_CLASSES - 1
