"""
evaluation/e6_finegrained/data/ft_meta.py
=========================================
TASK-08: single source of truth for the OFFICIAL CUB / Aircraft splits and
label mappings, shared by the split builder (tools/build_ft_split.py) and
the clean fine-tune trainer (tools/train_ft.py). Both sides parsing the
same code path means the committed val-split labels can never drift from
the labels the trainer assigns.

Conventions (identical to the legacy loaders in cub_dataset.py /
aircraft_dataset.py, which produced the void numbers — the LABEL MAPPINGS
were correct there and are kept):
  - CUB:      class = image_class_labels.txt value - 1  (0-indexed)
  - Aircraft: class = line index of the variant in variants.txt (file order)

Items are (relpath, label) with relpath relative to the DATASET ROOT
(CUB_200_2011/ or fgvc-aircraft-2013b/), so a split file is self-contained:
  - CUB:      "images/001.Black_footed_Albatross/....jpg"
  - Aircraft: "data/images/0034309.jpg"

Metadata can be read from an extracted root OR directly from the
distribution tarball (only the few small text members are read — no image
extraction), so split building works on a login node in seconds.
"""

import tarfile
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

DATASETS = ("cub", "aircraft")

# metadata text files per dataset, keyed by the name parsers use; matched
# by path suffix inside the tar (tar roots: CUB_200_2011/,
# fgvc-aircraft-2013b/)
META_FILES = {
    "cub": {
        "images": "images.txt",
        "labels": "image_class_labels.txt",
        "split": "train_test_split.txt",
    },
    "aircraft": {
        "variants": "data/variants.txt",
        "trainval": "data/images_variant_trainval.txt",
        "test": "data/images_variant_test.txt",
    },
}


def load_meta_texts(dataset: str, root=None, tar=None) -> dict:
    """Read the metadata text files as strings, from an extracted dataset
    root (`root`) or from the distribution tarball (`tar`). Exactly one of
    the two must be given."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    if (root is None) == (tar is None):
        raise ValueError("give exactly one of root= or tar=")

    wanted = META_FILES[dataset]
    texts = {}
    if root is not None:
        for key, rel in wanted.items():
            texts[key] = (Path(root) / rel).read_text()
        return texts

    with tarfile.open(tar, "r:*") as tf:
        names = tf.getnames()
        for key, rel in wanted.items():
            matches = [n for n in names if n.endswith("/" + rel) or n == rel]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"{rel!r}: expected exactly one member match in {tar}, "
                    f"got {matches}")
            texts[key] = tf.extractfile(matches[0]).read().decode()
    return texts


def _parse_cub(texts: dict):
    """Official CUB-200-2011 train/test items. Labels 0-indexed."""
    imgs = {}
    for line in texts["images"].splitlines():
        if line.strip():
            img_id, rel = line.split()
            imgs[int(img_id)] = f"images/{rel}"
    labels = {}
    for line in texts["labels"].splitlines():
        if line.strip():
            img_id, cls = line.split()
            labels[int(img_id)] = int(cls) - 1
    is_train = {}
    for line in texts["split"].splitlines():
        if line.strip():
            img_id, flag = line.split()
            is_train[int(img_id)] = int(flag) == 1

    train = sorted((imgs[i], labels[i]) for i in imgs if is_train[i])
    test = sorted((imgs[i], labels[i]) for i in imgs if not is_train[i])
    n_classes = max(labels.values()) + 1
    return train, test, n_classes


def _parse_aircraft(texts: dict):
    """Official FGVC-Aircraft trainval/test items (variant task).
    Labels follow variants.txt file order (legacy convention)."""
    variants = [l.strip() for l in texts["variants"].splitlines() if l.strip()]
    class_to_idx = {v: i for i, v in enumerate(variants)}

    def parse_split(text):
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            img_id, variant = line.split(" ", 1)
            items.append((f"data/images/{img_id}.jpg", class_to_idx[variant]))
        return sorted(items)

    return (parse_split(texts["trainval"]), parse_split(texts["test"]),
            len(variants))


def official_splits(dataset: str, root=None, tar=None):
    """(train_items, test_items, n_classes) of the OFFICIAL dataset splits,
    items sorted by relpath. 'train' means CUB train / Aircraft trainval."""
    texts = load_meta_texts(dataset, root=root, tar=tar)
    if dataset == "cub":
        return _parse_cub(texts)
    return _parse_aircraft(texts)


# ── Transforms — legacy e6 pipeline VERBATIM (mirrored hyperparameters) ───────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(img_size: int = 224):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(img_size: int = 224):
    return T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FTItemsDataset(Dataset):
    """Dataset over an explicit (relpath, label) item list — the loaded
    form of a committed split file (or of the official test enumeration)."""

    def __init__(self, root, items, transform=None):
        self.root = Path(root)
        self.items = list(items)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel_path, label = self.items[idx]
        img = Image.open(self.root / rel_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
