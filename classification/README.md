# ImageNet Classification

Experiment E2 of the SAGA paper.

**Variants:** 9 (ViT-S/B/L × baseline / registers / SAGA)  
**Dataset:** ImageNet-1K (1.28M train, 50K val)  
**Epochs:** 300  
**Hardware:** 4 × A100 (40GB) per run  

---

## Setup

```bash
# 1. Copy and edit the paths config
cp configs/paths.yaml.template configs/paths.yaml
# Edit configs/paths.yaml with your cluster paths

# 2. Install dependencies (if not already in conda env)
pip install timm pyyaml scipy
```

---

## Running

```bash
cd /path/to/SAGA

# All 9 variants
sbatch --array=0-8 classification/scripts/e2_train_alex.sh

# Single model scale
sbatch --array=0-2 classification/scripts/e2_train_alex.sh  # ViT-S
sbatch --array=3-5 classification/scripts/e2_train_alex.sh  # ViT-B
sbatch --array=6-8 classification/scripts/e2_train_alex.sh  # ViT-L

# Single variant (e.g. ViT-B SAGA = id 5)
sbatch --array=5   classification/scripts/e2_train_alex.sh
```

Auto-resume: if a job hits the 24h time limit, resubmit the same command. The script detects `last.pth` and resumes automatically.

---

## Variant mapping

| ID | Name | Arch | Gate |
|----|------|------|------|
| 0 | ViT-S_baseline | vit_small_patch16_224 | none |
| 1 | ViT-S_registers | vit_small_patch16_224 | 4 registers |
| 2 | ViT-S_SAGA | vit_small_patch16_224 | SAGA |
| 3 | ViT-B_baseline | vit_base_patch16_224 | none |
| 4 | ViT-B_registers | vit_base_patch16_224 | 4 registers |
| 5 | ViT-B_SAGA | vit_base_patch16_224 | SAGA |
| 6 | ViT-L_baseline | vit_large_patch16_224 | none |
| 7 | ViT-L_registers | vit_large_patch16_224 | 4 registers |
| 8 | ViT-L_SAGA | vit_large_patch16_224 | SAGA |

---

## Outputs

```
$OUT_ROOT/e2/
├── checkpoints/
│   └── ViT-B_SAGA/
│       ├── best.pth                  ← best validation accuracy
│       ├── last.pth                  ← latest checkpoint (for resume)
│       ├── gate_maps_epoch0025.pt    ← learned φ_h at epoch 25
│       ├── gate_maps_epoch0050.pt    ← learned φ_h at epoch 50
│       ├── ...
│       └── config.yaml               ← exact config used
├── results/
│   └── ViT-B_SAGA.json               ← metrics + training history
└── logs/
    └── e2_ARRAYID_TASKID.log
```

Gate map snapshots (`gate_maps_epoch*.pt`) are saved every 25 epochs. Load with:
```python
import torch
maps = torch.load('gate_maps_epoch0300.pt')
# maps[layer_idx] → tensor [num_heads, 14, 14]
```