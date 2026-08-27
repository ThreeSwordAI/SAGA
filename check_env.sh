#!/bin/bash
# Run this on the cluster to verify the environment is ready for E2.
# Usage: bash check_env.sh

echo "======================================================"
echo "  E2 Environment Check"
echo "  $(date)"
echo "======================================================"

PYTHON=/home/vault/iwi5/iwi5359h/envs/saga/bin/python

echo ""
echo "── Python ─────────────────────────────────────────────"
$PYTHON --version

echo ""
echo "── Core libraries ─────────────────────────────────────"
$PYTHON -c "
import torch; print(f'torch       {torch.__version__}')
import torchvision; print(f'torchvision {torchvision.__version__}')
import timm; print(f'timm        {timm.__version__}')
import yaml; print(f'pyyaml      OK')
import scipy; print(f'scipy       {scipy.__version__}')
import numpy; print(f'numpy       {numpy.__version__}')
"

echo ""
echo "── CUDA ────────────────────────────────────────────────"
$PYTHON -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version:   {torch.version.cuda}')
"

echo ""
echo "── SAGA module ─────────────────────────────────────────"
$PYTHON -c "
import sys
sys.path.insert(0, '/home/hpc/iwi5/iwi5359h/my_repos/SAGA')
from saga import build_saga_vit, SpatialGate
print('saga imports:   OK')

# Build ViT-S with gate=True
model = build_saga_vit('vit_small_patch16_224', gate=True, num_classes=1000)
n_total = sum(p.numel() for p in model.parameters()) / 1e6
n_gate  = sum(p.numel() for n, p in model.named_parameters() if 'phi' in n)
print(f'ViT-S + SAGA:   {n_total:.1f}M params, {n_gate} gate params')

# Build ViT-S baseline
base = build_saga_vit('vit_small_patch16_224', gate=False, num_classes=1000)
print(f'ViT-S baseline: {sum(p.numel() for p in base.parameters())/1e6:.1f}M params')

# Build ViT-S + registers
import timm
reg = timm.create_model('vit_small_patch16_224', pretrained=False,
                         num_classes=1000, reg_tokens=4)
print(f'ViT-S +regs:    {sum(p.numel() for p in reg.parameters())/1e6:.1f}M params')
print('All model builds: OK')
"

echo ""
echo "── Config loading ──────────────────────────────────────"
$PYTHON -c "
import sys, yaml
sys.path.insert(0, '/home/hpc/iwi5/iwi5359h/my_repos/SAGA')
from pathlib import Path

cfg_dir = Path('/home/hpc/iwi5/iwi5359h/my_repos/SAGA/classification/configs')

# Check paths.yaml exists
if not (cfg_dir / 'paths.yaml').exists():
    print('ERROR: paths.yaml not found — copy from paths.yaml.template and fill in paths')
    exit(1)

# Load and merge config for variant 0
import yaml
raw = yaml.safe_load(open(cfg_dir / 'variants.yaml'))
base = yaml.safe_load(open(cfg_dir / 'base.yaml'))
v = next(v for v in raw['variants'] if v['id'] == 0)
print(f'Variant 0: {v[\"name\"]}  arch={v[\"model\"][\"arch\"]}  gate={v[\"model\"][\"gate\"]}')
print('Config loading: OK')
"

echo ""
echo "── timm register token support ─────────────────────────"
$PYTHON -c "
import timm, inspect
sig = inspect.signature(timm.create_model)
has_reg = 'reg_tokens' in str(sig)
print(f'num_reg_tokens supported: {has_reg}')
if not has_reg:
    print('WARNING: timm version too old for register tokens')
    print('Run: pip install --upgrade timm --user')
import timm; print(f'timm version: {timm.__version__}')
"

echo ""
echo "======================================================"
echo "  Check complete — $(date)"
echo "======================================================"