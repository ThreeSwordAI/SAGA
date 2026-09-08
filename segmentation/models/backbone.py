"""
segmentation/models/backbone.py
================================
Segmentation re-uses the dense-prediction backbone verbatim.

Until TASK-09 this file was a byte-for-byte COPY of
detection/models/backbone.py, which meant the B6 register-token bug had to
be fixed twice and could silently diverge. It is now a thin re-export: one
implementation, one fix, no drift.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from detection.models.backbone import DetectionBackbone

__all__ = ["DetectionBackbone"]
