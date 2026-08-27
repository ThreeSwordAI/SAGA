import sys
from pathlib import Path

# Make `saga` and `tools.*` importable from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
