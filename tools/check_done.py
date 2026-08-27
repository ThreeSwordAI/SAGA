#!/usr/bin/env python3
"""
tools/check_done.py
===================
Idempotency guard for generated re-derivation scripts.

    python tools/check_done.py OUTPUT_JSON EXPECTED_CKPT_SHA256

Exit 0 iff OUTPUT_JSON exists, parses, and its "ckpt_sha256" equals the
expected value (i.e. this step already ran against the same checkpoint).
Any other state (missing, corrupt, different checkpoint) exits 1 so the
step re-runs.
"""

import json
import sys


def is_done(json_path: str, expected_sha: str) -> bool:
    try:
        with open(json_path) as f:
            data = json.load(f)
        return data.get("ckpt_sha256") == expected_sha
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: check_done.py OUTPUT_JSON EXPECTED_CKPT_SHA256",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if is_done(sys.argv[1], sys.argv[2]) else 1)
