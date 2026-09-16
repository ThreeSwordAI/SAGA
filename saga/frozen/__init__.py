"""
saga/frozen/
============
The shared frozen-intervention layer (TASK I0).

Every later work package (I1-I7) edits a frozen forward pass or extracts
stage-tagged patch statistics. They all go through this package, so the
project has exactly ONE definition of "feature stage", ONE definition of
"prefix tokens", and ONE guarantee that a temporary edit was temporary.

    from saga.frozen import (STAGES, HIST_STAGE, capture_stages,
                             terminal_gate_override, gate_edit,
                             receiver_perturbation, state_hash)

Nothing in this package trains, fits, or optimises anything: there is no
optimizer import below this directory, by design (TASK I0 §11).
"""

from saga.frozen.diag import (DIAG_KEYS, MAD_K, MapAccumulator,
                              load_canon_thresholds, mad_threshold,
                              patch_diagnostics)
from saga.frozen.edits import (frozen_state, gate_edit, receiver_perturbation,
                               state_hash, terminal_gate_override)
from saga.frozen.stages import (HIST_STAGE, HIST_STAGE_CITATION, STAGES,
                                capture_stages, forward_with_stages,
                                stage_block_index)

__all__ = [
    "STAGES", "HIST_STAGE", "HIST_STAGE_CITATION", "capture_stages",
    "forward_with_stages", "stage_block_index",
    "state_hash", "frozen_state", "terminal_gate_override", "gate_edit",
    "receiver_perturbation",
    "DIAG_KEYS", "MAD_K", "MapAccumulator", "mad_threshold",
    "patch_diagnostics", "load_canon_thresholds",
]
