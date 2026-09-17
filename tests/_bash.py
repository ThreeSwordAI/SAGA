"""
tests/_bash.py
==============
One way to find a bash that can actually run, shared by every test that
shells out to `bash -n`.

WHY THIS IS NOT `shutil.which("bash")`. On the Windows development box
`which` finds `C:\\Windows\\system32\\bash.EXE` first — the WSL stub — and
WSL on that machine cannot start:

    <3>WSL (12 - Relay) ERROR: CreateProcessParseCommon:1023: getpwuid(0) failed 2
    wsl: Failed to mount C:\\, see dmesg for more details.
    <3>WSL (21 - Relay) ERROR: CreateProcessCommon:818: execvpe(/bin/bash) failed

`bash -n` then exits 1 with that on stderr, and a syntax check that never
ran reads as a syntax error. Measured 2026-09-17: four `env_alex.sh` checks
failed that way in PowerShell while passing under Git Bash, purely because
the two shells order PATH differently. A false red on a shell-syntax test is
worse than no test — it trains the reader to ignore it, and it blocks a
release gate that says "pytest green before the job runs".

So: PROBE each candidate by running it, and skip where none works rather
than failing on a broken one.

Extracted from `tests/test_I1_spatial.py::_working_bash` (TASK A / I1), which
had already solved this for the frozen job files; `tests/test_task10_ttr.py`
still trusted the PATH. Both now import from here.
"""

import shutil
import subprocess
from pathlib import Path

#: Probed in order. `which` first so a properly configured machine is used
#: as-is; the explicit paths are the fallbacks for the Windows box.
CANDIDATES = (
    lambda: shutil.which("bash"),
    lambda: r"C:\Program Files\Git\bin\bash.exe",
    lambda: r"C:\Program Files (x86)\Git\bin\bash.exe",
    lambda: "/bin/bash",
    lambda: "/usr/bin/bash",
)


def working_bash():
    """A bash that starts and exits 0, or None."""
    for get in CANDIDATES:
        try:
            candidate = get()
        except Exception:                                # pragma: no cover
            continue
        if not candidate or not Path(candidate).exists():
            continue
        try:
            probe = subprocess.run([candidate, "-c", "exit 0"],
                                   capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return candidate
    return None


#: Resolved once per session: the probe costs a process launch per candidate.
BASH = working_bash()
