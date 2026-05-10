from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = EXAMPLE_DIR / "train.py"


def run_experiment(overrides: Sequence[str], *, cwd: Path | None = None) -> int:
    """Run the discrete Burgers training entry point with Hydra overrides."""

    workdir = cwd or EXAMPLE_DIR
    command = [sys.executable, str(TRAIN_SCRIPT), *overrides]
    completed = subprocess.run(command, cwd=str(workdir), check=False)
    return completed.returncode


def fixed_output_dir(name: str) -> str:
    """Build a stable Hydra output directory for an experiment."""

    return f"experiment/{name}"
