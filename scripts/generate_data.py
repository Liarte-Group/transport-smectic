"""Generate the output data of the paper for the three textures (flat, fcd, tfcd)."""
import os
from pathlib import Path

from smectic_transport.cli import run_config

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # results are written to data/<texture>/ inside the repository

for name in ("flat", "fcd", "tfcd"):
    run_config(ROOT / "configs" / f"{name}.yaml")
