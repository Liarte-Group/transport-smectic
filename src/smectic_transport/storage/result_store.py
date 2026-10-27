"""Result storage: compressed .npz files that carry the full configuration and software provenance."""
import datetime
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from ..experiment.simulation import SimulationResult

_ARRAYS = ("t_msd", "msd", "msd_x", "msd_y", "msd_sem", "t_rmsd", "rmsd", "rmsd_sem",
           "t_traj", "traj", "final_state")


class ResultStore:
    """Saves and loads SimulationResult objects."""

    @staticmethod
    def provenance():
        """Software versions, git commit and date of the run."""
        import numba

        from .. import __version__
        root = Path(__file__).resolve().parents[3]
        try:
            git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
            commit, dirty = git("rev-parse", "HEAD") or None, bool(git("status", "--porcelain"))
        except Exception:
            commit, dirty = None, None
        return dict(package=__version__, python=platform.python_version(), numpy=np.__version__,
                    numba=numba.__version__, platform=platform.platform(), git_commit=commit,
                    git_dirty=dirty, created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))

    @classmethod
    def save(cls, result, path):
        """Write a result and its metadata to a compressed .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(config=result.config, run=result.metadata, provenance=cls.provenance())
        np.savez_compressed(path, metadata=np.array(json.dumps(meta)), **{k: getattr(result, k) for k in _ARRAYS})
        return path

    @staticmethod
    def load(path):
        """Read a result written by `save`."""
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["metadata"]))
            arrays = {k: f[k] for k in _ARRAYS}
        return SimulationResult(config=meta["config"], metadata={**meta["run"], "provenance": meta["provenance"]}, **arrays)

    @classmethod
    def load_folder(cls, directory):
        """Load every .npz result in a folder, sorted by (ε, Pe)."""
        results = [cls.load(p) for p in sorted(Path(directory).glob("*.npz"))]
        return sorted(results, key=lambda r: (r.config["dynamics"]["epsilon"], r.config["dynamics"]["peclet"]))
