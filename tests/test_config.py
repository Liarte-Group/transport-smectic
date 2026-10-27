"""Checks of the configuration files."""
import warnings
from pathlib import Path

import pytest

from smectic_transport import ExperimentConfig


def test_shipped_configurations():
    """The three configurations load, use dt = 1e-3, T = 1e6, relaxation t = 100, and give unique files."""
    for name in ("flat", "fcd", "tfcd"):
        cfg = ExperimentConfig.from_yaml(Path(__file__).parents[1] / "configs" / f"{name}.yaml")
        assert (cfg.texture.type, cfg.time.dt, cfg.time.t_max, cfg.time.t_relax) == (name, 1e-3, 1e6, 100.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # tfcd at ε = 60 has ε·dt = 0.06 (accuracy note, not an error)
            points = cfg.expand_sweep()
        assert len({p.output_path() for p in points}) == len(points)


def test_invalid_input_is_rejected():
    """Typos and unresolved time steps raise errors."""
    with pytest.raises(KeyError):
        ExperimentConfig.from_dict({"dynamics": {"epsilom": 3}})
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"dynamics": {"epsilon": 60.0}, "time": {"dt": 0.01}})
