"""Shared helpers for the tests."""
import numpy as np

from smectic_transport import ExperimentConfig


def small_config(**overrides):
    """A small, fast configuration with optional dotted-key overrides."""
    cfg = ExperimentConfig.from_dict(dict(
        texture=dict(type="flat"), dynamics=dict(peclet=1.0, epsilon=0.0),
        time=dict(dt=1e-3, t_max=5.0, t_relax=0.0), particles=dict(n=2000, seed=12345),
        output=dict(n_points=40), run=dict(engine="cpu")))
    return cfg.with_overrides(overrides) if overrides else cfg


def max_z(sim, sem, exact, rel_tol=0.0):
    """Largest deviation from `exact` in standard errors, after a relative tolerance for O(dt) errors."""
    excess = np.maximum(np.abs(sim - exact) - rel_tol * np.abs(exact), 0.0)
    return float(np.max(excess / np.where(sem > 0, sem, np.inf)))
