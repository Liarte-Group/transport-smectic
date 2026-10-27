"""Checks of the bookkeeping: reference times, reproducibility, storage and CPU/GPU agreement."""
import os
import subprocess
import sys
from pathlib import Path

import numba
import numpy as np
import pytest

from smectic_transport import ResultStore, Simulation
from conftest import small_config


@pytest.mark.parametrize("t_relax", [0.0, 1.0])
def test_relaxation_only_affects_msd(t_relax):
    """MSD is referenced to r(t_relax); RMSD to θ(0); trajectories cover every step."""
    cfg = small_config(**{"texture.type": "tfcd", "dynamics.epsilon": 6.0, "time.t_relax": t_relax,
                          "time.t_max": 2.0, "particles.n": 40, "output.traj_particles": 40,
                          "output.traj_frames": 10**6, "output.n_points": 30, "run.chunk_steps": 777})
    r = Simulation(cfg).run(progress=False)
    n0, traj = cfg.n_relax, r.traj
    assert r.t_rmsd[-1] == pytest.approx(t_relax + 2.0) and r.t_msd[-1] == pytest.approx(2.0)
    for t, v in zip(r.t_rmsd, r.rmsd):
        m = int(round(t / cfg.time.dt))
        assert v == pytest.approx(np.mean((traj[:, m, 2] - traj[:, 0, 2]) ** 2), rel=1e-12)
    for lag, v in zip(r.t_msd, r.msd):
        m = n0 + int(round(lag / cfg.time.dt))
        assert v == pytest.approx(np.mean(np.sum((traj[:, m, :2] - traj[:, n0, :2]) ** 2, axis=1)), rel=1e-12)


def test_reproducible_for_any_thread_count(tmp_path):
    """Same seed gives identical results with 1 or all threads, and survives save/load."""
    cfg = small_config(**{"texture.type": "fcd", "dynamics.epsilon": 5.0, "particles.n": 300, "time.t_max": 1.0})
    a = Simulation(cfg).run(progress=False)
    n = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        b = Simulation(cfg).run(progress=False)
    finally:
        numba.set_num_threads(n)
    assert np.array_equal(a.final_state, b.final_state) and np.array_equal(a.rmsd, b.rmsd)
    c = ResultStore.load(ResultStore.save(a, tmp_path / "r.npz"))
    assert np.array_equal(c.msd, a.msd) and c.config["particles"]["seed"] == 12345


GPU_CHECK = r"""
import numpy as np
from smectic_transport import ExperimentConfig, Simulation
for tex in ("flat", "fcd", "tfcd"):
    cfg = ExperimentConfig.from_dict(dict(texture=dict(type=tex), dynamics=dict(peclet=3.0, epsilon=8.0),
        time=dict(dt=1e-3, t_max=0.2, t_relax=0.05), particles=dict(n=5, seed=7),
        output=dict(n_points=15, traj_particles=2, traj_frames=40, traj_stride=3), run=dict(chunk_steps=61)))
    a = Simulation(cfg).run(engine="cpu", progress=False)
    b = Simulation(cfg).run(engine="gpu", progress=False)
    for k in ("final_state", "traj", "msd", "rmsd"):
        assert np.allclose(getattr(a, k), getattr(b, k), rtol=1e-12, atol=1e-14), (tex, k)
print("OK")
"""


def test_cpu_and_gpu_engines_agree():
    """The GPU kernel (run in Numba's CUDA simulator) reproduces the CPU trajectories."""
    import importlib.util
    if importlib.util.find_spec("numba_cuda") is None:
        pytest.skip("GPU extra not installed (uv sync --extra gpu)")
    env = dict(os.environ, NUMBA_ENABLE_CUDASIM="1",
               PYTHONPATH=str(Path(__file__).parents[1] / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, "-c", GPU_CHECK], env=env, capture_output=True, text=True, timeout=1200)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-2000:]
