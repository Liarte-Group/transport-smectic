"""GPU engine: one CUDA thread per particle, compiled from the same model code as the CPU engine."""
import types

import numpy as np
from numba import config as numba_config
from numba import cuda

from ..numerics import integrator as _integrator
from ..numerics.integrator import ParticleIntegrator


@cuda.jit(device=True, inline=True)
def _atomic_add(arr, i, j, k, v):
    """Atomic summation shared by all GPU threads (float64, compute capability ≥ 6.0)."""
    cuda.atomic.add(arr, (i, j, k), v)


def _pure_python(dispatcher, memo={}):
    """Python version of a jitted function and its callees (only for Numba's CUDA simulator)."""
    from numba.core.registry import CPUDispatcher

    if dispatcher not in memo:
        f = dispatcher.py_func
        g = dict(f.__globals__)
        memo[dispatcher] = types.FunctionType(f.__code__, g, f.__name__, f.__defaults__, f.__closure__)
        for name in f.__code__.co_names:
            if isinstance(g.get(name), CPUDispatcher):
                g[name] = _pure_python(g[name])
    return memo[dispatcher]


if numba_config.ENABLE_CUDASIM:  # test-suite only: emulated device arrays cannot enter CPU-compiled code
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")
    g = dict(_integrator.__dict__)
    g["euler_maruyama_step"] = _pure_python(_integrator.euler_maruyama_step)
    g["normal_pair"] = _pure_python(_integrator.normal_pair)
    _build = types.FunctionType(ParticleIntegrator.build.__code__, g, "build")
    _advance_particle = _build(cuda.jit(device=True), _atomic_add)
else:
    _advance_particle = ParticleIntegrator.build(cuda.jit(device=True), _atomic_add)


@cuda.jit
def _advance_kernel(n_rows, n_start, n_end, code, dt, peclet, epsilon, x0, noise_amp,
                    X, Y, TH, X_REF, Y_REF, TH0, SPARE, HAS_SPARE, RNG,
                    n_relax, rmsd_steps, msd_steps, acc_rmsd, acc_msd, traj_stride, traj):
    """Advance particle p = thread index."""
    p = cuda.grid(1)
    if p < X.shape[0]:
        _advance_particle(p, p % n_rows, n_start, n_end, code, dt, peclet, epsilon, x0, noise_amp,
                          X, Y, TH, X_REF, Y_REF, TH0, SPARE, HAS_SPARE, RNG,
                          n_relax, rmsd_steps, msd_steps, acc_rmsd, acc_msd, traj_stride, traj)


class GPUEngine:
    """Copies the state to the GPU once, advances it in short launches, copies it back once."""

    name = "gpu"
    rows = 64

    def __init__(self, state, params, threads_per_block=128):
        self.params = params
        self.device_state = {k: cuda.to_device(np.ascontiguousarray(v)) for k, v in state.items()}
        self.tpb = int(threads_per_block)
        self.blocks = (state["x"].shape[0] + self.tpb - 1) // self.tpb

    def description(self):
        """Short description of the GPU in use."""
        dev = cuda.get_current_device()
        name = getattr(dev, "name", "CUDA simulator")
        return f"GPU {name.decode() if isinstance(name, bytes) else name}"

    def advance(self, n_start, n_end):
        """Advance all particles from step n_start to n_end."""
        d, p = self.device_state, self.params
        _advance_kernel[self.blocks, self.tpb](
            d["acc_msd"].shape[0], n_start, n_end, p["code"], p["dt"], p["peclet"], p["epsilon"],
            p["x0"], p["noise_amp"], d["x"], d["y"], d["theta"], d["x_ref"], d["y_ref"], d["theta0"],
            d["spare"], d["has_spare"], d["rng"], p["n_relax"], d["rmsd_steps"], d["msd_steps"],
            d["acc_rmsd"], d["acc_msd"], p["traj_stride"], d["traj"])
        cuda.synchronize()

    def fetch(self):
        """Copy the state back to host memory."""
        return {k: v.copy_to_host() for k, v in self.device_state.items()}
