"""CPU engine: particles distributed over all cores with Numba."""
import numba
from numba import njit, prange

from ..numerics.integrator import ParticleIntegrator


@njit(inline="always")
def _add(arr, i, j, k, v):
    """Plain summation (each CPU block owns its own row)."""
    arr[i, j, k] += v


_advance_particle = ParticleIntegrator.build(njit, _add)


@njit(parallel=True, cache=True)
def _advance_blocks(n_rows, n_start, n_end, code, dt, peclet, epsilon, x0, noise_amp,
                    X, Y, TH, X_REF, Y_REF, TH0, SPARE, HAS_SPARE, RNG,
                    n_relax, rmsd_steps, msd_steps, acc_rmsd, acc_msd, traj_stride, traj):
    """Advance fixed blocks of particles in parallel; results do not depend on the thread count."""
    n = X.shape[0]
    for b in prange(n_rows):
        for p in range((b * n) // n_rows, ((b + 1) * n) // n_rows):
            _advance_particle(p, b, n_start, n_end, code, dt, peclet, epsilon, x0, noise_amp,
                              X, Y, TH, X_REF, Y_REF, TH0, SPARE, HAS_SPARE, RNG,
                              n_relax, rmsd_steps, msd_steps, acc_rmsd, acc_msd, traj_stride, traj)


class CPUEngine:
    """Keeps the state in host memory and advances it chunk by chunk."""

    name = "cpu"
    rows = 256

    def __init__(self, state, params, threads=None):
        if threads:
            numba.set_num_threads(int(threads))
        self.state, self.params = state, params

    def description(self):
        """Short description of the hardware in use."""
        return f"CPU, {numba.get_num_threads()} threads"

    def advance(self, n_start, n_end):
        """Advance all particles from step n_start to n_end."""
        s, p = self.state, self.params
        _advance_blocks(s["acc_msd"].shape[0], n_start, n_end, p["code"], p["dt"], p["peclet"],
                        p["epsilon"], p["x0"], p["noise_amp"], s["x"], s["y"], s["theta"],
                        s["x_ref"], s["y_ref"], s["theta0"], s["spare"], s["has_spare"], s["rng"],
                        p["n_relax"], s["rmsd_steps"], s["msd_steps"], s["acc_rmsd"], s["acc_msd"],
                        p["traj_stride"], s["traj"])

    def fetch(self):
        """Return the current state as host arrays."""
        return self.state
