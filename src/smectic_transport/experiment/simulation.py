"""Simulation: prepares the initial state, runs an engine and collects MSD, RMSD and trajectories."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np

from ..engines import EngineSelector
from ..model import ActiveSwimmer, Texture
from ..numerics import RandomStreams
from .config import ExperimentConfig
from .schedule import SamplingSchedule


@dataclass
class SimulationResult:
    """Output of one run: observables with standard errors, trajectories and final state."""

    config: dict
    t_msd: np.ndarray          # lag τ measured from t_relax
    msd: np.ndarray
    msd_x: np.ndarray
    msd_y: np.ndarray
    msd_sem: np.ndarray
    t_rmsd: np.ndarray         # time measured from t = 0
    rmsd: np.ndarray
    rmsd_sem: np.ndarray
    t_traj: np.ndarray
    traj: np.ndarray           # (particle, frame, [x, y, θ]) from t = 0
    final_state: np.ndarray    # (particle, [x, y, θ])
    metadata: dict = field(default_factory=dict)


class Simulation:
    """One run of N independent swimmers in a smectic texture."""

    def __init__(self, config: ExperimentConfig):
        self.config = config.with_seed()
        c = self.config
        self.texture = Texture.from_name(c.texture.type, c.texture.x0)
        self.swimmer = ActiveSwimmer(c.dynamics.peclet, c.dynamics.epsilon, c.dynamics.rotational_diffusion)
        self.schedule = SamplingSchedule(c.n_relax, c.n_total, c.output.n_points)

    def initial_state(self, rows):
        """Arrays of the initial state: all swimmers at the same point, orientations as configured."""
        c, n = self.config, self.config.particles.n
        streams = RandomStreams(c.particles.seed, n)
        x = np.full(n, float(c.particles.initial_position[0]))
        y = np.full(n, float(c.particles.initial_position[1]))
        if c.particles.initial_orientation == "uniform":
            theta = streams.uniform_angles()
        else:
            theta = np.full(n, float(self.texture.director_angle(x[0], y[0])))
        o = c.output
        frames = min(o.traj_frames, c.n_total // o.traj_stride + 1)
        traj = np.zeros((o.traj_particles, frames, 3))
        traj[:, 0] = np.stack([x, y, theta], axis=1)[: o.traj_particles]
        return dict(
            x=x, y=y, theta=theta, x_ref=x.copy(), y_ref=y.copy(), theta0=theta.copy(),
            spare=np.zeros(n), has_spare=np.zeros(n, dtype=np.uint8), rng=streams.states,
            rmsd_steps=self.schedule.rmsd_steps, msd_steps=self.schedule.msd_steps,
            acc_rmsd=np.zeros((min(rows, n), self.schedule.rmsd_steps.size, 2)),
            acc_msd=np.zeros((min(rows, n), self.schedule.msd_steps.size, 3)), traj=traj)

    def parameters(self):
        """Scalar parameters passed to the engine."""
        c = self.config
        return dict(code=self.texture.code, dt=c.time.dt, peclet=self.swimmer.peclet,
                    epsilon=self.swimmer.epsilon, x0=self.texture.x0,
                    noise_amp=self.swimmer.noise_amplitude(c.time.dt), n_relax=c.n_relax,
                    traj_stride=c.output.traj_stride)

    def run(self, engine=None, progress=True):
        """Run the simulation and return a SimulationResult."""
        c = self.config
        name, reason = EngineSelector.resolve(engine or c.run.engine)
        cls = EngineSelector.engine_class(name)
        kwargs = {"threads": c.run.threads} if name == "cpu" else {"threads_per_block": c.run.threads_per_block}
        eng = cls(self.initial_state(cls.rows), self.parameters(), **kwargs)
        log = (lambda m: print(m, file=sys.stderr, flush=True)) if progress else (lambda m: None)
        log(f"[{c.texture.type}] Pe={c.dynamics.peclet:g} eps={c.dynamics.epsilon:g} N={c.particles.n} "
            f"steps={c.n_total:,} | {eng.description()} ({reason})")

        t0 = last = time.perf_counter()
        for start in range(0, c.n_total, c.run.chunk_steps):
            end = min(start + c.run.chunk_steps, c.n_total)
            eng.advance(start, end)
            now = time.perf_counter()
            if now - last > 10.0 or end == c.n_total:
                done = end / c.n_total
                log(f"  {100 * done:5.1f}%  elapsed {now - t0:8.0f} s  remaining {(now - t0) * (1 / done - 1):8.0f} s")
                last = now
        wall = time.perf_counter() - t0
        return self._collect(eng.fetch(), name, eng.description(), wall)

    def _collect(self, s, engine, device, wall):
        """Turn the summed samples into means and standard errors."""
        c, n = self.config, self.config.particles.n
        am, ar = s["acc_msd"].sum(axis=0), s["acc_rmsd"].sum(axis=0)
        msd, msd_sem = _mean_sem(am[:, 0] + am[:, 1], am[:, 2], n)
        rmsd, rmsd_sem = _mean_sem(ar[:, 0], ar[:, 1], n)
        t_msd, t_rmsd = self.schedule.times(c.time.dt)
        return SimulationResult(
            config=c.to_dict(), t_msd=t_msd, msd=msd, msd_x=am[:, 0] / n, msd_y=am[:, 1] / n,
            msd_sem=msd_sem, t_rmsd=t_rmsd, rmsd=rmsd, rmsd_sem=rmsd_sem,
            t_traj=np.arange(s["traj"].shape[1]) * c.output.traj_stride * c.time.dt, traj=s["traj"],
            final_state=np.stack([s["x"], s["y"], s["theta"]], axis=1),
            metadata=dict(engine=engine, device=device, wall_time_s=wall))


def _mean_sem(sum1, sum2, n):
    """Mean and standard error of the mean from the sums of x and x²."""
    mean = sum1 / n
    if n < 2:
        return mean, np.full_like(mean, np.nan)
    return mean, np.sqrt(np.maximum(sum2 - n * mean**2, 0.0) / (n - 1) / n)
