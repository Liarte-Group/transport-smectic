"""Sampling schedule: when MSD, RMSD and trajectories are recorded during a run."""
import numpy as np


class SamplingSchedule:
    """Log-spaced sampling steps; MSD lags start at the end of relaxation, RMSD times at t = 0."""

    def __init__(self, n_relax, n_total, n_points):
        self.n_relax = int(n_relax)
        self.n_total = int(n_total)
        self.rmsd_steps = self.log_steps(self.n_total, n_points)
        self.msd_steps = self.n_relax + self.log_steps(self.n_total - self.n_relax, n_points)

    @staticmethod
    def log_steps(n_max, n_points):
        """Unique integer steps in [1, n_max], approximately log-spaced, always ending at n_max."""
        s = np.unique(np.clip(np.rint(np.logspace(0.0, np.log10(n_max), n_points)), 1, n_max).astype(np.int64))
        return s if s[-1] == n_max else np.append(s, n_max)

    def times(self, dt):
        """(MSD lags, RMSD times) in units of τ_R."""
        return (self.msd_steps - self.n_relax) * dt, self.rmsd_steps * dt
