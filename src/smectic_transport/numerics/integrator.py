"""Time integration of one particle with on-the-fly MSD and RMSD sampling (shared by CPU and GPU)."""
from ..model.swimmer import euler_maruyama_step
from .random_streams import normal_pair


class ParticleIntegrator:
    """Builds the per-particle time loop; only the way samples are summed differs between engines."""

    @staticmethod
    def build(compile_fn, accumulate):
        """Compile the particle loop with `compile_fn` (njit or cuda.jit) and a summation rule.
        MSD uses r(t_relax) as reference, RMSD uses θ(0); samples are summed into shared arrays."""

        def advance_particle(p, row, n_start, n_end,
                             code, dt, peclet, epsilon, x0, noise_amp,
                             X, Y, TH, X_REF, Y_REF, TH0, SPARE, HAS_SPARE, RNG,
                             n_relax, rmsd_steps, msd_steps, acc_rmsd, acc_msd,
                             traj_stride, traj):
            """Advance particle p from step n_start to n_end and record its samples."""
            x, y, th = X[p], Y[p], TH[p]
            spare = SPARE[p]
            has_spare = HAS_SPARE[p] != 0
            kr = 0
            while kr < rmsd_steps.shape[0] and rmsd_steps[kr] <= n_start:
                kr += 1
            km = 0
            while km < msd_steps.shape[0] and msd_steps[km] <= n_start:
                km += 1
            store_traj = p < traj.shape[0]

            for n in range(n_start, n_end):
                if has_spare:                       # Box–Muller gives two numbers: use both
                    xi = spare
                    has_spare = False
                else:
                    xi, spare = normal_pair(RNG, p)
                    has_spare = True
                x, y, th = euler_maruyama_step(code, x, y, th, xi, dt, peclet, epsilon, x0, noise_amp)
                m = n + 1                           # completed steps, t = m·dt

                if m == n_relax:                    # end of relaxation: MSD reference
                    X_REF[p] = x
                    Y_REF[p] = y
                if kr < rmsd_steps.shape[0] and m == rmsd_steps[kr]:
                    d2 = (th - TH0[p]) ** 2
                    accumulate(acc_rmsd, row, kr, 0, d2)
                    accumulate(acc_rmsd, row, kr, 1, d2 * d2)
                    kr += 1
                if km < msd_steps.shape[0] and m == msd_steps[km]:
                    dx = x - X_REF[p]
                    dy = y - Y_REF[p]
                    r2 = dx * dx + dy * dy
                    accumulate(acc_msd, row, km, 0, dx * dx)
                    accumulate(acc_msd, row, km, 1, dy * dy)
                    accumulate(acc_msd, row, km, 2, r2 * r2)
                    km += 1
                if store_traj and m % traj_stride == 0:
                    f = m // traj_stride
                    if f < traj.shape[1]:
                        traj[p, f, 0] = x
                        traj[p, f, 1] = y
                        traj[p, f, 2] = th

            X[p], Y[p], TH[p] = x, y, th
            SPARE[p] = spare
            HAS_SPARE[p] = 1 if has_spare else 0

        return compile_fn(advance_particle)
