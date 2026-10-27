# Geometry-controlled active transport in smectic textures

Model and simulation code accompanying the paper *"Geometry-controlled active transport in
smectic textures"*.

An active Brownian swimmer moves at constant speed and is aligned by the director of a smectic-A
liquid crystal. In dimensionless units (time $1/D_R$, length $a$):

$$
\dot{\mathbf r} = \mathrm{Pe}\,(\cos\theta,\sin\theta), \qquad
\dot\theta = -\frac{\varepsilon}{2}\sin 2\big(\theta-\theta_n(\mathbf r)\big) + \sqrt{2}\,\eta(t)
$$

The code covers three textures $\theta_n(\mathbf r)$:

| configuration | texture |
|---|---|
| `flat` | uniform planar layers |
| `fcd`  | one toroidal focal conic domain (cores at $(\pm x_0, 0)$) |
| `tfcd` | array of toroidal focal conic domains (triangular lattice of cores) |

The output is data only: MSD, rotational MSD (RMSD), their standard errors and sample
trajectories. No figures are produced.

## Installation

The project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/yhonyar/smectic-active-matter.git
cd smectic-active-matter
uv sync                    # CPU
uv sync --extra gpu        # NVIDIA GPU (CUDA 12, compute capability ≥ 6.0)
uv run smectic-transport info
```

The engine is selected automatically: the GPU is used when available, otherwise all CPU cores.
Both run the same compiled model with the same random streams, so a given seed gives the same
trajectories on either engine.

## Generating the data

```bash
uv run python scripts/generate_data.py                   # all three textures
uv run smectic-transport run configs/fcd.yaml            # one texture (all Pe, ε)
uv run smectic-transport run configs/fcd.yaml --dry-run  # list the runs
uv run smectic-transport run configs/fcd.yaml --set dynamics.peclet=1.0 --set dynamics.epsilon=12
```

All runs use $\Delta t = 10^{-3}$, $N = 5000$ swimmers released at the origin with uniform
orientations, and the $(\mathrm{Pe}, \varepsilon)$ values listed in `configs/*.yaml`.

| observable | reference | time range |
|---|---|---|
| MSD $\langle\lvert\mathbf r(t_{\rm r}+\tau)-\mathbf r(t_{\rm r})\rvert^2\rangle$ | after relaxation, $t_{\rm r}=100$ | $\tau \in (0, 10^6]$ |
| RMSD $\langle[\theta(t)-\theta(0)]^2\rangle$ | $t = 0$, no relaxation | $t \in (0, 10^6+100]$ |

Each run writes `data/<texture>/<texture>_Pe…_eps…_N5000_T1e+06_seed….npz`. Existing files are
skipped, so an interrupted sweep can be restarted with the same command. The production runs
($10^9$ steps per swimmer) are intended for a GPU.

## Output format

```python
from smectic_transport import ResultStore

r = ResultStore.load("data/fcd/fcd_Pe1_eps10_N5000_T1e+06_seed20260102.npz")
r.t_msd, r.msd, r.msd_sem          # MSD vs lag (msd_x, msd_y: components)
r.t_rmsd, r.rmsd, r.rmsd_sem       # RMSD vs time
r.t_traj, r.traj                   # trajectories (particle, frame, [x, y, θ])
r.config, r.metadata               # parameters, seed, software versions, git commit
```

`ResultStore.load_folder("data/fcd")` loads a whole sweep.

## Code organisation

```
src/smectic_transport/
├── model/        textures.py (PlanarTexture, FocalConicDomain, FocalConicArray), swimmer.py (ActiveSwimmer)
├── numerics/     random_streams.py (RandomStreams), integrator.py (ParticleIntegrator, Euler–Maruyama)
├── engines/      EngineSelector, cpu.py (CPUEngine), gpu.py (GPUEngine)
├── experiment/   config.py (ExperimentConfig), schedule.py (SamplingSchedule), simulation.py (Simulation)
├── storage/      result_store.py (ResultStore)
└── cli.py        smectic-transport run | info
configs/          flat.yaml, fcd.yaml, tfcd.yaml
scripts/          generate_data.py
tests/            uv run pytest
docs/             overview diagram (abp-smectic.excalidraw)
```

## Tests

```bash
uv run pytest
```

The tests check the director fields, the exact free-swimmer limits, the reference times of MSD
and RMSD, reproducibility, and agreement between the CPU and GPU engines.

## Citation and license

Please cite the paper and this repository ([`CITATION.cff`](CITATION.cff)). MIT License.

Yhony M. Arce (IFT-UNESP), with William Oropesa (SAIFR-ICTP), Alef R. Siqueira (IFT-UNESP), 
Hartmut Löwen (HHU Düsseldorf), André Vieira (USP) and Danilo B. Liarte (IFT-UNESP & SAIFR-ICTP).
