"""Experiment configuration: typed, validated parameters read from the YAML files in configs/."""
from __future__ import annotations

import copy
import dataclasses
import itertools
import secrets
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..model.textures import TEXTURES


@dataclass
class TextureSection:
    """Texture type and core spacing."""

    type: str = "flat"          # flat | fcd | tfcd
    x0: float = 1.0             # half distance between neighbouring defect cores


@dataclass
class DynamicsSection:
    """Dimensionless swimmer parameters."""

    peclet: float = 1.0
    epsilon: float = 0.0
    rotational_diffusion: float = 1.0


@dataclass
class TimeSection:
    """Time step, MSD lag range and relaxation time."""

    dt: float = 1e-3
    t_max: float = 1e3          # longest MSD lag (the run lasts t_relax + t_max)
    t_relax: float = 0.0        # relaxation before the MSD reference; the RMSD starts at t = 0


@dataclass
class ParticlesSection:
    """Number of swimmers, seed and initial condition."""

    n: int = 1000
    seed: int | None = None
    initial_position: list = field(default_factory=lambda: [0.0, 0.0])
    initial_orientation: str = "uniform"   # uniform | director


@dataclass
class OutputSection:
    """Sampling of the observables and output file names."""

    n_points: int = 2000        # log-spaced sampling points of MSD and RMSD
    traj_particles: int = 0     # particles whose trajectory is stored
    traj_frames: int = 1
    traj_stride: int = 1
    directory: str = "data/{texture}"
    filename: str = "{texture}_Pe{peclet:g}_eps{epsilon:g}_N{n}_T{t_max:g}_seed{seed}.npz"


@dataclass
class RunSection:
    """Engine choice and execution settings."""

    engine: str = "auto"        # auto | cpu | gpu
    chunk_steps: int = 100_000
    threads: int | None = None
    threads_per_block: int = 128


@dataclass
class ExperimentConfig:
    """Complete description of one simulation, or of a sweep over (Pe, ε)."""

    name: str = "experiment"
    texture: TextureSection = field(default_factory=TextureSection)
    dynamics: DynamicsSection = field(default_factory=DynamicsSection)
    time: TimeSection = field(default_factory=TimeSection)
    particles: ParticlesSection = field(default_factory=ParticlesSection)
    output: OutputSection = field(default_factory=OutputSection)
    run: RunSection = field(default_factory=RunSection)
    sweep: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d):
        """Build and validate a configuration; unknown keys raise an error."""
        d = copy.deepcopy(d or {})
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name in d:
                value = d.pop(f.name)
                default = f.default_factory() if f.default_factory is not dataclasses.MISSING else None
                kwargs[f.name] = _section(type(default), value, f.name) if dataclasses.is_dataclass(default) else value
        if d:
            raise KeyError(f"unknown configuration keys: {sorted(d)}")
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path):
        """Read a configuration from a YAML file."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_dict(self):
        """Plain-dictionary copy (stored in every output file)."""
        return dataclasses.asdict(self)

    def with_overrides(self, overrides):
        """Copy with dotted-key changes, e.g. {'dynamics.peclet': 4.0}."""
        d = self.to_dict()
        for key, value in overrides.items():
            node, *path = d, *key.split(".")
            for k in path[:-1]:
                node = node[k] if isinstance(node.get(k), dict) else _missing(key)
            if path[-1] not in node:
                _missing(key)
            node[path[-1]] = value
        return ExperimentConfig.from_dict(d)

    def expand_sweep(self):
        """One configuration per combination of the values listed in `sweep`."""
        if not self.sweep:
            return [self]
        keys = list(self.sweep)
        points = []
        for values in itertools.product(*(self.sweep[k] for k in keys)):
            point = self.with_overrides(dict(zip(keys, values)))
            point.sweep = {}
            points.append(point)
        return points

    def with_seed(self):
        """Copy with a concrete seed (drawn at random if none is given, then recorded)."""
        if self.particles.seed is not None:
            return self
        return self.with_overrides({"particles.seed": secrets.randbits(63)})

    @property
    def n_relax(self):
        """Number of relaxation steps before the MSD reference."""
        return int(round(self.time.t_relax / self.time.dt))

    @property
    def n_total(self):
        """Total number of time steps of the run."""
        return self.n_relax + int(round(self.time.t_max / self.time.dt))

    def output_path(self):
        """File where the result of this run is stored."""
        keys = dict(texture=self.texture.type, peclet=self.dynamics.peclet, epsilon=self.dynamics.epsilon,
                    n=self.particles.n, seed=self.particles.seed, dt=self.time.dt,
                    t_max=self.time.t_max, t_relax=self.time.t_relax, name=self.name)
        return Path(self.output.directory.format(**keys)) / self.output.filename.format(**keys)

    def validate(self):
        """Check physical and numerical consistency of the parameters."""
        t, dy, p, o = self.time, self.dynamics, self.particles, self.output
        if self.texture.type not in TEXTURES:
            raise ValueError(f"texture.type must be one of {sorted(TEXTURES)}")
        if self.texture.x0 <= 0 or dy.peclet < 0 or dy.epsilon < 0 or dy.rotational_diffusion <= 0:
            raise ValueError("need x0 > 0, peclet >= 0, epsilon >= 0, rotational_diffusion > 0")
        if t.dt <= 0 or t.t_max <= 0 or t.t_relax < 0:
            raise ValueError("need dt > 0, t_max > 0, t_relax >= 0")
        for name in ("t_max", "t_relax"):
            ratio = getattr(t, name) / t.dt
            if abs(ratio - round(ratio)) > 1e-6:
                raise ValueError(f"time.{name} must be a multiple of dt")
        if p.n < 1 or p.initial_orientation not in ("uniform", "director") or len(p.initial_position) != 2:
            raise ValueError("invalid particles section")
        if o.n_points < 2 or o.traj_particles < 0 or o.traj_particles > p.n or o.traj_frames < 1 or o.traj_stride < 1:
            raise ValueError("invalid output section")
        if self.run.engine not in ("auto", "cpu", "gpu"):
            raise ValueError("run.engine must be 'auto', 'cpu' or 'gpu'")
        if dy.epsilon * t.dt >= 0.5:
            raise ValueError(f"epsilon*dt = {dy.epsilon * t.dt:g}: time step too large for the alignment torque")
        if dy.epsilon * t.dt > 0.05:
            warnings.warn(f"epsilon*dt = {dy.epsilon * t.dt:g} > 0.05; check convergence in dt", stacklevel=2)
        if self.texture.type != "flat" and dy.peclet * t.dt / self.texture.x0 > 0.05:
            warnings.warn("Pe*dt/x0 > 0.05; check convergence in dt", stacklevel=2)


def _missing(key):
    """Raise the error for an unknown dotted key."""
    raise KeyError(f"unknown configuration key {key!r}")


def _section(cls, value, prefix):
    """Build one section, rejecting unknown keys and converting numbers written as text."""
    value = dict(value or {})
    defaults = cls()
    unknown = set(value) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise KeyError(f"unknown keys in '{prefix}': {sorted(unknown)}")
    for k, v in value.items():
        ref = getattr(defaults, k)
        if v is None or isinstance(ref, (str, list, bool)):
            continue
        if isinstance(ref, float):
            value[k] = float(v)
        elif isinstance(ref, int) or k == "seed":
            value[k] = int(float(v)) if k != "seed" else int(v)
    return cls(**value)
