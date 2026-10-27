"""Reproducible random numbers: one xoroshiro128+ stream per particle, identical on CPU and GPU."""
import math

import numpy as np
from numba import njit, uint64

_INV_2_53 = 1.0 / 9007199254740992.0


@njit(inline="always")
def _rotl(x, k):
    """Bit rotation of a 64-bit integer."""
    return (x << uint64(k)) | (x >> uint64(64 - k))


@njit(inline="always")
def _splitmix64(z):
    """SplitMix64 hash, used to derive independent seeds."""
    z = z + uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> uint64(30))) * uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> uint64(27))) * uint64(0x94D049BB133111EB)
    return z ^ (z >> uint64(31))


@njit(cache=True)
def init_states(seed, n):
    """Seed n independent generator states from a single integer seed."""
    states = np.empty((n, 2), dtype=np.uint64)
    base = _splitmix64(uint64(seed))
    for p in range(n):
        s0 = _splitmix64(_splitmix64(base ^ _splitmix64(uint64(p) + uint64(0x632BE59BD9B4E019))))
        s1 = _splitmix64(s0)
        states[p, 0] = s0
        states[p, 1] = s1 if (s0 | s1) != uint64(0) else uint64(1)
    return states


@njit(inline="always")
def uniform(states, p):
    """Uniform number in (0, 1] from the stream of particle p (xoroshiro128+)."""
    s0 = states[p, 0]
    s1 = states[p, 1]
    result = s0 + s1
    s1 = s1 ^ s0
    states[p, 0] = _rotl(s0, 24) ^ s1 ^ (s1 << uint64(16))
    states[p, 1] = _rotl(s1, 37)
    return (float(result >> uint64(11)) + 1.0) * _INV_2_53


@njit(inline="always")
def normal_pair(states, p):
    """Two independent standard normal numbers (Box–Muller) for particle p."""
    rad = math.sqrt(-2.0 * math.log(uniform(states, p)))
    ang = 6.283185307179586 * uniform(states, p)
    return rad * math.cos(ang), rad * math.sin(ang)


class RandomStreams:
    """Per-particle random generators seeded deterministically from (seed, particle index)."""

    def __init__(self, seed, n_particles):
        self.seed = int(seed)
        self.states = init_states(np.uint64(self.seed % 2**64), int(n_particles))

    def uniform_angles(self):
        """One uniform angle in [0, 2π) per particle, drawn from its own stream."""
        return np.array([2.0 * math.pi * uniform(self.states, p) for p in range(self.states.shape[0])])
