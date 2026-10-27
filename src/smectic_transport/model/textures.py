"""Smectic-A textures: director fields of the planar, FCD and TFCD configurations."""
import math

import numpy as np
from numba import njit

PLANAR, FCD, TFCD = 0, 1, 2
_CORE_EPS2 = 1e-20  # |r - r_core| < 1e-10 is treated as the core itself


@njit(inline="always")
def _radial(dx, dy):
    """(cos 2θn, sin 2θn) of a director pointing radially away from a core; θn = 0 at the core."""
    r2 = dx * dx + dy * dy
    if r2 > _CORE_EPS2:
        return (dx * dx - dy * dy) / r2, 2.0 * dx * dy / r2
    return 1.0, 0.0


@njit(inline="always")
def director_q(code, x, y, x0):
    """Return (cos 2θn, sin 2θn) of the texture `code` at (x, y).
    The torque only depends on 2θn, which makes the director head–tail symmetric."""
    if code == PLANAR:
        return -1.0, 0.0                      # θn = π/2: layers along x
    if code == FCD:
        dx = x - x0 if x >= 0.0 else x + x0   # cores at (+x0, 0) and (-x0, 0)
        return _radial(dx, y)
    # TFCD: cores on a triangular lattice with spacing L = 2 x0; nearest core is used
    L = 2.0 * x0
    H = L * 1.7320508075688772
    cx1 = math.floor(x / L + 0.5) * L
    cy1 = math.floor(y / H + 0.5) * H
    cx2 = (math.floor((x - 0.5 * L) / L + 0.5) + 0.5) * L
    cy2 = (math.floor((y - 0.5 * H) / H + 0.5) + 0.5) * H
    if (x - cx1) ** 2 + (y - cy1) ** 2 < (x - cx2) ** 2 + (y - cy2) ** 2:
        return _radial(x - cx1, y - cy1)
    return _radial(x - cx2, y - cy2)


class Texture:
    """Base class: a smectic texture defined by its director field θn(x, y)."""

    code = None
    name = None

    def __init__(self, x0=1.0):
        self.x0 = float(x0)

    def director_angle(self, x, y):
        """Director angle θn ∈ (-π/2, π/2] at the points (x, y)."""
        def angle(a, b):
            """Angle at one point."""
            c2, s2 = director_q(self.code, a, b, self.x0)
            return 0.5 * math.atan2(s2, c2)

        return np.vectorize(angle)(np.asarray(x, float), np.asarray(y, float))

    @staticmethod
    def from_name(name, x0=1.0):
        """Build a texture from its configuration name: 'flat', 'fcd' or 'tfcd'."""
        try:
            return TEXTURES[name](x0)
        except KeyError:
            raise ValueError(f"unknown texture {name!r}; use one of {sorted(TEXTURES)}") from None


class PlanarTexture(Texture):
    """Uniform planar smectic: flat layers along x, director along y."""

    code, name = PLANAR, "flat"


class FocalConicDomain(Texture):
    """Cross-section of one toroidal FCD: layers concentric around cores at (±x0, 0)."""

    code, name = FCD, "fcd"


class FocalConicArray(Texture):
    """Array of toroidal FCDs: cores on a triangular lattice of spacing 2·x0."""

    code, name = TFCD, "tfcd"


TEXTURES = {cls.name: cls for cls in (PlanarTexture, FocalConicDomain, FocalConicArray)}
