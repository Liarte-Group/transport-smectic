"""Checks of the model against exact results."""
import math

import numpy as np
import pytest

from smectic_transport import Simulation
from smectic_transport.model import TEXTURES, director_q
from conftest import max_z, small_config


def reference_angle(name, x, y, x0):
    """Director angle written directly with atan2, as in the original implementation."""
    if name == "flat":
        return math.pi / 2
    if name == "fcd":
        return math.atan2(y, x - x0 if x >= 0 else x + x0)
    L, H = 2 * x0, 2 * x0 * math.sqrt(3)
    c1 = (math.floor(x / L + 0.5) * L, math.floor(y / H + 0.5) * H)
    c2 = ((math.floor((x - L / 2) / L + 0.5) + 0.5) * L, (math.floor((y - H / 2) / H + 0.5) + 0.5) * H)
    c = c1 if (x - c1[0]) ** 2 + (y - c1[1]) ** 2 < (x - c2[0]) ** 2 + (y - c2[1]) ** 2 else c2
    return math.atan2(y - c[1], x - c[0])


@pytest.mark.parametrize("name", sorted(TEXTURES))
def test_director_fields(name):
    """(cos 2θn, sin 2θn) used by the integrator equals the atan2 definition of each texture."""
    code = TEXTURES[name].code
    for x, y in np.random.default_rng(0).uniform(-15, 15, size=(3000, 2)):
        th = reference_angle(name, x, y, 1.0)
        assert director_q(code, x, y, 1.0) == pytest.approx((math.cos(2 * th), math.sin(2 * th)), abs=1e-12)


def test_free_swimmer():
    """ε = 0: MSD = 2Pe²(t − 1 + e^{−t}) and RMSD = 2t."""
    cfg = small_config(**{"dynamics.peclet": 2.0, "particles.n": 4000})
    r = Simulation(cfg).run(progress=False)
    assert max_z(r.msd, r.msd_sem, 8.0 * (r.t_msd - 1 + np.exp(-r.t_msd)), rel_tol=3 * cfg.time.dt) < 5
    assert max_z(r.rmsd, r.rmsd_sem, 2.0 * r.t_rmsd) < 5
