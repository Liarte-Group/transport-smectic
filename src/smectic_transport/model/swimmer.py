"""Active swimmer: self-propulsion along θ and alignment torque from the smectic director."""
import math

from numba import njit

from .textures import director_q


@njit(inline="always")
def euler_maruyama_step(code, x, y, theta, xi, dt, peclet, epsilon, x0, noise_amp):
    """One Euler–Maruyama step of dr/dt = Pe û(θ), dθ/dt = -(ε/2) sin 2(θ-θn) + √(2D_r) η.
    `xi` is a standard normal number and `noise_amp` = √(2 D_r dt); θ is never wrapped."""
    c = math.cos(theta)
    s = math.sin(theta)
    c2n, s2n = director_q(code, x, y, x0)
    torque = -0.5 * epsilon * (2.0 * s * c * c2n - (c * c - s * s) * s2n)
    return x + peclet * c * dt, y + peclet * s * dt, theta + torque * dt + noise_amp * xi


class ActiveSwimmer:
    """Dimensionless swimmer parameters: Péclet number, anchoring strength, rotational diffusion."""

    def __init__(self, peclet, epsilon, rotational_diffusion=1.0):
        self.peclet = float(peclet)
        self.epsilon = float(epsilon)
        self.rotational_diffusion = float(rotational_diffusion)

    def noise_amplitude(self, dt):
        """Standard deviation √(2 D_r dt) of the angular noise per time step."""
        return math.sqrt(2.0 * self.rotational_diffusion * dt)

    step = staticmethod(euler_maruyama_step)
