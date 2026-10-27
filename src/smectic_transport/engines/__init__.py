"""Engines layer: selects the hardware (GPU if available, otherwise CPU) without changing the model."""


class EngineSelector:
    """Chooses and builds the engine requested in the configuration ('auto', 'cpu' or 'gpu')."""

    @staticmethod
    def gpu_status():
        """Return (available, reason) for the CUDA GPU engine; never raises."""
        try:
            from numba import config, cuda
        except Exception as exc:
            return False, f"numba-cuda not importable ({type(exc).__name__})"
        try:
            if not cuda.is_available():
                return False, "no CUDA device or driver found"
            if config.ENABLE_CUDASIM:
                return True, "CUDA simulator"
            if cuda.get_current_device().compute_capability < (6, 0):
                return False, "GPU compute capability < 6.0 (no float64 atomics)"
            return True, "CUDA device found"
        except Exception as exc:
            return False, f"CUDA initialisation failed ({type(exc).__name__})"

    @classmethod
    def resolve(cls, requested="auto"):
        """Return (engine name, reason) actually used for the request."""
        requested = (requested or "auto").lower()
        if requested == "cpu":
            return "cpu", "CPU requested"
        ok, reason = cls.gpu_status()
        if requested == "gpu":
            if not ok:
                raise RuntimeError(f"GPU requested but unavailable: {reason}")
            return "gpu", reason
        if requested == "auto":
            return ("gpu", reason) if ok else ("cpu", f"CPU fallback: {reason}")
        raise ValueError(f"unknown engine {requested!r}; use 'auto', 'cpu' or 'gpu'")

    @staticmethod
    def engine_class(name):
        """Engine class for 'cpu' or 'gpu' (imported lazily)."""
        if name == "gpu":
            from .gpu import GPUEngine
            return GPUEngine
        from .cpu import CPUEngine
        return CPUEngine


__all__ = ["EngineSelector"]
