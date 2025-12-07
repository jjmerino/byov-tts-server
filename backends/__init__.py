import os
from typing import TYPE_CHECKING

from .base import TTSBackend

if TYPE_CHECKING:
    from .pytorch_backend import PyTorchBackend
    from .mps_backend import MPSBackend


def get_backend() -> TTSBackend:
    """
    Get the appropriate TTS backend based on TTS_BACKEND environment variable.

    Supported values:
        - "pytorch" (default): PyTorch/CUDA backend for NVIDIA GPUs
        - "mps": PyTorch MPS backend for Apple Silicon Macs (same models as pytorch)

    Returns:
        TTSBackend instance
    """
    backend_type = os.getenv("TTS_BACKEND", "pytorch").lower()

    if backend_type == "mps":
        from .mps_backend import MPSBackend
        return MPSBackend()
    elif backend_type == "pytorch":
        from .pytorch_backend import PyTorchBackend
        return PyTorchBackend()
    else:
        raise ValueError(
            f"Unknown TTS_BACKEND: {backend_type}. "
            "Supported values: 'pytorch', 'mps'"
        )


__all__ = ["TTSBackend", "get_backend"]
