"""CUDA detection and device selection utilities.

Single source of truth for device selection across notebooks and src modules.
Always prefers CUDA when available; falls back to CPU gracefully.
"""

from __future__ import annotations

from typing import Any

import torch


def get_device() -> torch.device:
    """Return the best available device: CUDA if present, CPU otherwise."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_info() -> dict[str, Any]:
    """Return a structured snapshot of available compute hardware.

    All fields are always present; GPU-specific fields are None on CPU.
    """
    cuda_ok = torch.cuda.is_available()
    info: dict[str, Any] = {
        "device": "cuda" if cuda_ok else "cpu",
        "cuda_available": cuda_ok,
        "cuda_version": torch.version.cuda if cuda_ok else None,
        "gpu_count": torch.cuda.device_count() if cuda_ok else 0,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "torch_version": torch.__version__,
    }
    if cuda_ok and info["gpu_count"] > 0:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
    return info


def require_cuda() -> torch.device:
    """Return a CUDA device or raise with a clear setup message.

    Use this only in contexts that genuinely cannot run on CPU (e.g. notebook 05
    real prompt sweep). For everything else use get_device() for graceful fallback.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required here but none was found.\n"
            "  • On Colab: Runtime → Change runtime type → GPU\n"
            "  • Locally:  pip install torch==2.4.0 "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "  • CPU fallback is intentionally disabled for this step because "
            "results would not be meaningful."
        )
    return torch.device("cuda")


def log_device_info() -> torch.device:
    """Print a device summary table and return the selected device.

    Designed for notebook preamble cells so every run records which hardware
    it used without any extra boilerplate.
    """
    info = device_info()
    device = get_device()
    print("=" * 48)
    print("Device summary")
    print("=" * 48)
    print(f"  Selected device : {info['device'].upper()}")
    print(f"  CUDA available  : {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"  CUDA version    : {info['cuda_version']}")
        print(f"  GPU count       : {info['gpu_count']}")
        print(f"  GPU name        : {info['gpu_name']}")
        print(f"  GPU VRAM        : {info['gpu_vram_gb']} GB")
    print(f"  PyTorch version : {info['torch_version']}")
    print("=" * 48)
    return device


__all__ = [
    "get_device",
    "device_info",
    "require_cuda",
    "log_device_info",
]
