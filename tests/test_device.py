"""Tests for src.device — CUDA detection and device selection utilities.

Skipped automatically when torch is not installed (e.g. plain CI without GPU deps).
On Colab / HPC where torch is available these run against the real torch import.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping device tests")

from src.device import device_info, get_device, log_device_info, require_cuda  # noqa: E402


class TestGetDevice:
    def test_returns_torch_device(self) -> None:
        result = get_device()
        assert isinstance(result, torch.device)

    def test_returns_cuda_when_available(self) -> None:
        with patch("src.device.torch.cuda.is_available", return_value=True):
            assert get_device() == torch.device("cuda")

    def test_returns_cpu_when_unavailable(self) -> None:
        with patch("src.device.torch.cuda.is_available", return_value=False):
            assert get_device() == torch.device("cpu")


class TestDeviceInfo:
    def test_all_keys_present_on_cpu(self) -> None:
        with patch("src.device.torch.cuda.is_available", return_value=False):
            info = device_info()
        expected_keys = {
            "device",
            "cuda_available",
            "cuda_version",
            "gpu_count",
            "gpu_name",
            "gpu_vram_gb",
            "torch_version",
        }
        assert set(info.keys()) == expected_keys

    def test_cpu_values(self) -> None:
        with patch("src.device.torch.cuda.is_available", return_value=False):
            info = device_info()
        assert info["device"] == "cpu"
        assert info["cuda_available"] is False
        assert info["cuda_version"] is None
        assert info["gpu_count"] == 0
        assert info["gpu_name"] is None
        assert info["gpu_vram_gb"] is None
        assert isinstance(info["torch_version"], str)

    def test_cuda_values_populated(self) -> None:
        fake_props = MagicMock()
        fake_props.name = "NVIDIA GeForce RTX 3080"
        fake_props.total_memory = 10 * (1024 ** 3)  # 10 GB

        with (
            patch("src.device.torch.cuda.is_available", return_value=True),
            patch("src.device.torch.cuda.device_count", return_value=1),
            patch("src.device.torch.cuda.get_device_properties", return_value=fake_props),
            patch("src.device.torch.version.cuda", "12.1"),
        ):
            info = device_info()

        assert info["device"] == "cuda"
        assert info["cuda_available"] is True
        assert info["gpu_count"] == 1
        assert info["gpu_name"] == "NVIDIA GeForce RTX 3080"
        assert info["gpu_vram_gb"] == pytest.approx(10.0, abs=0.1)


class TestRequireCuda:
    def test_raises_runtime_error_on_cpu(self) -> None:
        with patch("src.device.torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA GPU is required"):
                require_cuda()

    def test_error_message_contains_setup_instructions(self) -> None:
        with patch("src.device.torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                require_cuda()
        msg = str(exc_info.value)
        assert "pip install" in msg
        assert "Colab" in msg

    def test_returns_cuda_device_when_available(self) -> None:
        with patch("src.device.torch.cuda.is_available", return_value=True):
            result = require_cuda()
        assert result == torch.device("cuda")


class TestLogDeviceInfo:
    def test_returns_torch_device(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("src.device.torch.cuda.is_available", return_value=False):
            result = log_device_info()
        assert isinstance(result, torch.device)

    def test_prints_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("src.device.torch.cuda.is_available", return_value=False):
            log_device_info()
        out = capsys.readouterr().out
        assert "Device summary" in out
        assert "CPU" in out

    def test_prints_cuda_info_when_available(self, capsys: pytest.CaptureFixture[str]) -> None:
        fake_props = MagicMock()
        fake_props.name = "RTX 3060"
        fake_props.total_memory = 12 * (1024 ** 3)

        with (
            patch("src.device.torch.cuda.is_available", return_value=True),
            patch("src.device.torch.cuda.device_count", return_value=1),
            patch("src.device.torch.cuda.get_device_properties", return_value=fake_props),
            patch("src.device.torch.version.cuda", "12.1"),
        ):
            log_device_info()

        out = capsys.readouterr().out
        assert "CUDA" in out
        assert "RTX 3060" in out
