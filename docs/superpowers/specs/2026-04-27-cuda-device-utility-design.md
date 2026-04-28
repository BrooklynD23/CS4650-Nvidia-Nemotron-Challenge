# Design: CUDA Device Detection Utility

**Date:** 2026-04-27  
**Status:** Approved  
**Scope:** Add a centralized device detection module (`src/device.py`) so every notebook and src module picks up CUDA automatically and falls back to CPU gracefully.

---

## Problem

The repo has no shared device-detection logic. Notebook 05 inlines a CUDA check and hard-fails without a GPU. Notebooks 06–09 have no device detection at all. When any of the four team members runs on a different machine (Colab, RTX 3080 local, RTX 3060 campus, HPC) the same code should just work — using CUDA when present, CPU otherwise.

---

## Architecture

### `src/device.py` — new module

Four public functions, no global state:

```
get_device() -> torch.device
    Returns torch.device("cuda") if torch.cuda.is_available(), else torch.device("cpu").
    Prints a one-line summary on first call (which device was selected and why).

device_info() -> dict[str, Any]
    Returns a structured dict safe to call on CPU or CUDA:
      - device: "cuda" | "cpu"
      - cuda_available: bool
      - cuda_version: str | None
      - gpu_count: int
      - gpu_name: str | None        (first GPU, or None on CPU)
      - gpu_vram_gb: float | None   (total VRAM of first GPU, or None on CPU)
      - torch_version: str

require_cuda() -> torch.device
    Returns torch.device("cuda") or raises RuntimeError with a clear message
    explaining how to get a GPU runtime. Used only in notebook 05 which
    cannot produce meaningful results on CPU.

log_device_info() -> torch.device
    Calls device_info(), prints a human-readable summary table, returns get_device().
    Designed for notebook preamble cells — one call gives the team visibility
    into what hardware each run used.
```

### `src/__init__.py`

Export the four functions so notebooks can do:
```python
from src.device import get_device, log_device_info
```

### `requirements.txt`

Add a comment block explaining that `torch==2.4.0` from PyPI installs the CPU build by default. CUDA users must install from the PyTorch index:
```
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```
No structural change to the deps list — just documentation.

### Notebook changes

| Notebook | Change |
|---|---|
| `05_prompting_and_decode_sweeps` | Replace inline `torch.cuda.is_available()` raise with `require_cuda()` from `src.device` |
| `06_trajectory_collection_and_error_slices` | Add preamble cell calling `log_device_info()` |
| `07_solver_framework_design` | Add preamble cell calling `log_device_info()` |
| `08_synthetic_data_recipe` | Add preamble cell calling `log_device_info()` |
| `09_sft_runbook_and_masking` | Add preamble cell calling `log_device_info()` |

Notebooks 00–04 do no model loading; no change needed.

### `tests/test_device.py` — new test file

- `test_get_device_returns_torch_device` — always passes, verifies return type
- `test_device_info_keys_present_on_cpu` — mocks `torch.cuda.is_available()` → False, asserts all keys present
- `test_device_info_keys_present_on_cuda` — mocks CUDA available + `torch.cuda.get_device_properties`, asserts gpu_name/vram populated
- `test_require_cuda_raises_on_cpu` — mocks CUDA unavailable, asserts RuntimeError with helpful message
- `test_require_cuda_returns_device_on_gpu` — mocks CUDA available, asserts returns `torch.device("cuda")`
- `test_log_device_info_returns_device` — smoke test, mocks CUDA unavailable, asserts return type

---

## Data Flow

```
notebook preamble cell
  └─ log_device_info()         ← prints summary, returns device
       └─ get_device()         ← torch.cuda.is_available() → "cuda" or "cpu"
       └─ device_info()        ← gathers GPU stats if CUDA present

model loading cell (in notebooks)
  └─ device = get_device()
  └─ model.to(device) or AutoModel.from_pretrained(..., device_map="auto")

notebook 05 only
  └─ require_cuda()            ← raises RuntimeError if no GPU
```

---

## Error Handling

- `get_device()` never raises — always returns a valid `torch.device`
- `device_info()` never raises — GPU fields are `None` on CPU
- `require_cuda()` raises `RuntimeError` with: device that was found, and the pip install command to get a CUDA torch build
- All functions are safe to import even when torch is not installed yet (import guard pattern: the `RuntimeError` on missing torch happens at call time, not import time — actually torch is always present per `requirements.txt`)

---

## Out of Scope

- Multi-GPU selection (always uses first GPU / `device_map="auto"` for HF models)
- MPS (Apple Silicon) support — team hardware is all NVIDIA
- Adding `device` to `EvalRunConfig` — device is an environment concern, not a run-config concern; the runner stays model-agnostic
