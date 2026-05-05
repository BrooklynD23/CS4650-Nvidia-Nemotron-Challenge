# Synthetic Data Provenance

**Issue:** #10 / #24  
**Date:** 2026-05-05  
**Status:** Active

---

## Artifact Layout

```
data/synthetic/
└── <run_id>.jsonl        # one SFTExample JSON object per line
└── <run_id>.sha256       # SHA-256 hex digest of the .jsonl file
```

`run_id` format: `<YYYYMMDD>_<8-char-hex>` (e.g. `20260505_a3f1c8b2`).

The `.jsonl` file is the canonical artifact. The `.sha256` sidecar is generated
immediately after the run and **must be stored alongside the JSONL** to enable
re-verification.

---

## Provenance Dict Schema

Each line of the JSONL is an `SFTExample` (see `src/contracts.py:97`) with a
`provenance` sub-dict:

| Field | Type | Example | Description |
|---|---|---|---|
| `teacher` | `str` | `"metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"` | Model ID that generated the answer |
| `generated_at` | `str` (ISO-8601 UTC) | `"2026-05-05T14:32:07Z"` | UTC timestamp of generation |
| `source_run_id` | `str` | `"20260505_a3f1c8b2"` | Run ID of the generation job (matches filename stem) |
| `solver_confidence` | `float` | `0.95` | Solver confidence score (0.0–1.0); `0.0` if LLM fallback was used |

Full example line:

```json
{
  "id": "syn_0001",
  "prompt": "What is 2 + 2?",
  "answer": "\\boxed{4}",
  "category": "arithmetic",
  "provenance": {
    "teacher": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
    "generated_at": "2026-05-05T14:32:07Z",
    "source_run_id": "20260505_a3f1c8b2",
    "solver_confidence": 0.95
  }
}
```

---

## Fingerprint Algorithm

The `.sha256` sidecar is the SHA-256 hex digest of the raw `.jsonl` bytes:

```python
import hashlib
from pathlib import Path

def compute_sha256(jsonl_path: Path) -> str:
    return hashlib.sha256(jsonl_path.read_bytes()).hexdigest()

def write_sidecar(jsonl_path: Path) -> None:
    jsonl_path.with_suffix(".sha256").write_text(compute_sha256(jsonl_path) + "\n")
```

No normalization is applied — line endings and byte order are preserved as-is.

---

## Re-Verification Steps

1. **Locate the artifact pair**
   ```
   data/synthetic/<run_id>.jsonl
   data/synthetic/<run_id>.sha256
   ```

2. **Re-compute the digest**
   ```bash
   python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" \
     data/synthetic/<run_id>.jsonl
   ```

3. **Compare against the sidecar**
   ```bash
   diff <(python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" \
           data/synthetic/<run_id>.jsonl) \
        <(tr -d '\n' < data/synthetic/<run_id>.sha256)
   # exit 0 = match, exit 1 = tampered or corrupted
   ```

4. **Validate provenance fields** — for each line assert:
   - `provenance.teacher` matches the expected model ID
   - `provenance.source_run_id` matches the filename stem
   - `provenance.generated_at` is a valid ISO-8601 UTC string
   - `provenance.solver_confidence` is in `[0.0, 1.0]`

5. **Check answer format** — every `answer` must contain `\boxed{}`:
   ```bash
   python -c "
   import json, sys
   lines = open(sys.argv[1]).readlines()
   bad = [i+1 for i,l in enumerate(lines) if r'\boxed{' not in json.loads(l).get('answer','')]
   print('Bad lines:', bad or 'none')
   " data/synthetic/<run_id>.jsonl
   ```
