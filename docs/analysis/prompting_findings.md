# Prompting Findings

Status: implementation complete, execution blocked pending real sweep inputs.

## Current State

Notebook `notebooks/05_prompting_and_decode_sweeps.ipynb` now contains the runnable sweep workflow for issue `#21`:
- strict split loading from `data/eval/validation_200.jsonl` and `data/eval/golden_20.jsonl`
- sparse sweep over zero-shot CoT and few-shot CoT across the required decode grid
- Best-of-N majority-vote follow-up for `N=8` and `N=32`
- artifact writing through `run_baseline_eval()` into `data/eval/runs/<run_id>/`
- aggregate CSV writing to `experiments/prompting_sweep_<date>.csv`
- promoted-config golden regression check via `evaluate_golden_gate()`

## Blockers

The current checkout does not include the required split artifacts:
- `data/eval/validation_200.jsonl`
- `data/eval/golden_20.jsonl`

The only baseline artifacts currently present are the stub runs from notebook `04` under `data/eval/runs/baseline-stub-v1*`. Those are not sufficient to execute the sweep because they do not contain the frozen prompt-bearing split rows required by the runner.

## Next Execution Step

After issue `#18` outputs the real split files and issue `#19` produces a baseline run on the same dataset version, run notebook `05` end-to-end in a CUDA environment. The notebook will then overwrite this file with ranked findings and write the aggregate CSV.


## Prompt Candidate Notes

### final-boxed-answer-v1

Prompt idea:
Ask the model to solve normally, but always end with:

Final answer: \boxed{...}

Why:
The evaluator expects boxed answers, so this may reduce formatting errors.