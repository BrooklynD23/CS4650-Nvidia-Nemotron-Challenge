# experiments/submissions/

This directory is the recommended landing zone for **Kaggle submission artifacts** produced by the packager.

Expected layout per run:

```
experiments/submissions/<run_id>/
  submission.zip
  submission.manifest.json
```

Notes:

- These artifacts are **not committed to git**.
- The zip must contain **only** `adapter_config.json` and `adapter_model.safetensors` at the archive root.
- The manifest is intentionally **outside** the zip.

How to produce a bundle:

```bash
python scripts/package_submission.py \
  --adapter-dir adapters/<your_adapter_dir> \
  --output-dir experiments/submissions/<run_id> \
  --base-model-id metric/nemotron-3-nano-30b-a3b-bf16/transformers/default \
  --adapter-rank 32 \
  --dataset-version <dataset_version> \
  --eval-sha <eval_sha>
```
