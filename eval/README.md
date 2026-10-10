# eval/ — full-dataset eval CLI (HuggingFace-only)

```bash
# smoke (harness check only, never report):
PYTHONPATH=. python -m eval.evaluate gaia --level 1 --limit 2 --list-only

# full (Kaggle, GPU):
pip install datasets pyarrow pandas   # GAIA parquet mirrors
PYTHONPATH=. python -m eval.evaluate gaia --level 1 --tag kaggle-gaia
```

* Full runs by default (`--limit 0`). Any `--limit > 0` prints a SMOKE banner.
* `--list-only` fetches + normalizes + prints counts, no GPU/model.
* Results: `eval/results/*.json` (gitignored). `--resume-from` a prior results
  file to continue chunked runs; `--level` to chunk.
* This package only adds files under `eval/`.
