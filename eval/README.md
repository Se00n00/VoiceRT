# eval/ — full-dataset eval CLI (HuggingFace-only)

```bash
# smoke (harness check only, never report):
PYTHONPATH=. python -m eval.evaluate bfcl --version v3 --limit 2 --list-only
PYTHONPATH=. python -m eval.evaluate gaia --level 1 --limit 2 --list-only

# full (Kaggle, GPU):
pip install datasets pyarrow pandas   # V1/V2 mirrors + GAIA parquet
PYTHONPATH=. python -m eval.evaluate bfcl --version all --tag kaggle
PYTHONPATH=. python -m eval.evaluate gaia --level 1 --tag kaggle-gaia
```

* Full runs by default (`--limit 0`). Any `--limit > 0` prints a SMOKE banner.
* `--list-only` fetches + normalizes + prints counts, no GPU/model.
* `--max-seq` default 16384 (see `eval/bfcl/data/README.md` for sizing).
* Results: `eval/results/*.json` (gitignored). `--resume-from` a prior results
  file to continue chunked runs; `--categories` / `--level` to chunk.
* V1/V2/V3 scope: V3 incl. all 5 multi-turn files; V2 via parquet mirrors;
  V1 needs explicit `--v1-questions/--v1-grades` (no machine GT on HF).
* This package only adds files under `eval/`; legacy `benchmarks/*.json*`
  caches are never read (delete command in `eval/bfcl/data/README.md`).
