# eval/bfcl data — HuggingFace-only, nothing committed

Full datasets are fetched at eval time; this directory intentionally holds no
manifests or ground truth.

## Sources

* V3 (official, JSON lines, dep-free):
  `gorilla-llm/Berkeley-Function-Calling-Leaderboard` @ `--hf-rev` (default
  `main`). Files: `BFCL_v3_{simple,multiple,parallel,parallel_multiple,sql,
  java,javascript,irrelevance,chatable,live_simple,live_multiple,
  live_parallel,live_parallel_multiple,live_irrelevance}.json`,
  `BFCL_v3_multi_turn_{base,composite,long_context,miss_func,miss_param}.json`,
  `possible_answer/*`, `multi_turn_func_doc/*`.
  Excluded (not AST-gradeable offline): `live_relevance` (any relevant call
  passes), `rest` (no open answers), `exec_*` (needs code execution).
* V2 (third-party parquet mirrors, need `pip install datasets pyarrow`):
  `hjshah/bfcl_v2_{ast,python,non_python,relevance}`. Official V2 static
  vanished upstream 2026-09-23; verify the `test_category` mix with
  `--list-only` before reporting numbers.
* V1: the V1 *test* set has no machine answers anywhere on HF (upstream
  deleted it; the `gorilla-openfunctions-v1` mirror is training data, not the
  test). V1 loads only via explicit `--v1-questions/--v1-grades`; otherwise it
  contributes zero cases (never silent zeros).

## Removing the old committed caches (manual, you run it)

`eval/` never reads `benchmarks/*.json*`. To comply with HF-only, delete the
legacy caches yourself (outside `eval/`, so this package does not do it):

```bash
rm -f benchmarks/bfcl_v1.json benchmarks/bfcl_v2.json benchmarks/bfcl_v3.json \
      benchmarks/bfcl_v1_grades.json benchmarks/bfcl_v2_full.jsonl.gz \
      benchmarks/bfcl_v3_full.jsonl.gz benchmarks/toolcall_subset.json
```

## Min-context (measured estimator, chars/4)

V1 max ~450 · V2-full max ~841 · V3-full max ~7363 (32-func multi-turn).
`--max-seq` default **16384** (budget = max_seq − max_tokens − 512):
8192 = floor (drops longest long-context), 16384 = min-relevant for V3
multi-turn + GAIA, 32768 = full-fidelity long-context + page-fetch bodies.
MiniCPM5-1B native ctx is 131072; KV at 16k ≈ 400MB BF16 — fine on Kaggle T4,
OOM on the local 4GB box (Kaggle-only for full runs).
