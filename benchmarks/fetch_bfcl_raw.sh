#!/usr/bin/env bash
# Download BFCL raw data into a clean versioned layout.
#
# Usage: bash benchmarks/fetch_bfcl_raw.sh [RAW_DIR]
#   RAW_DIR defaults to /tmp/bfcl_raw (gitignored scratch, NOT the repo).
#
# Layout produced:
#   v1/data/  gorilla_openfunctions_v1_test_{simple,multiple_function}.json
#   v2/data/  {simple,multiple,parallel,parallel_multiple,irrelevance,chatable,sql}.json
#   v2/answers/ {simple,multiple,parallel,parallel_multiple,sql}.json
#   v3/data/  (same 7 static) + live_{simple,multiple,parallel,parallel_multiple}.json
#   v3/answers/ (same 5 static) + live x4
#   v3/data/multi_turn_{base,composite,long_context}.json + answers
#   funcdoc/  8 multi-turn API specs (message_api.json, ...)
#
# Pins: V3 (static+live+MT+funcdoc) from main — verified 2026-09-23 that
# main's V3 static content is identical to dataset commit 023218c
# (2024-08-07). V1/V2 static files were captured 2026-09-23 and then
# VANISHED from upstream (404 at pinned rev AND main — the repo now ships
# V3 only); they are checksummed in benchmarks/bfcl_raw.sha256 and baked
# into benchmarks/bfcl_{v1,v2,v3}.json + bfcl_{v2,v3}_full.jsonl.gz.
# This script re-fetches V3 and best-effort attempts V1/V2 (warns, no fail).
# Verified 2026-09-23: official counts match
# (V1 200+200, V2 1540 static, V3 2140 static + 1351 live w/ answers).
set -u
RAW="${1:-/tmp/bfcl_raw}"
PIN=023218c807d73c3684cdb354efd426a52ea4f159   # V1/V2 best-effort only (404s)
BASE="https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve"
mkdir -p "$RAW/v1/data" "$RAW/v2/data" "$RAW/v2/answers" \
         "$RAW/v3/data" "$RAW/v3/answers" "$RAW/funcdoc"

get() { # get <rev> <remote> <local>
  curl -sfL -m 120 "$BASE/$1/$2" -o "$RAW/$3" \
    && echo "OK $3" || { echo "FAIL $2"; return 1; }
}
try() { # try <rev> <remote> <local> — warn only (archived upstream)
  get "$1" "$2" "$3" || echo "NOTE: $2 not fetchable (archived?); keep checksummed copy"
}
FAIL=0

# V1/V2 static: archived upstream 2026-09-23 — best effort, never fatal.
try "$PIN" gorilla_openfunctions_v1_test_simple.json v1/data/gorilla_openfunctions_v1_test_simple.json
try "$PIN" gorilla_openfunctions_v1_test_multiple_function.json v1/data/gorilla_openfunctions_v1_test_multiple_function.json
for c in simple multiple parallel parallel_multiple irrelevance chatable sql; do
  try "$PIN" "BFCL_v2_$c.json" "v2/data/$c.json"
done
for c in simple multiple parallel parallel_multiple sql; do
  try "$PIN" "possible_answer/BFCL_v2_$c.json" "v2/answers/$c.json"
done

# V3: everything live on main (static verified identical to 023218c).
for c in simple multiple parallel parallel_multiple irrelevance chatable sql; do
  get main "BFCL_v3_$c.json" "v3/data/$c.json" || FAIL=1
done
for c in simple multiple parallel parallel_multiple sql; do
  get main "possible_answer/BFCL_v3_$c.json" "v3/answers/$c.json" || FAIL=1
done
for m in multi_turn_base multi_turn_composite multi_turn_long_context; do
  get main "BFCL_v3_$m.json" "v3/data/$m.json" || FAIL=1
  get main "possible_answer/BFCL_v3_$m.json" "v3/answers/$m.json" || FAIL=1
done

# V3 live (rolling user data — main, NOT pinned)
for c in live_simple live_multiple live_parallel live_parallel_multiple; do
  get main "BFCL_v3_$c.json" "v3/data/$c.json" || FAIL=1
  get main "possible_answer/BFCL_v3_$c.json" "v3/answers/$c.json" || FAIL=1
done

# Shared func docs: 8 separate multi-turn API specs
for f in gorilla_file_system math_api message_api posting_api ticket_api trading_bot travel_booking vehicle_control; do
  get main "multi_turn_func_doc/$f.json" "funcdoc/$f.json" || FAIL=1
done

echo "live-data date: $(date -u +%F)"
[ "$FAIL" -eq 0 ] && echo "RAW COMPLETE -> $RAW" || { echo "RAW INCOMPLETE"; exit 1; }
