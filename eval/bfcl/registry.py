"""Registry: which HF files make up the full gradeable V1/V2/V3 sets.

Official source ``gorilla-llm/Berkeley-Function-Calling-Leaderboard`` currently
hosts V3 only (V1/V2 static vanished upstream 2026-09-23). V1/V2 come from
third-party HF mirrors (parquet) and are best-effort: verify with
``evaluate.py bfcl --version v2 --list-only`` on the eval machine.
"""

OFFICIAL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# (remote data file, answers file | None, category, expected_kind)
# expected_kind "none" = negative control (no call is correct).
V3_SINGLE = [
    ("BFCL_v3_simple.json", "possible_answer/BFCL_v3_simple.json", "simple", "ast"),
    ("BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json", "multiple", "ast"),
    ("BFCL_v3_parallel.json", "possible_answer/BFCL_v3_parallel.json", "parallel", "ast"),
    ("BFCL_v3_parallel_multiple.json", "possible_answer/BFCL_v3_parallel_multiple.json",
     "parallel_multiple", "ast"),
    ("BFCL_v3_sql.json", "possible_answer/BFCL_v3_sql.json", "sql", "ast"),
    ("BFCL_v3_java.json", "possible_answer/BFCL_v3_java.json", "java", "ast"),
    ("BFCL_v3_javascript.json", "possible_answer/BFCL_v3_javascript.json", "javascript", "ast"),
    ("BFCL_v3_irrelevance.json", None, "irrelevance", "none"),
    ("BFCL_v3_chatable.json", None, "chatable", "none"),
    ("BFCL_v3_live_simple.json", "possible_answer/BFCL_v3_live_simple.json", "simple", "ast"),
    ("BFCL_v3_live_multiple.json", "possible_answer/BFCL_v3_live_multiple.json", "multiple", "ast"),
    ("BFCL_v3_live_parallel.json", "possible_answer/BFCL_v3_live_parallel.json", "parallel", "ast"),
    ("BFCL_v3_live_parallel_multiple.json",
     "possible_answer/BFCL_v3_live_parallel_multiple.json", "parallel_multiple", "ast"),
    ("BFCL_v3_live_irrelevance.json", None, "irrelevance", "none"),
]

# Deliberately excluded (not AST-gradeable offline):
# - BFCL_v3_live_relevance.json (any relevant call passes; no fixed GT)
# - BFCL_v3_rest.json (no open answers file)
# - BFCL_v3_exec_*.json (needs code execution harness)

# (remote data file, answers file): all share category "multi_turn".
V3_MULTI = [
    ("BFCL_v3_multi_turn_base.json", "possible_answer/BFCL_v3_multi_turn_base.json"),
    ("BFCL_v3_multi_turn_composite.json", "possible_answer/BFCL_v3_multi_turn_composite.json"),
    ("BFCL_v3_multi_turn_long_context.json",
     "possible_answer/BFCL_v3_multi_turn_long_context.json"),
    ("BFCL_v3_multi_turn_miss_func.json",
     "possible_answer/BFCL_v3_multi_turn_miss_func.json"),
    ("BFCL_v3_multi_turn_miss_param.json",
     "possible_answer/BFCL_v3_multi_turn_miss_param.json"),
]

FUNC_DOC_DIR = "multi_turn_func_doc"

# involved class -> func-doc files defining it (class names don't match
# filenames; explicit map so a missing class fails loudly).
CLASS_TO_FILES = {
    "GorillaFileSystem": ["gorilla_file_system.json"],
    "TwitterAPI": ["posting_api.json"],
    "VehicleControlAPI": ["vehicle_control.json"],
    "TradingBot": ["trading_bot.json"],
    "MessageAPI": ["message_api.json"],
    "TravelAPI": ["travel_booking.json"],
    "MathAPI": ["math_api.json"],
    "TicketAPI": ["ticket_api.json"],
}

# V1/V2 mirrors (third-party parquet; schema verified at runtime).
V1_MIRROR_REPO = "Post-training-Data-Flywheel/gorilla-openfunctions-v1"
V2_MIRROR_REPOS = {
    "ast": "hjshah/bfcl_v2_ast",
    "python": "hjshah/bfcl_v2_python",
    "non_python": "hjshah/bfcl_v2_non_python",
    "relevance": "hjshah/bfcl_v2_relevance",
}

GRADEABLE_CATEGORIES = sorted({
    cat for _, _, cat, _ in V3_SINGLE
} | {"multi_turn", "simple", "multiple", "parallel", "parallel_multiple",
     "sql", "java", "javascript", "irrelevance", "chatable"})
