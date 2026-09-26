"""Model driving for eval/bfcl (imports the production path, adds budget/policy).

Nothing here is a sample: --limit exists only as a smoke flag and prints a
SMOKE banner whenever it truncates the full case list.
"""
import asyncio
import time


def _step_max_tokens(llm) -> int:
    base = int(getattr(getattr(llm, "config", None), "max_tokens", 48) or 48)
    backend = str(getattr(getattr(llm, "config", None), "backend", ""))
    return max(base, 320) if backend.startswith("minicpm") else max(base, 256)


async def _run_async(cases, model, backend, max_seq, k, resume_keys,
                   allow_cpu=False):
    from benchmarks.bfcl_eval import (
        error_row,
        filter_cases,
        run_multiturn,
        run_single,
        skipped_row,
    )
    from src.agent.chat_model import LocalChatModel
    from src.models.llm import LlmConfig, LlmModel, assert_ready_leg

    if resume_keys:
        cases = filter_cases(cases, done=resume_keys)

    device = "cpu" if allow_cpu else "cuda"
    llm = LlmModel(LlmConfig(model=model, backend=backend,
                             max_seq=max_seq, device=device))
    print(f"warming {model} [{backend}] max_seq={max_seq} device={device} ...",
          flush=True)
    await llm.warm()
    if allow_cpu:
        from src.models.llm import leg_device

        print(f"CPU MODE: leg on {leg_device(llm)} (slow, smoke only).",
              flush=True)
    else:
        print(f"ready on {assert_ready_leg(llm)}.", flush=True)
    cm = LocalChatModel(llm=llm)
    max_tokens = _step_max_tokens(llm)
    budget = int(max_seq) - int(max_tokens) - 512

    from benchmarks.bfcl_eval import estimate_case_tokens

    rows = []
    for case in cases:
        # V1 without hand grades is not gradeable: skip loudly, never fail.
        if case.get("expected") is None and case.get("expected_kind") == "hand":
            rows.append(skipped_row(case, "V1 hand grade missing"))
            continue
        est = estimate_case_tokens(case, max_tokens)
        if est > budget:
            row = skipped_row(case, f"prompt ~{est} tok > {budget} budget")
            rows.append(row)
            print(f"[SKIP] {row['id']:28s} {row['status_reason']}", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            if case.get("category") == "multi_turn":
                row = await run_multiturn(cm, llm, case, max_tokens)
            else:
                row = await run_single(cm, llm, case, max_tokens, k)
            row["seconds"] = round(time.perf_counter() - t0, 2)
            row["status"] = "ok"
        except Exception as exc:
            row = error_row(case, f"{type(exc).__name__}: {exc}",
                            time.perf_counter() - t0)
            print(f"[ERROR] {row['id']:28s} {row['status_reason']}", flush=True)
            rows.append(row)
            continue
        rows.append(row)
        first_calls = row.get("first_calls") or []
        got = first_calls[0][0] if first_calls else "none"
        want = row.get("expected_tool") or "none"
        mark = "OK " if row["passed"] else ("fp " if row["fp"] else "FAIL")
        print(f"[{mark}] {row['id']:28s} want={str(want):24s} got={str(got):24s} "
              f"retry={row['retries']} {row['seconds']:6.1f}s", flush=True)
    meta = {"model": model, "backend": backend, "max_seq": max_seq,
            "max_tokens": max_tokens, "budget": budget, "k": k}
    return rows, meta


def run(cases, model, backend, max_seq, k, resume_keys, allow_cpu=False):
    return asyncio.run(_run_async(cases, model, backend, max_seq, k,
                                  resume_keys, allow_cpu))
