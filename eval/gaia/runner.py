"""GAIA agent runner: one VoiceAgent per task, rooted at a fresh workdir.

Note: this drives the current ``VoiceAgent.__call__`` API (per-task
``work_dir``). ``benchmarks.agent_eval_lib`` targets a retired ``run_text``
API and is used here only for ``make_workdir`` (pure tempdir helper).
Steps are session-history turns (tool-call granularity is not exposed by
``__call__``); there is no confirm gate in this API (confirms=0).
"""
import asyncio
import gc
import os
import shutil


async def _probe_leg(model, backend, allow_cpu):
    from src.models.llm import LlmConfig, LlmModel, assert_cuda_leg, leg_device

    device = "cpu" if allow_cpu else "cuda"
    llm = LlmModel(LlmConfig(model=model, backend=backend, device=device))
    print(f"warming LLM {model} [{backend}] device={device} ...", flush=True)
    await llm.warm()
    if allow_cpu:
        print(f"CPU MODE: leg on {leg_device(llm)} (slow, smoke only).",
              flush=True)
    else:
        print(f"agent ready (llm-only) on {assert_cuda_leg(llm)}.", flush=True)
    del llm
    gc.collect()


def _build_task_agent(model, backend, workdir, allow_cpu):
    from src.main import VoiceAgent, VoiceAgentConfig
    from src.models.llm import LlmConfig

    device = "cpu" if allow_cpu else "cuda"
    return VoiceAgent(VoiceAgentConfig(
        llm=LlmConfig(model=model, backend=backend, device=device),
        work_dir=workdir,
        recursion_limit=25,
    ))


async def _run_one(agent, prompt, session_id, timeout_s):
    import time

    t0 = time.perf_counter()
    try:
        reply = await asyncio.wait_for(
            agent(prompt, session_id=session_id), timeout=timeout_s)
        timed_out, error = False, ""
    except asyncio.TimeoutError:
        reply, timed_out, error = "", True, "timeout"
    except Exception as exc:
        reply, timed_out, error = "", False, f"{type(exc).__name__}: {exc}"[:200]
    dt = time.perf_counter() - t0
    try:
        turns = len(agent.history(session_id)) // 2
    except Exception:
        turns = 0
    return {"reply": reply or "", "steps": turns, "actions": [],
            "confirms": 0, "timed_out": timed_out, "error": error,
            "seconds": round(dt, 2), "n_events": 0}


async def _run_async(tasks, model, backend, timeout_s, allow_cpu=False):
    from benchmarks.agent_eval_lib import make_workdir

    from eval.gaia.grader import grade_task

    await _probe_leg(model, backend, allow_cpu)
    rows = []
    for task in tasks:
        workdir = make_workdir()
        src, name = task.get("file_path") or "", task.get("file_name") or ""
        if name and src and os.path.isfile(src):
            try:
                shutil.copy(src, os.path.join(workdir, os.path.basename(name)))
            except Exception:
                pass
        prompt = task["prompt"]
        if name:
            prompt += f"\n(Attachment staged in the working directory as {name}.)"
        agent = _build_task_agent(model, backend, workdir, allow_cpu)
        await agent.llm.warm()
        res = await _run_one(agent, prompt, f"eval-{task['id']}", timeout_s)
        checks = grade_task(workdir, res["reply"], task)
        passed = bool(checks) and all(c["ok"] for c in checks)
        print(f"[{'OK ' if passed else 'FAIL'}] {task['id']:24s} "
              f"L{task['level']} turns={res['steps']} {res['seconds']:6.1f}s",
              flush=True)
        rows.append({"id": task["id"], "level": task["level"],
                     "prompt": task["prompt"][:200],
                     "passed": passed, "checks": checks,
                     "replies": [res["reply"]] if res["reply"] else [],
                     "summary": {}, **res})
        del agent
        gc.collect()
    return rows, {"model": model, "backend": backend}


def run(tasks, model, backend, timeout_s, allow_cpu=False):
    return asyncio.run(_run_async(tasks, model, backend, timeout_s, allow_cpu))


def summarize(rows):
    dts = sorted(r.get("seconds", 0.0) for r in rows)
    by_level = {}
    for lv in sorted({r.get("level") for r in rows}):
        sub = [r for r in rows if r.get("level") == lv]
        by_level[f"L{lv}"] = {
            "n": len(sub),
            "passed": sum(1 for r in sub if r.get("passed")),
        }
    return {
        "n": len(rows),
        "passed": sum(1 for r in rows if r.get("passed")),
        "pass_rate": round(sum(1 for r in rows if r.get("passed")) / len(rows), 3) if rows else 0.0,
        "by_level": by_level,
        "median_seconds": dts[len(dts) // 2] if dts else 0.0,
    }
