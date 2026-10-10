"""Minimal console on the LLM harness (Groq brain by default, local tools).

``GPU-free``: no server, no voice agent, no GPU. Builds the deep agent
directly — ``LlmModel(backend="groq")`` + ``LocalChatModel`` +
``LocalShellBackend`` + todo/trim middleware — and runs an interactive
prompt. Needs ``GROQ_API_KEY`` in the environment or .env
(``--backend gemini`` selects the Gemini leg + ``GEMINI_API_KEY`` instead).

``PYTHONPATH=. .venv/bin/python -m src.server.console_cli [--backend B] [--model M] [--workdir D]``
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

DIM = "\033[2m"
RESET = "\033[0m"


async def run_turn(agent, text: str, recursion_limit: int = 40):
    """One turn -> (reply, [(tool, args)], n_observations). Never raises
    for model failures (they come back as ``error: ...`` reply text)."""
    try:
        res = await agent.ainvoke(
            {"messages": [HumanMessage(content=str(text))]},
            config={"recursion_limit": recursion_limit},
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as reply text
        return f"error: {type(exc).__name__}: {exc}"[:300], [], 0
    msgs = res.get("messages", []) if isinstance(res, dict) else []
    reply, calls, n_obs = "", [], 0
    for m in msgs:
        if isinstance(m, ToolMessage):
            n_obs += 1
        elif isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                if isinstance(tc, dict):
                    calls.append((tc.get("name", "?"), tc.get("args", {})))
                else:
                    calls.append(
                        (getattr(tc, "name", "?"), getattr(tc, "args", {}))
                    )
            if m.content:
                from src.models.llm import split_thinking

                _, answer = split_thinking(str(m.content))
                if str(answer or "").strip():
                    reply = str(answer)
    return reply, calls, n_obs


async def repl(agent) -> None:
    use_color = sys.stdout.isatty()
    dim, reset = (DIM, RESET) if use_color else ("", "")
    while True:
        print(f"{dim}> {reset}", end="", flush=True)
        try:
            line = await asyncio.to_thread(sys.stdin.readline)
        except KeyboardInterrupt:
            print("", flush=True)
            return
        if not line:
            return  # EOF (Ctrl+D)
        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/q", "exit", "quit"):
            return
        reply, calls, _n_obs = await run_turn(agent, text)
        for name, call_args in calls:
            try:
                arg_str = json.dumps(call_args, default=str)[:160]
            except Exception:
                arg_str = str(call_args)[:160]
            print(f"{dim}… {name} {arg_str}{reset}", flush=True)
        print(reply, flush=True)


async def amain(backend: str, model: str, workdir: str) -> int:
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend
    from langchain.agents.middleware import TodoListMiddleware

    from src.agent.chat_model import LocalChatModel
    from src.agent.middleware.trim import TrimObservationsMiddleware
    from src.models.gemini import DEFAULT_MODEL as GEMINI_DEFAULT
    from src.models.groq import DEFAULT_MODEL as GROQ_DEFAULT
    from src.models.llm import LlmConfig, LlmModel
    from src.prompts.base import SYSTEM_PROMPT
    from src.prompts.middleware import TODO_SYSTEM_PROMPT, TODO_TOOL_DESCRIPTION

    os.makedirs(workdir, exist_ok=True)
    backend = str(backend or "groq").lower()
    if backend == "gemini":
        resolved = model or GEMINI_DEFAULT
        llm = LlmModel(LlmConfig(backend="gemini", gemini_model=resolved))
    elif backend == "groq":
        resolved = model or GROQ_DEFAULT
        llm = LlmModel(LlmConfig(backend="groq", groq_model=resolved))
    else:
        resolved = model or "(backend default)"
        llm = LlmModel(LlmConfig(backend=backend))
    try:
        await llm.warm()
    except SystemExit as exc:
        print(exc)
        return 2
    agent = create_deep_agent(
        model=LocalChatModel(llm=llm),
        backend=LocalShellBackend(root_dir=workdir, timeout=30),
        tools=[],
        middleware=[
            TodoListMiddleware(
                system_prompt=TODO_SYSTEM_PROMPT,
                tool_description=TODO_TOOL_DESCRIPTION,
            ),
            TrimObservationsMiddleware(limit=1500),
        ],
        system_prompt=SYSTEM_PROMPT,
    )
    print(f"{backend} console — model {resolved} · workdir {workdir} · "
          f"/quit to exit", flush=True)
    await repl(agent)
    return 0


def main() -> int:
    from src.models.groq import DEFAULT_MODEL as GROQ_DEFAULT

    ap = argparse.ArgumentParser(description="minimal cloud-harness console"
                                 " (default backend: groq)")
    ap.add_argument("--backend", default="groq",
                    help="groq (default), gemini, ...")
    ap.add_argument("--model", default="",
                    help=f"model id (default: {GROQ_DEFAULT} on groq)")
    ap.add_argument("--workdir", default="")
    args = ap.parse_args()
    workdir = args.workdir or tempfile.mkdtemp(prefix=f"{args.backend}-console-")
    try:
        return asyncio.run(amain(args.backend, args.model, workdir))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
