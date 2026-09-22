"""Plain REPL over the unified VoiceAgent - no TUI, no VAD/STT/TTS weights.

Text in -> think/tool loop -> text out (same agent as voice + terminal).

Run:
  PYTHONPATH=. python llm_chat.py
  PYTHONPATH=. python llm_chat.py --model Qwen/Qwen3-0.6B --backend qwen
  PYTHONPATH=. python llm_chat.py --no-tools  (raw generate, no tool loop)

Keys: type + Enter, /quit to exit, /new for new session.
"""
import argparse
import asyncio
import uuid

from src.main import VoiceAgent, VoiceAgentConfig
from src.models.llm import LlmConfig, LlmModel, split_thinking


async def confirm(action) -> bool | str:
    op = getattr(action, "op", "?")
    detail = getattr(action, "command", "") or getattr(action, "path", "") or ""
    try:
        ans = input(f"[confirm] {op} {detail} [y/N/always]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if ans in ("always", "a"):
        return "always"
    return ans in ("y", "yes")


async def raw_loop(llm: LlmModel, max_tokens: int) -> None:
    """No tools: messages() + stream() directly. True per-token output."""
    history: list = []
    print("raw mode - no tools, just chat. /quit to exit.")
    while True:
        try:
            text = input("> user: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("/quit", "/q", "exit"):
            break
        msgs = llm.messages(text, history)
        print("> agent: ", end="", flush=True)
        raw_buf = ""
        printed = 0
        try:
            async for tok in llm.stream(msgs, max_tokens=max_tokens):
                if not tok.piece:
                    continue
                raw_buf += tok.piece
                # Stream answer only: thinking stays out of the reply line
                # (split handles unclosed <think> mid-stream).
                _, answer = split_thinking(raw_buf)
                new = answer[printed:]
                if new:
                    print(new, end="", flush=True)
                    printed += len(new)
        except Exception as exc:
            print(f"\n> error: {exc}")
            continue
        print()  # close stream line
        thinking, answer = split_thinking(raw_buf)
        if thinking:
            print(f"  think: {thinking[:800]}")
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": answer or raw_buf}]
        history = history[-20:]  # keep prompt small


async def harness_loop(llm: LlmModel, cwd: str, logf=None,
                     sandbox=None) -> None:
    def log(*parts):
        if logf is not None:
            print(*parts, file=logf, flush=True)
    agent = VoiceAgent(VoiceAgentConfig(sandbox=sandbox))
    agent.llm = llm  # warmed leg above; vad/stt/tts stay cold (unused)
    agent.tts = None
    from src.agent.chat_model import LocalChatModel

    agent.chat_model = LocalChatModel(llm=llm)
    sid = uuid.uuid4().hex[:8]
    print(f"harness mode - session {sid}, cwd {cwd}. /quit to exit, /new for new session.")
    while True:
        try:
            text = input("> user: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("/quit", "/q", "exit"):
            break
        if text == "/new":
            sid = uuid.uuid4().hex[:8]
            print(f"new session {sid}")
            continue
        if text.startswith("/cwd "):
            import os

            p = text[5:].strip()
            if p and os.path.isdir(p):
                cwd = os.path.abspath(p)
                print(f"cwd {cwd}")
            else:
                print(f"bad dir: {p}")
            continue
        streaming = False
        raw_buf = ""
        printed = 0
        last_reply = ""
        last_think = ""
        log(f"=== turn: {text!r} sid={sid} cwd={cwd}")
        try:
            async for ev in agent.run_text(text, session_id=sid, cwd=cwd,
                                           confirm_fn=confirm):
                d = ev.data or {}
                if ev.kind == "token":
                    piece = str(d.get("piece", "") or "")
                    if piece:
                        raw_buf += piece
                        # Answer-only streaming: think tags/trace never hit
                        # the reply line (split handles unclosed <think>).
                        _, answer = split_thinking(raw_buf)
                        new = answer[printed:]
                        if new:
                            if not streaming:
                                print("> agent: ", end="", flush=True)
                                streaming = True
                            print(new, end="", flush=True)
                            printed += len(new)
                    continue
                if streaming:
                    print()  # close stream line before next event
                    streaming = False
                if ev.kind == "thinking":
                    think = str(d.get("text", "") or "")
                    if think and think != last_think:
                        last_think = think
                        print(f"  think: {think[:800]}")
                elif ev.kind == "action":
                    a = d.get("action", {})
                    print(f"  $ {a.get('action')}: {a.get('command') or a.get('path') or a.get('pattern') or (a.get('code') or '')[:60] or ''}")
                elif ev.kind == "observation":
                    print(f"  -> {str(d.get('observation', ''))[:1200]}")
                elif ev.kind in ("chat", "summary"):
                    reply = str(d.get("reply", "") or "")
                    if not reply.strip():
                        pass
                    elif printed > 0:
                        # Already streamed live above; don't reprint.
                        last_reply = reply
                    elif reply != last_reply:
                        last_reply = reply
                        print(f"> agent: {reply}")
                elif ev.kind in ("deny", "error", "stuck", "confirm"):
                    print(f"  [{ev.kind}] {d}")
                elif ev.kind == "queued":
                    print(f"  [queued #{d.get('position', '?')}] injected after active turn")
                if ev.kind != "thinking":
                    # thinking arrives mid-step (tokens -> thinking -> action/chat);
                    # keep raw_buf/printed until the step-ending event so the
                    # chat/summary dedup still knows the reply already streamed.
                    log(f"--- step end: {ev.kind} raw={raw_buf!r} data={str(d)[:1500]!r}")
                    raw_buf = ""
                    printed = 0
                else:
                    log(f"--- thinking: {str(d.get('text', ''))[:1500]!r}")
            if streaming:
                print()
        except Exception as exc:
            print(f"> error: {exc}")


async def main() -> None:
    ap = argparse.ArgumentParser(description="plain LLM REPL (no TUI)")
    ap.add_argument("--model", default="openbmb/MiniCPM5-1B")
    ap.add_argument("--backend", default="minicpm", help="minicpm | minicpm_q4k | qwen")
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-tools", action="store_true", help="raw generate, skip harness tool loop")
    ap.add_argument("--log", default="", help="log step raw outputs + events to FILE for debugging")
    ap.add_argument("--sandbox", action="store_true",
                    help="run exec tools inside a per-turn docker container (needs docker group)")
    args = ap.parse_args()

    llm = LlmModel(LlmConfig(model=args.model, backend=args.backend))
    if args.max_tokens != 48:
        # harness budgets floor at 320/256 anyway; raw loop uses it directly
        from dataclasses import replace

        llm.config = replace(llm.config, max_tokens=args.max_tokens)
    print(f"warming {args.model} [{args.backend}] ...")
    await llm.warm()
    from src.models.llm import assert_cuda_leg

    print(f"llm ready on {assert_cuda_leg(llm)}.")

    sandbox = None
    if args.sandbox:
        from src.sandbox.docker import SandboxConfig

        sandbox = SandboxConfig()
    logf = open(args.log, "a") if args.log else None
    try:
        if args.no_tools:
            await raw_loop(llm, args.max_tokens)
        else:
            await harness_loop(llm, args.cwd, logf=logf, sandbox=sandbox)
    finally:
        if logf is not None:
            logf.close()


if __name__ == "__main__":
    asyncio.run(main())
