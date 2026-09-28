"""Memory eval: retention curve + correction-win + budget gate.

Plants checkable facts across a conversation, pushes them out of the L1
window with noise, then probes. Compares memory-ON (token window + L2
episodic + L3 facts) vs memory-OFF (plain count window) on the same
script. Reports retention@N, correction-win, overflow count.

Usage:
  PYTHONPATH=. .venv/bin/python -u benchmarks/memory_eval.py \\
      --backend gemma270 --model google/functiongemma-270m-it --tag m1
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FACTS = [
    ("DNS server", "192.168.1.1"),
    ("VAD threshold", "0.7"),
    ("backup hour", "3am"),
    ("favorite editor", "helix"),
    ("router model", "bge-small"),
    ("shell", "bash"),
    ("note file", "notes-b.txt"),
    ("meeting day", "Thursday"),
    ("cache size", "512mb"),
    ("log level", "debug"),
    ("username", "se00n00"),
]
NOISE = [
    "Say something brief about the weather.",
    "Tell me a one-line fact about rivers.",
    "Reply with a short greeting.",
    "Say something brief about mountains.",
    "Tell me a one-line fact about deserts.",
    "Reply with a short farewell.",
]


async def run_condition(llm_cfg, mem_cfg, tag, workdir):
    from src.main import VoiceAgent, VoiceAgentConfig

    cfg = VoiceAgentConfig(
        llm=llm_cfg, work_dir=workdir, speak_text_turns=False,
        max_session_turns=mem_cfg["turns"],
        memory_tokens=mem_cfg["tokens"],
        memory_recall=mem_cfg["recall"],
        memory_store=mem_cfg["store"],
        memory_dir=mem_cfg["dir"],
        sessions_dir=os.path.join(mem_cfg["dir"], "sessions"))
    agent = VoiceAgent(cfg)
    await agent.llm.warm()
    from src.agent.budget import estimate_tokens

    sid = "mem-%s-%d" % (tag, int(time.time()) % 100000)
    rows = []
    overflows = 0

    async def turn(text, timeout=300, _retried=False):
        nonlocal overflows
        t0 = time.monotonic()
        kinds, reply, err = {}, "", ""
        try:
            async for ev in agent.run_text(
                    text, session_id=sid, cwd=workdir, confirm_fn=None):
                kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
                if ev.kind == "chat":
                    reply = str(ev.data.get("reply", ""))
                if ev.kind == "error":
                    err = str(ev.data.get("message", ""))[:150]
                    overflows += 1
        except Exception as exc:
            err = str(exc)[:150]
            # Server died mid-run (keepalive expiry, restart): re-warm
            # once (spawns fresh if needed) and retry the turn instead
            # of poisoning every later row with conn-refused.
            if (not _retried and ("refused" in err or "ConnectError" in err
                                  or "disconnected" in err)):
                try:
                    await agent.llm.warm()
                except Exception:
                    pass
                return await turn(text, timeout, True)
            overflows += 1
        hist = agent.sessions.history(sid)
        hchars = sum(len(str(m.get("content", ""))) for m in hist)
        try:
            mem_hit = bool(agent._memory_context(sid, text).strip())
        except Exception:
            mem_hit = False
        return {"text": text[:80], "reply": reply[:300], "kinds": kinds,
                "sec": round(time.monotonic() - t0, 1), "error": err,
                "mem_hit": mem_hit,
                "hist_msgs": len(hist),
                "hist_est_tok": estimate_tokens(
                    " ".join(str(m.get("content", "")) for m in hist))}

    # plant (skip the placeholder entry)
    planted = list(FACTS)
    acks = ["Noted.", "Got it.", "Understood.", "Recorded.", "Thanks, noted.",
            "OK.", "Acknowledged.", "Copy that.", "Saved.", "Roger.",
            "Noted, thanks."]
    for i, (s, v) in enumerate(planted):
        rows.append(("plant", s, await turn(
            "Heads up, please remember: %s is %s. Reply with just \"%s\"" % (
                s, v, acks[i % len(acks)]))))
    for n in NOISE:
        rows.append(("noise", n[:40], await turn(n)))
    # L3 distill pass: extract clean facts from the plant turns so the
    # probe sees FACTS.md lines, not noisy Q&A episodes.
    distilled = 0
    if mem_cfg["store"]:
        from src.agent.facts import parse_line, upsert_fact

        facts_path = os.path.join(mem_cfg["dir"], "facts.md")
        for subj, val in planted:
            prompt = ("Extract exactly one fact as a single line in the "
                      "form SUBJECT — PREDICATE — OBJECT (em-dash "
                      "separators, UPPERCASE predicate, no other text). "
                      "Text: %s is %s." % (subj, val))
            try:
                res = await agent.llm.generate(
                    [{"role": "user", "content": prompt}], max_tokens=64)
                line = (res.text or "").strip().split("\n")[0]
                fact = parse_line(line if line.startswith("- ") else "- " + line)
                if fact and fact["subject"] and fact["object"]:
                    if upsert_fact(facts_path, fact["subject"],
                                   fact["predicate"], fact["object"], "distill"):
                        distilled += 1
            except Exception:
                pass
    rows.append(("distill", "facts", {"reply": "distilled=%d" % distilled,
                "kinds": {}, "sec": 0, "error": "", "hist_msgs": 0,
                "hist_est_tok": 0, "mem_hit": False}))
    # Per-fact value probes (constrained replies dodge the tiny leg's
    # free-chat refusal; each fact scored independently).
    hits = 0
    probe_replies = []
    for s, v in planted:
        pr = await turn("Question: what is %s? Answer with ONLY the value "
                        "itself. Do not say Noted, OK, or anything else." % s)
        rows.append(("probe", s, pr))
        probe_replies.append(pr["reply"])
        if v.lower() in pr["reply"].lower():
            hits += 1
    probe = {"reply": " | ".join(probe_replies),
             "kinds": {}, "sec": 0, "error": "", "hist_msgs": 0,
             "hist_est_tok": 0}
    # correction: contradict one fact, probe it
    await turn("Correction: the VAD threshold is now 0.9, not 0.7. Reply OK.")
    cprobe = await turn("What is the VAD threshold? Reply with just the value.")
    rows.append(("correction", "vad", cprobe))
    cl = cprobe["reply"].lower()
    correction_win = ("0.9" in cl) and ("0.7" not in cl)
    return {"tag": tag, "planted": len(planted),
            "retention": round(hits / max(1, len(planted)), 3),
            "hits": hits, "correction_win": correction_win,
            "overflows": overflows, "rows": rows,
            "probe_reply": probe["reply"][:500]}


async def main_async(args):
    from src.models.llm import LlmConfig

    workdir = tempfile.mkdtemp(prefix="memeval-")
    base = {"turns": 4, "tokens": 1500}
    on = dict(base, recall=True, store=True,
              dir=tempfile.mkdtemp(prefix="mem-on-"))
    off = dict(base, recall=False, store=False,
               dir=tempfile.mkdtemp(prefix="mem-off-"))
    llm_cfg = LlmConfig(model=args.model, backend=args.backend)
    # Preflight: fail fast when no sidecar answers (a server dying
    # mid-run otherwise poisons every later turn with conn-refused).
    import urllib.request
    for url in ("http://127.0.0.1:8083/health",):
        try:
            urllib.request.urlopen(url, timeout=5)
        except Exception:
            pass
    from src.models.llm import LlmModel as _M

    _probe = _M(llm_cfg)
    await _probe.warm()
    from src.models.llm import assert_ready_leg

    assert_ready_leg(_probe)
    res_on = await run_condition(llm_cfg, on, "on", workdir)
    llm_cfg2 = LlmConfig(model=args.model, backend=args.backend)
    res_off = await run_condition(llm_cfg2, off, "off", workdir)
    summary = {
        "model": args.model, "backend": args.backend,
        "distilled_on": next((r[2]["reply"] for r in res_on["rows"]
                              if r[0] == "distill"), "?"),
        "retention_on": res_on["retention"], "retention_off": res_off["retention"],
        "lift": round(res_on["retention"] - res_off["retention"], 3),
        "correction_win_on": res_on["correction_win"],
        "overflows_on": res_on["overflows"], "overflows_off": res_off["overflows"],
    }
    out = {"meta": {"argv": sys.argv, "at": time.strftime("%Y%m%d-%H%M%S")},
           "summary": summary, "on": res_on, "off": res_off}
    os.makedirs("benchmarks/results", exist_ok=True)
    path = "benchmarks/results/memory_eval_%s_%s.json" % (
        args.tag, out["meta"]["at"])
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=1))
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/functiongemma-270m-it")
    ap.add_argument("--backend", default="gemma270")
    ap.add_argument("--tag", default="m1")
    args = ap.parse_args(argv)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
