"""GAIA grading: normalized final-answer match + attachment-aware file checks."""
import os
import re
import string

_WS_RE = re.compile(r"\s+")


def normalize(text) -> str:
    text = str(text or "").strip().lower()
    text = "".join(c for c in text if c not in string.punctuation)
    return _WS_RE.sub(" ", text).strip()


def grade_reply(reply, expected) -> dict:
    want, got = normalize(expected), normalize(reply)
    ok = bool(want) and (want == got or want in got)
    return {"check": "final_answer", "ok": ok,
            "detail": f"expected {str(expected)[:120]!r} in reply"}


def grade_task(workdir, reply, task) -> list:
    """Pure oracle grading for one GAIA task. Never raises."""
    results = []
    try:
        results.append(grade_reply(reply, task.get("answer")))
        if task.get("file_name"):
            exists = os.path.exists(os.path.join(workdir, task["file_name"]))
            results.append({"check": "attachment_staged", "ok": exists,
                            "detail": f"{task['file_name']} staged: {exists}"})
    except Exception as exc:
        results.append({"check": "grader", "ok": False,
                        "detail": f"grader error: {exc}"[:200]})
    return results
