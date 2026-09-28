"""L3 semantic memory: flat user facts, human-editable markdown.

No managed service (Mem0-style cloud is wrong for on-device): one
``facts.md`` with ``- subject — PREDICATE — object`` lines, optional
``<!-- source, date -->`` trailer. Upsert by (subject, predicate):
newer values replace older ones (truth anchoring at write time); the
packer injects the block facts-first with a conflict-wins line at read
time. Pure file ops, atomic writes, never raises.
"""
import os
import re
import tempfile
import time

__all__ = ["load_facts", "upsert_fact", "format_block", "parse_line",
           "ANCHOR_LINE"]

ANCHOR_LINE = ("Dated user facts below override older context on conflict; "
               "newer entries win.")
FACTS_HEADER = "# facts — user truths (auto-curated, human-editable)\n"


def parse_line(line: str):
    """'- s — P — o <!-- src -->' -> dict or None. Pure."""
    t = str(line or "").strip()
    if not t.startswith("- "):
        return None
    body = t[2:]
    src = ""
    m = re.search(r"<!--(.*?)-->\s*$", body)
    if m:
        src = m.group(1).strip()
        body = body[:m.start()].strip()
    parts = [p.strip() for p in body.split("—")]
    if len(parts) != 3 or not all(parts):
        # tolerate ascii -- and - separators
        parts = [p.strip() for p in re.split(r"\s+--\s+|\s+-\s+", body)]
        if len(parts) != 3 or not all(parts):
            return None
    return {"subject": parts[0], "predicate": parts[1].upper(),
            "object": parts[2], "source": src}


def load_facts(path: str = "memory/facts.md") -> list:
    """Read facts file -> [{subject, predicate, object, source}]."""
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                f_ = parse_line(line)
                if f_ is not None:
                    out.append(f_)
    except Exception:
        pass
    return out


def upsert_fact(path: str, subject: str, predicate: str, obj: str,
                source: str = "") -> bool:
    """Insert or replace (subject, predicate). Newest wins. Atomic."""
    subject, predicate, obj = (str(subject or "").strip(),
                               str(predicate or "").strip().upper(),
                               str(obj or "").strip())
    if not subject or not predicate or not obj:
        return False
    path = str(path or "memory/facts.md")
    try:
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        kept = [f for f in load_facts(path)
                if not (f["subject"].lower() == subject.lower()
                        and f["predicate"] == predicate)]
        stamp = time.strftime("%Y-%m-%d")
        src = f"{source}, {stamp}" if source else stamp
        kept.append({"subject": subject, "predicate": predicate,
                     "object": obj, "source": src})
        lines = [FACTS_HEADER]
        for f in kept:
            lines.append(f"- {f['subject']} — {f['predicate']} — "
                         f"{f['object']} <!-- {f['source']} -->\n")
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def format_block(facts: list, cap_chars: int = 1500) -> str:
    """Facts -> capped system-prompt block (facts-first + anchor)."""
    lines = [ANCHOR_LINE]
    used = len(ANCHOR_LINE)
    for f in facts or []:
        line = f"- {f['subject']} — {f['predicate']} — {f['object']}"
        if used + len(line) > cap_chars:
            room = cap_chars - used - 3
            if room > 20:
                lines.append(line[:room] + "...")
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)
