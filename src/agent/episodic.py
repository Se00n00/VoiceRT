"""L2 episodic memory: what happened, searchable by meaning.

Local-only by design: one sqlite file (no vector-DB server), embeddings
from the already-vendored bge-small-en-v1.5 (same model as the tool
router), cosine in numpy, recency-weighted. Keyword-overlap fallback
when the embedder is cold — recall degrades, never raises.

Write path: L1 eviction summarizes dropped turns -> store().
Read path: recall(query, k) -> top-k episode dicts for prompt injection.
"""
import os
import sqlite3
import threading
import time

__all__ = ["EpisodicStore", "summarize_turns"]

EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session TEXT NOT NULL DEFAULT '',
  at REAL NOT NULL,
  summary TEXT NOT NULL,
  embedding BLOB,
  n_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_episodes_at ON episodes(at);
"""


def summarize_turns(turns: list) -> str:
    """Extractive fallback summary: first lines, capped. Pure."""
    bits = []
    for m in turns or []:
        if isinstance(m, dict):
            t = str(m.get("content", "") or "").strip().split("\n")[0]
        else:
            t = str(m or "").strip().split("\n")[0]
        if t:
            bits.append(t[:200])
        if sum(len(b) for b in bits) > 800:
            break
    return " | ".join(bits)[:1000]


class EpisodicStore:
    """Thread-safe sqlite episode store with cosine recall."""

    def __init__(self, path: str = "memory/episodic.db", embed_fn=None):
        self.path = str(path or "memory/episodic.db")
        self._embed_fn = embed_fn
        self._lock = threading.Lock()
        self._model = None
        self._tok = None

    # -- setup ------------------------------------------------------
    def _connect(self):
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        cx = sqlite3.connect(self.path, timeout=10.0)
        cx.executescript(SCHEMA)
        return cx

    def _embed(self, text: str):
        if self._embed_fn is not None:
            try:
                v = self._embed_fn(text)
                return list(v) if v is not None else None
            except Exception:
                return None
        if self._model is None:
            try:
                from transformers import AutoModel, AutoTokenizer

                self._tok = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
                self._model = AutoModel.from_pretrained(EMBED_MODEL_ID)
                self._model.eval()
            except Exception:
                return None
        try:
            import torch

            ids = self._tok(str(text or "")[:1000], return_tensors="pt",
                            truncation=True, max_length=512)
            with torch.no_grad():
                out = self._model(**ids).last_hidden_state[:, 0][0]
            import torch.nn.functional as _F

            return _F.normalize(out, p=2, dim=1)[0].tolist()
        except Exception:
            return None

    # -- write ------------------------------------------------------
    def store(self, summary: str, session: str = "") -> int:
        """Persist one episode. Returns row id (-1 on failure)."""
        summary = str(summary or "").strip()[:2000]
        if not summary:
            return -1
        vec = self._embed(summary)
        blob = None
        if vec:
            try:
                import numpy as _np

                blob = _np.asarray(vec, dtype="<f4").tobytes()
            except Exception:
                blob = None
        try:
            with self._lock:
                cx = self._connect()
                cur = cx.execute(
                    "INSERT INTO episodes (session, at, summary, embedding,"
                    " n_tokens) VALUES (?,?,?,?,?)",
                    (str(session or ""), time.time(), summary, blob,
                     max(1, len(summary) // 4)))
                cx.commit()
                rowid = cur.lastrowid
                cx.close()
            return int(rowid)
        except Exception:
            return -1

    # -- read -------------------------------------------------------
    @staticmethod
    def _keyword_score(query: str, summary: str) -> float:
        qw = {w.lower() for w in str(query or "").split() if len(w) > 2}
        sw = {w.lower() for w in str(summary or "").split() if len(w) > 2}
        if not qw or not sw:
            return 0.0
        return len(qw & sw) / len(qw)

    def recall(self, query: str, k: int = 3, session: str | None = None,
               max_age_s: float = 0) -> list:
        """Top-k episodes: cosine (embeddings present) blended with keyword
        overlap, recency-weighted. Returns [{id, session, at, summary,
        score}]. Never raises."""
        try:
            with self._lock:
                cx = self._connect()
                q = ("SELECT id, session, at, summary, embedding FROM episodes"
                     " ORDER BY at DESC LIMIT 2000")
                rows = cx.execute(q).fetchall()
                cx.close()
        except Exception:
            return []
        if session is not None:
            rows = [r for r in rows if r[1] == session]
        if max_age_s and max_age_s > 0:
            cutoff = time.time() - max_age_s
            rows = [r for r in rows if r[2] >= cutoff]
        qv = self._embed(query)
        now = time.time()
        scored = []
        for rid, sess, at, summary, blob in rows:
            cos = 0.0
            if qv and blob:
                try:
                    import numpy as _np

                    v = _np.frombuffer(blob, dtype="<f4>")
                    q = _np.asarray(qv, dtype=float)
                    if v.shape == q.shape and float(_np.dot(v, v)) > 0:
                        cos = float(_np.dot(v, q) / (
                            (_np.dot(v, v) ** 0.5) * (_np.dot(q, q) ** 0.5)
                            + 1e-9))
                except Exception:
                    cos = 0.0
            kw = self._keyword_score(query, summary)
            recency = 1.0 / (1.0 + max(0.0, now - at) / 86400.0)
            score = 0.6 * max(0.0, cos) + 0.3 * kw + 0.1 * recency
            scored.append((score, rid, sess, at, summary))
        scored.sort(key=lambda t: -t[0])
        return [{"id": r, "session": s, "at": a, "summary": sm,
                 "score": round(sc, 4)}
                for sc, r, s, a, sm in scored[:max(1, int(k))]]

    def count(self) -> int:
        try:
            with self._lock:
                cx = self._connect()
                n = cx.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                cx.close()
            return int(n)
        except Exception:
            return 0
