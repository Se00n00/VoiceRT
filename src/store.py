"""
Local session store in ~/.local/share/voicert. Plus file metadata:
contract sqlite mainstore + timestamp log + tool spill files,
dependencies stdlib only, who calls it server bootstrap and harness.
"""

import hashlib
import json
import os
import sqlite3
import time

from src.paths import VoicertPaths

__all__ = ["VoicertStore"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  sid TEXT PRIMARY KEY,
  title TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL DEFAULT 0,
  messages TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS tool_outputs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sid TEXT NOT NULL DEFAULT '',
  name TEXT NOT NULL DEFAULT '',
  ts REAL NOT NULL DEFAULT 0,
  path TEXT NOT NULL DEFAULT '',
  size INTEGER NOT NULL DEFAULT 0
);
"""


class VoicertStore:
    """SQLite mainstore plus log and tool-output spill.

    ..
    """

    def __init__(self, data_dir: str = "") -> None:
        """Bind to a data dir, default XDG data dir.

        ..
        """
        self.data_dir = os.path.expanduser(data_dir) if data_dir else VoicertPaths.data_dir()
        self.db_path = os.path.join(self.data_dir, "voicert.db")
        self.log_path = os.path.join(self.data_dir, "voicert.log")
        self.spill_dir = os.path.join(self.data_dir, "tool_outputs")

    def ensure(self) -> str:
        """Create data dir, tables and spill dir.

        ..
        """
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.spill_dir, exist_ok=True)
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(SCHEMA)
            con.commit()
        finally:
            con.close()
        return self.data_dir

    def log(self, level: str, message: str, sid: str = "") -> None:
        """Append one timestamped log line, never raises.

        ..
        """
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            line = f"{ts} [{str(level or 'info').upper()}]"
            if sid:
                line += f" {sid}"
            line += f" {str(message or '')[:2000]}\n"
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            pass

    def save_session(self, sid: str, messages: list, title: str = "") -> bool:
        """Upsert one session row, True on success.

        ..
        """
        if not str(sid or "").strip():
            return False
        try:
            self.ensure()
            con = sqlite3.connect(self.db_path)
            try:
                con.execute(
                    "INSERT INTO sessions (sid, title, updated_at, messages)"
                    " VALUES (?, ?, ?, ?)"
                    " ON CONFLICT(sid) DO UPDATE SET title=excluded.title,"
                    " updated_at=excluded.updated_at, messages=excluded.messages",
                    (
                        str(sid),
                        str(title or ""),
                        time.time(),
                        json.dumps(list(messages or [])),
                    ),
                )
                con.commit()
            finally:
                con.close()
            return True
        except (OSError, sqlite3.Error, ValueError, TypeError):
            return False

    def load_session(self, sid: str) -> dict:
        """Load one session row or empty mapping.

        ..
        """
        out: dict = {}
        try:
            if not os.path.isfile(self.db_path):
                return out
            con = sqlite3.connect(self.db_path)
            try:
                row = con.execute(
                    "SELECT title, updated_at, messages FROM sessions WHERE sid = ?",
                    (str(sid),),
                ).fetchone()
            finally:
                con.close()
        except (OSError, sqlite3.Error):
            return out
        if not row:
            return out
        try:
            messages = json.loads(row[2] or "[]")
        except ValueError:
            messages = []
        if not isinstance(messages, list):
            messages = []
        return {"sid": str(sid), "title": str(row[0] or ""), "messages": messages}

    def spill(self, sid: str, name: str, text: str, limit: int = 8000) -> str:
        """Spill long tool output to a file, path or empty.

        ..
        """
        body = str(text or "")
        if len(body) <= max(1000, int(limit or 8000)):
            return ""
        try:
            os.makedirs(self.spill_dir, exist_ok=True)
            digest = hashlib.sha256(body.encode()).hexdigest()[:12]
            fname = f"{int(time.time())}_{digest}.txt"
            path = os.path.join(self.spill_dir, fname)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"sid={sid} name={name}\n\n")
                fh.write(body)
            con = sqlite3.connect(self.db_path)
            try:
                con.execute(
                    "INSERT INTO tool_outputs (sid, name, ts, path, size)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (str(sid or ""), str(name or ""), time.time(), path, len(body)),
                )
                con.commit()
            finally:
                con.close()
            return path
        except (OSError, sqlite3.Error, ValueError):
            return ""
