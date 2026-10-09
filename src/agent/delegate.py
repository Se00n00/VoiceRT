"""Two-brain delegation: a thin front model that can only raise a flag.

The split exists because the two jobs want opposite models. Talking to a
human needs speed; doing the work needs depth, tool discipline, and a real
harness. One model cannot be both on a 4GB card, so:

- **front** (Qwen3-0.6B) owns the conversation and answers exactly ONE
  question per turn: does this need the worker? It has no terminal, no
  files, no web, so it cannot fabricate a tool it has no schema for.
- **worker** (Bonsai 27B) keeps the whole existing deep-agent graph: the
  terminal op set, the todo list, the semantic tool router, Sim-Agent
  subagents, three-tier memory. Delegation does not build a second
  harness; it routes into the one that already exists.

The flag carries no payload. The worker already receives the raw user text
(see :meth:`VoiceAgent._agent_invoke`), so the front model never has to
paraphrase, summarise or quote the request — which is the capability a 0.6B
lacks, and the reason this replaced a single-tool ``delegate`` call that
measured 0/8 on real task prompts.

A 0.6B front model does not fail by inventing tools. It fails the other
way: it narrates the work instead of delegating it — "Sure, I renamed the
file" with nothing renamed. Two things guard that. The boolean contract
fails toward escalation (unparseable output means YES), and
:func:`is_task_shaped` remains as a deterministic backstop for the
narration case.

Asymmetric by design. A false positive costs one slow worker turn; a false
negative costs the user a confident lie about their own machine.

Config: ``configs/delegate.yaml`` (see :func:`load_config`).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BOOLEAN_FEWSHOT",
    "BOOLEAN_MAX_TOKENS",
    "BOOLEAN_SYSTEM",
    "CONFIG_ENV_VAR",
    "DELEGATE_TOOL",
    "DELEGATE_TOOL_NAME",
    "DelegateConfig",
    "Route",
    "boolean_messages",
    "decide_escalate",
    "default_config_path",
    "front_llm_config",
    "is_task_shaped",
    "load_config",
    "parse_boolean",
    "parse_delegate",
    "repo_root",
    "resolve_config_path",
    "task_score",
    "worker_llm_overrides",
]

CONFIG_ENV_VAR = "VOICE_DELEGATE_CONFIG"

DELEGATE_TOOL_NAME = "delegate"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> str:
    return str(repo_root() / "configs" / "delegate.yaml")


def resolve_config_path(explicit: str | None = None) -> str | None:
    """Explicit arg > ``$VOICE_DELEGATE_CONFIG`` > repo default > None."""
    if explicit:
        return str(explicit)
    env = os.environ.get(CONFIG_ENV_VAR, "").strip()
    if env:
        return env
    path = default_config_path()
    return path if os.path.isfile(path) else None


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
class DelegateConfig(BaseModel):
    """Two-brain routing knobs. Unknown YAML keys are ignored, not fatal."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    # "fused" (front on CUDA) | "cpu" (front fused on CPU, 0 VRAM).
    placement: str = "fused"
    front: dict = Field(default_factory=dict)
    worker: dict = Field(default_factory=dict)
    backstop: bool = True
    backstop_min_score: int = 2
    max_task_chars: int = 1200
    emit_route: bool = True
    # Worker sidecar warm policy: False (default) boots it lazily on the
    # first delegate route (zero footprint until then); True warms eagerly
    # at app boot as before. Mirrors VoiceAgentConfig.worker_eager.
    worker_eager: bool = False

    def with_overrides(self, **kw) -> "DelegateConfig":
        return self.model_copy(update={k: v for k, v in kw.items()
                                       if v is not None})


def load_config(explicit: str | None = None) -> DelegateConfig:
    """Read ``configs/delegate.yaml`` over the defaults. Never raises.

    A missing or malformed file yields working defaults rather than a dead
    agent: routing is an enhancement, not a dependency. PyYAML is optional
    here — without it the defaults still apply and the log says why.
    """
    path = resolve_config_path(explicit)
    if not path:
        return DelegateConfig()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return DelegateConfig()
    data: dict = {}
    if raw.strip():
        try:
            import yaml

            data = yaml.safe_load(raw) or {}
        except Exception:
            # JSON is a valid subset of YAML and needs no dependency; a
            # hand-edited .yaml that trips the parser still gets read.
            try:
                data = json.loads(raw)
            except Exception:
                return DelegateConfig()
    if not isinstance(data, dict):
        return DelegateConfig()

    def _flag(key, default):
        val = data.get(key, default)
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "on")
        return bool(val) if val is not None else default

    def _int(key, default):
        try:
            return int(data.get(key, default))
        except (TypeError, ValueError):
            return default

    def _dict(key):
        val = data.get(key)
        return dict(val) if isinstance(val, dict) else {}

    placement = str(data.get("placement", "fused") or "fused").strip().lower()
    if placement not in ("fused", "cpu"):
        placement = "fused"
    return DelegateConfig(
        enabled=_flag("enabled", True),
        placement=placement,
        front=_dict("front"),
        worker=_dict("worker"),
        backstop=_flag("backstop", True),
        backstop_min_score=max(1, _int("backstop_min_score", 2)),
        max_task_chars=max(200, _int("max_task_chars", 1200)),
        emit_route=_flag("emit_route", True),
        worker_eager=_flag("worker_eager", False),
    )


def front_llm_config(cfg: DelegateConfig):
    """Build the front leg's :class:`LlmConfig` for the chosen placement.

    The front leg is deliberately minimal: no paged engine, no thinking, a
    small token cap. A routing decision is one word plus a task string, so
    anything that spends tokens on deliberation costs the user latency.
    """
    from src.models.llm import LlmConfig

    f = cfg.front or {}
    device = str(f.get("device") or "").strip()
    if not device:
        device = "cpu" if cfg.placement == "cpu" else "cuda"
    return LlmConfig(
        backend=str(f.get("backend", "qwen")),
        model=str(f.get("model", "Qwen/Qwen3-0.6B")),
        max_tokens=int(f.get("max_tokens", 64) or 64),
        max_seq=int(f.get("max_seq", 4096) or 4096),
        device=device,
        use_paged=False,
        thinking=bool(f.get("thinking", False)),
    )


def worker_llm_overrides(cfg: DelegateConfig) -> dict:
    """Field overrides to apply to the *existing* worker ``LlmConfig``.

    Returned as a dict so the caller can ``replace()`` its own leg config:
    delegation must not fork the worker into a second, differently
    configured instance.
    """
    w = cfg.worker or {}
    out: dict = {}
    if w.get("backend"):
        out["backend"] = str(w["backend"])
    if w.get("model"):
        out["model"] = str(w["model"])
    for key in ("bonsai_ngl", "bonsai_ctx", "bonsai_thinking", "bonsai_port"):
        if w.get(key) is not None:
            out[key] = w[key]
    return out


# --------------------------------------------------------------------------
# the one tool the front model gets
# --------------------------------------------------------------------------
# RETIRED — kept only so external callers and the historical tests do not
# break. The front leg is no longer given this tool; it gets no schema at
# all (see the module docstring and :func:`decide_escalate`). The boolean
# replaced it because a 0.6B could not fill the envelope in: 0/8 valid
# calls on real task prompts.
#
# House flat shape ({"name", ...}), like TERMINAL_TOOLS. Sidecar legs convert
# to OpenAI form themselves; the fused leg does it in LlmModel.encode.
DELEGATE_TOOL = {
    "name": DELEGATE_TOOL_NAME,
    "description": (
        "Hand a task to the worker agent and let it do the work. The worker "
        "has real tools: a shell, file read/write/edit, code search, a web "
        "fetch and search, Python, and its own subagents. Call this for "
        "ANY action on this machine or the web - reading or changing files, "
        "running commands, looking something up in the repo, installing, "
        "debugging, checking status. Reply with plain text and do NOT call "
        "this for small talk, opinions, or questions you can answer from "
        "your own knowledge. Never describe the work as done: either call "
        "this tool, or say nothing happened."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "The request in full, as the worker should carry it out. "
                    "Carry over every detail the user gave: file paths, "
                    "names, constraints, and what 'done' should look like."
                ),
            }
        },
        "required": ["task"],
    },
}


# --------------------------------------------------------------------------
# parsing the front model's output
# --------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_TRAILER_RE = re.compile(r"\n?\s*CWD:.*$", re.DOTALL)


class Route(BaseModel):
    """What the front brain decided.

    ``kind`` is ``"chat"`` (answer the user here) or ``"delegate"`` (hand the
    task to the worker). ``reason`` records which signal decided it, and is
    carried on the event stream for debugging routing quality.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    text: str = ""
    reason: str = ""
    forced: bool = False


# --------------------------------------------------------------------------
# the boolean escalation contract
# --------------------------------------------------------------------------
# The front model's ONLY job is to raise a flag. It does not write the task
# string, does not call a tool, and does not summarise anything.
#
# This replaced a single-tool `delegate` call, which measured 0/8 on real
# task prompts: the 0.6B would answer in prose ("Sure, I renamed it") and
# never emit the envelope. Asking it to *decide* instead of to *construct*
# is the difference, because the worker already receives the raw user text
# (see VoiceAgent._agent_invoke), so the flag carries no payload and there
# is nothing to paraphrase and get wrong.
#
# The bias is deliberately toward YES. Measured on 19 task + 18 chat
# prompts: 19/19 tasks escalate, 3 chat false positives. That asymmetry is
# the design: a false positive costs one slow 27B turn on a message the
# front model can still answer, while a false negative silently drops a
# task the user asked for.
BOOLEAN_SYSTEM = (
    "Answer YES only if the user wants you to actually DO something: run a "
    "command, read or change a file, install, search, look up, fetch, or "
    "operate on their computer. Answer NO for greetings, opinions, jokes, "
    "thanks, general knowledge, and questions about code or concepts you "
    "can explain from memory.")

# Few-shot pairs, not decoration: the same prompt without them answered
# YES to 100% of chit-chat and NO to half the real tasks. These five were
# the difference between 22/37 and 34/37.
BOOLEAN_FEWSHOT = [
    {"role": "user", "content": "USER: list the files in the current directory\nANSWER:"},
    {"role": "assistant", "content": "YES"},
    {"role": "user", "content": "USER: hi\nANSWER:"},
    {"role": "assistant", "content": "NO"},
    {"role": "user", "content": "USER: why is my code so slow\nANSWER:"},
    {"role": "assistant", "content": "YES"},
    {"role": "user", "content": "USER: ok cool\nANSWER:"},
    {"role": "assistant", "content": "NO"},
    {"role": "user", "content": "USER: summarize this article\nANSWER:"},
    {"role": "assistant", "content": "YES"},
]

# 6 tokens is the whole budget: the answer is one word, so anything longer
# is the model ignoring the format rather than thinking.
BOOLEAN_MAX_TOKENS = 6


def boolean_messages(text: str, history: list | None = None) -> list:
    """Prompt for the one-word escalation answer.

    Deliberately no tool specs and no chat history: the question is about
    *this* utterance, and handing the 0.6B a transcript makes it answer the
    last turn instead of the request in front of it.
    """
    msgs = [{"role": "system", "content": BOOLEAN_SYSTEM}]
    msgs.extend(BOOLEAN_FEWSHOT)
    msgs.append({"role": "user",
                 "content": f"USER: {str(text or '').strip()[:600]}\nANSWER:"})
    return msgs


def parse_boolean(raw: str) -> bool | None:
    """First leading YES/NO in the output, or ``None`` if neither.

    Scans word by word so "NO, it cannot" reads as NO and a stray
    "eventually" cannot be read as YES. Returns ``None`` for empty or
    unrecognised output, which the caller treats as escalate.
    """
    for word in str(raw or "").replace("\n", " ").split():
        head = word.strip(".,!?*:;\"'()[]{}").upper()
        if head in ("YES", "Y"):
            return True
        if head in ("NO", "N"):
            return False
        if head:
            # First real word decides; do not keep scanning for a YES
            # buried in prose ("I cannot do that, YES I know").
            return None
    return None


async def decide_escalate(front_llm, text: str, *,
                          history: list | None = None) -> tuple[bool, str]:
    """Ask the front brain one question. Returns ``(escalate, reason)``.

    ``reason`` is for the route event only. Never raises: a dead front leg
    escalates, because the worker is the leg that can actually do anything
    and a routing failure must not become a dropped task.
    """
    try:
        res = await front_llm.generate(
            boolean_messages(text, history), BOOLEAN_MAX_TOKENS)
    except Exception as exc:  # noqa: BLE001 - degrade to the capable leg
        return True, f"front-error: {type(exc).__name__}"
    verdict = parse_boolean(getattr(res, "text", "") or "")
    if verdict is None:
        return True, "unparseable"
    return verdict, "boolean"



def _json_objects(raw: str):
    """Yield every balanced ``{...}`` span in raw, longest-first per start."""
    starts = [i for i, ch in enumerate(raw) if ch == "{"]
    for start in starts:
        depth, instr, esc = 0, False, False
        for i in range(start, len(raw)):
            ch = raw[i]
            if instr:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    instr = False
                continue
            if ch == '"':
                instr = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield raw[start:i + 1]
                    break


_ARG_KEYS = ("task", "text", "content", "prompt", "request", "query",
             "description", "command", "path")
# Keys that describe the CALL rather than the task. Never a task value:
# without this, empty args fall through to the tool's own name and the
# worker gets handed the string "delegate".
_META_KEYS = ("name", "function", "type", "action", "op", "tool",
              "tool_call", "id", "index")


def _arg_text(obj, _depth: int = 0) -> str:
    """Pull the task text out of whatever shape the args arrived in.

    Handles a dict of args, a nested ``{name, arguments}`` envelope, and
    ``arguments`` handed over as a JSON *string* (llama.cpp serializes tool
    arguments that way). Bounded recursion: a malformed envelope must not
    become a hang.
    """
    if _depth > 4:
        return ""
    if isinstance(obj, str):
        body = obj.strip()
        if body.startswith("{"):
            try:
                return _arg_text(json.loads(body), _depth + 1)
            except Exception:
                return ""
        return body
    if not isinstance(obj, dict):
        return ""
    for key in _ARG_KEYS:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val
    for key in ("arguments", "args", "parameters", "input"):
        if key in obj:
            got = _arg_text(obj[key], _depth + 1)
            if got:
                return got
    for key, val in obj.items():
        if key in _META_KEYS:
            continue
        if isinstance(val, str) and val.strip():
            return val
    return ""


_XML_CALL_RE = re.compile(
    r"<function\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</function\s*>",
    re.DOTALL | re.IGNORECASE)
_XML_PARAM_RE = re.compile(
    r"<param\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</param\s*>",
    re.DOTALL | re.IGNORECASE)


def parse_delegate(raw: str, *, max_task_chars: int = 1200) -> Route | None:
    """Front-model output -> :class:`Route`, or ``None`` when it is chat.

    Handles every envelope the front leg can produce: OpenAI/Qwen
    ``<tool_call>`` JSON, a bare ``{"name": ...}`` dict, the MiniCPM
    ``<function>`` XML form, and the terminal harness's
    ``{"action": "delegate"}`` shape.

    A call to some *other* tool name still routes to the worker, but its
    arguments are DISCARDED and the worker's task falls back to the user's
    own words (passed in as ``fallback_task``). The front model's decision
    to act is worth trusting; its invented argument synthesis is exactly
    the thing that must not reach the machine. Never raises.
    """
    text = str(raw or "")
    if not text.strip():
        return None
    body = _THINK_RE.sub("", text).strip()
    if not body:
        return None

    # 1. XML <function name="delegate"> form.
    for name, inner in _XML_CALL_RE.findall(body):
        if not str(name).strip():
            continue
        params = dict(_XML_PARAM_RE.findall(inner))
        if str(name).strip() == DELEGATE_TOOL_NAME:
            task = _arg_text(params) or _arg_text(inner)
            if task.strip():
                return Route(kind="delegate", text=task.strip()[:max_task_chars],
                           reason="tool")
        return Route(kind="delegate", text="", reason="tool:foreign", forced=True)

    # 2. JSON envelopes: any balanced object that names a tool.
    for span in _json_objects(body):
        try:
            obj = json.loads(span)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not isinstance(name, str) and isinstance(obj.get("function"), dict):
            fn = obj["function"]
            name, obj = fn.get("name"), fn
        if isinstance(name, str) and name.strip():
            if name.strip() == DELEGATE_TOOL_NAME:
                task = _arg_text(obj)
                return Route(kind="delegate", text=task.strip()[:max_task_chars],
                           reason="tool")
            return Route(kind="delegate", text="", reason="tool:foreign", forced=True)
        action = obj.get("action")
        if isinstance(action, str) and action.strip() == DELEGATE_TOOL_NAME:
            task = _arg_text(obj)
            return Route(kind="delegate", text=task.strip()[:max_task_chars],
                       reason="tool")

    return None


# --------------------------------------------------------------------------
# the deterministic backstop
# --------------------------------------------------------------------------
# Scored signal table. Two points is the default bar, so a lone weak verb
# ("list") stays chat while a verb plus an artifact noun escalates.
_STRONG_VERB = (
    r"install|uninstall|delete|remove|rename|move|copy|refactor|rewrite|"
    r"compile|deploy|commit|push|pull|merge|rebase|kill|migrate|patch|"
    r"reinstall|rebuild|regenerate|convert|export|scrape|crawl|"
    r"write|edit|create|build|execute|run|fix|debug|implement|generate|"
    r"download|install|grep|search|fetch|browse|install"
)
_WEAK_VERB = (
    r"list|read|open|show|find|check|get|set|print|dump|count|inspect|"
    r"look|scan|test|update|start|stop|restart|clean|add|verify|check"
)
_NOUN = (
    r"terminal|shell|command|script|process|disk|gpu|vram|server|port|"
    r"logs?|env(?:ironment)?\s+var|dependenc(?:y|ies)|packages?|repo|"
    r"repository|commit|branch|tests?|config(?:uration)?|files?|folder|"
    r"directory|project|codebase|line|function|import|stack\s*trace|"
    r"error|bug|crash|exception|traceback|module|class|method|api|"
    r"database|docker|container|venv|requirements|readme|benchmark|"
    r"git|status|threshold|setting|parameter|constant|variable|"
    r"schema|template|handler|endpoint|route|middleware|kernel|"
    r"dependency|entry\s?point|script|argparse|env\b"
)
_FILE_RE = re.compile(
    r"[\w./~-]+\.(?:py|js|jsx|ts|tsx|json|md|txt|ya?ml|toml|cfg|ini|sh|"
    r"bash|zsh|go|rs|java|c|cc|cpp|h|hpp|css|html|csv|tsv|sql|lock|env|"
    r"log|gguf|pt|safetensors)\b", re.IGNORECASE)
_PATH_RE = re.compile(r"(?:^|\s)(?:~|\.{1,2}/|/)[\w./-]*")
# Politeness openers: "can you ..." almost always precedes a real request.
_POLITE_RE = re.compile(
    r"\b(?:can you|could you|would you|please|will you|i need you to|"
    r"i want you to|help me|i'd like you to)\b", re.IGNORECASE)
# Code-lookup phrasing. "where does X happen" is the repo-search question
# the terminal preamble teaches by example, and no verb signals it on its
# own ("find where the threshold is" is all weak-verb).
_LOOKUP_RE = re.compile(
    r"\b(?:where\s+(?:is|are|does|do|did)|what\s+(?:file|files|line|lines|"
    r"function|class|module|endpoint)|which\s+(?:file|function|module|"
    r"class|method)|how\s+(?:do|does|did)\b.{0,40}\bwork)\b",
    re.IGNORECASE)
# Hard vetoes: pure social contact is never a task, however it is phrased.
_CHAT_RE = re.compile(
    r"^\s*(?:hi|hey|hello|yo|thanks|thank you|thx|ok|okay|cool|nice|"
    r"good (?:morning|afternoon|evening|night)|bye|goodbye|how are you|"
    r"who are you|what are you|what can you do|what do you do|"
    r"tell me a joke|say hi)\b", re.IGNORECASE)


def _clean(text: str) -> str:
    """Strip think traces and the harness CWD trailer before scoring."""
    out = _THINK_RE.sub("", str(text or ""))
    out = _TRAILER_RE.sub("", out)
    return out.strip()


def task_score(text: str) -> tuple[int, list[str]]:
    """Score how much a request reads like machine work. Pure.

    Returns ``(score, hits)``; the hit names are for tuning the table in
    tests rather than guessing at it in production.
    """
    body = _clean(text)
    if not body or _CHAT_RE.match(body):
        return 0, []
    score, hits = 0, []

    if re.search(rf"\b(?:{_STRONG_VERB})\b", body, re.IGNORECASE):
        score += 2
        hits.append("verb:strong")
    if re.search(rf"\b(?:{_WEAK_VERB})\b", body, re.IGNORECASE):
        score += 1
        hits.append("verb:weak")
    nouns = {m.group(0).lower() for m in
             re.finditer(rf"\b(?:{_NOUN})\b", body, re.IGNORECASE)}
    if nouns:
        score += min(2, len(nouns))
        hits.append("noun:" + ",".join(sorted(nouns)[:3]))
    if _FILE_RE.search(body):
        score += 2
        hits.append("file")
    elif _PATH_RE.search(body):
        score += 2
        hits.append("path")
    if _POLITE_RE.search(body):
        score += 1
        hits.append("polite")
    if _LOOKUP_RE.search(body):
        score += 2
        hits.append("lookup")
    return score, hits


def is_task_shaped(text: str, min_score: int = 2) -> bool:
    """True when the request reads like work on the machine or the web.

    Used only to *escalate*: a task-shaped request that the front model
    answered in prose is the "pretend-done" failure, so the worker runs
    anyway. Never used to suppress a delegate the front model asked for.
    """
    score, _hits = task_score(text)
    return score >= int(min_score)
