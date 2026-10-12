"""
Stock cloud chat models behind one backend switch.

contract, dependencies, who calls it: make_chat_model(backend) returns a
LangChain BaseChatModel for groq/gemini; depends on langchain-groq and
langchain-google-genai; used by console_cli, VoiceAgent, eval lib.
"""

import os

__all__ = [
    "CLOUD_BACKENDS",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GROQ_MODEL",
    "chat_text",
    "is_cloud_backend",
    "make_chat_model",
]

CLOUD_BACKENDS = ("groq", "gemini")

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


def _repo_root() -> str:
    """Repo root for the gitignored .env lookup.

    ..
    """
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def _load_dotenv() -> None:
    """Load repo-root .env once, ignoring a missing file.

    ..
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=os.path.join(_repo_root(), ".env"))
    except Exception:
        pass


def is_cloud_backend(backend: str) -> bool:
    """Whether the backend id selects a stock cloud model.

    ..
    """
    return str(backend or "").lower() in CLOUD_BACKENDS


def chat_text(content) -> str:
    """LangChain content (str or list blocks) -> plain text.

    ..
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict):
                t = b.get("text", "")
                if isinstance(t, str) and t:
                    parts.append(t)
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return str(content or "")


def make_chat_model(backend: str, model: str = "") -> object:
    """Build the stock chat model for groq or gemini.

    ..
    """
    name = str(backend or "").lower()
    if name == "groq":
        from langchain_groq import ChatGroq

        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            _load_dotenv()
            key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise SystemExit(
                "ABORT: no Groq API key. Set GROQ_API_KEY in the "
                "environment or the gitignored .env at the repo root."
            )
        return ChatGroq(
            model=str(model or DEFAULT_GROQ_MODEL),
            api_key=key,
            temperature=0.0,
            reasoning_effort="low",
        )
    if name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            _load_dotenv()
            key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not key:
            raise SystemExit(
                "ABORT: no Gemini API key. Set GEMINI_API_KEY in the "
                "environment or the gitignored .env at the repo root."
            )
        return ChatGoogleGenerativeAI(
            model=str(model or DEFAULT_GEMINI_MODEL),
            api_key=key,
            temperature=0.0,
        )
    raise ValueError(f"unknown cloud backend {backend!r}")
