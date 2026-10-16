"""
Chat model factory behind one match switch for cloud backends.

contract, dependencies, who calls it: ChatModel.get(backend) returns a
LangChain BaseChatModel with key from arg/env/keychain/.env and model
from arg/global-file/.env/builtin; is_cloud_backend gates the switch
and chat_text flattens replies; defaults live in
~/.config/voicert/models.yaml seeded on first start and edited via
save_model, keys via save_key into .env; depends on langchain provider
packages, python-dotenv and src.agent.keystore + src.app_config; used
by console_cli, VoiceAgent, eval instead of branching inline.
"""

import os

from dotenv import load_dotenv, set_key
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.agent.keystore import get_secret
from src.app_config import AppConfig

__all__ = ["ChatModel"]


class ChatModel:
    """One factory for every cloud chat backend."""

    BACKENDS = (
        "groq",
        "gemini",
        "anthropic",
        "openai",
        "openrouter",
        "deepseek",
        "openai_compat",
    )

    DEFAULTS = {
        "groq": "openai/gpt-oss-120b",
        "gemini": "gemini-3.5-flash",
        "anthropic": "claude-sonnet-4-5",
        "openai": "gpt-4o-mini",
        "openrouter": "deepseek/deepseek-chat",
        "deepseek": "deepseek-chat",
    }

    KEY_ENVS = {
        "groq": ("GROQ_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "openrouter": ("OPENROUTER_API_KEY",),
        "deepseek": ("DEEPSEEK_API_KEY",),
        "openai_compat": ("OPENAI_COMPAT_API_KEY",),
    }

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    GLOBAL_DIR = "~/.config/voicert"
    GLOBAL_FILE = "models.yaml"

    MODEL_ENVS = {
        "groq": ("GROQ_MODEL",),
        "gemini": ("GEMINI_MODEL",),
        "anthropic": ("ANTHROPIC_MODEL",),
        "openai": ("OPENAI_MODEL",),
        "openrouter": ("OPENROUTER_MODEL",),
        "deepseek": ("DEEPSEEK_MODEL",),
        "openai_compat": ("OPENAI_COMPAT_MODEL",),
    }

    _LOADED = False

    @staticmethod
    def _dotenv_path() -> str:
        """Repo-root .env path for key load/save.

        Two dirs up from this file is the root.
        ..
        """
        here = os.path.abspath(__file__)
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(here))), ".env")

    @classmethod
    def _load_env(cls) -> None:
        """Load repo-root .env into os.environ once.

        Missing file is fine; process env keeps winning.
        ..
        """
        if cls._LOADED:
            return
        cls._LOADED = True
        path = cls._dotenv_path()
        if os.path.isfile(path):
            load_dotenv(dotenv_path=path)

    @staticmethod
    def _global_path() -> str:
        """Global models file path, seeded on first open.

        XDG config dir; VOICERT_CONFIG_DIR overrides for tests.
        ..
        """
        from src.paths import VoicertPaths

        return VoicertPaths.models_path()

    @classmethod
    def _seed_global(cls, path: str) -> dict:
        """Write builtin defaults to a missing global file.

        Returns the seeded mapping for immediate use.
        ..
        """
        AppConfig.ensure_global()
        return AppConfig.load_models()

    @classmethod
    def _global_models(cls) -> dict:
        """Global model map, seeded on first open, else defaults.

        A corrupt file falls back to builtins without crashing.
        ..
        """
        return AppConfig.load_models()

    @classmethod
    def default_model(cls, backend: str) -> str:
        """Default model id for one backend id.

        Global file wins, then .env model var, then builtin.
        ..
        """
        cls._load_env()
        name = str(backend or "").lower().strip()
        hit = str(cls._global_models().get(name, "") or "").strip()
        if hit:
            return hit
        for env in cls.MODEL_ENVS.get(name, ()):
            val = os.environ.get(env, "").strip()
            if val:
                return val
        if name == "openai_compat":
            return ""
        return cls.DEFAULTS.get(name, "")

    @classmethod
    def save_model(cls, backend: str, model: str) -> str:
        """Save one backend default model into the global file.

        Returns the global file path written for confirmation.
        ..
        """
        return AppConfig.save_model(backend, model)

    @classmethod
    def save_key(cls, backend: str, api_key: str) -> str:
        """Save one backend key into .env and process env.

        Returns the .env path written for confirmation.
        ..
        """
        name = str(backend or "").lower().strip()
        if name not in cls.KEY_ENVS:
            raise ValueError(f"unknown chat backend {backend!r}")
        value = str(api_key or "").strip()
        if not value:
            raise ValueError("refusing to save an empty API key")
        env = cls.KEY_ENVS[name][0]
        path = cls._dotenv_path()
        set_key(path, env, value)
        os.environ[env] = value
        return path

    @staticmethod
    def _resolve_key(backend: str, api_key: str = "") -> str:
        """Resolve the API key for one backend id.

        Explicit arg wins, else env -> keychain -> .env.
        ..
        """
        if str(api_key or "").strip():
            return str(api_key).strip()
        name = str(backend or "").lower().strip()
        envs = ChatModel.KEY_ENVS.get(name, ())
        if not envs:
            return ""
        ChatModel._load_env()
        return get_secret(name, *envs)

    @staticmethod
    def _need_key(backend: str, api_key: str, label: str) -> str:
        """Resolved key or abort telling where to put it.

        Empty key exits with the env var hint.
        ..
        """
        hit = ChatModel._resolve_key(backend, api_key)
        if hit:
            return hit
        envs = ChatModel.KEY_ENVS.get(str(backend or "").lower(), ("API_KEY",))
        raise SystemExit(
            f"ABORT: no {label} API key. Set {envs[0]} in the environment."
        )

    @staticmethod
    def _openai_compat(model: str, key: str, base_url: str | None, headers: dict | None = None) -> object:
        """Build ChatOpenAI for one OpenAI-compatible endpoint.

        Never touches the network, only binds model/key/url.
        ..
        """
        kwargs: dict = {"model": str(model), "api_key": key, "temperature": 0.0}
        if base_url:
            kwargs["base_url"] = str(base_url)
        if headers:
            kwargs["default_headers"] = dict(headers)
        return ChatOpenAI(**kwargs)

    @classmethod
    def get(cls, backend: str, model: str = "", api_key: str = "", base_url: str = "") -> object:
        """Build the stock chat model for one backend id.

        Backend selects the branch via match; model/key/url fall
        ..
        """
        cls._load_env()
        name = str(backend or "").lower().strip()
        want = str(model or "").strip()
        url = str(base_url or "").strip()
        match name:
            case "groq":
                key = cls._need_key("groq", api_key, "Groq")
                return ChatGroq(
                    model=want or cls.default_model("groq"),
                    api_key=key,
                    temperature=0.0,
                    reasoning_effort="low",
                )
            
            case "gemini":
                key = cls._need_key("gemini", api_key, "Gemini")
                return ChatGoogleGenerativeAI(
                    model=want or cls.default_model("gemini"),
                    api_key=key,
                    temperature=0.0,
                )
            
            case "anthropic":
                key = cls._need_key("anthropic", api_key, "Anthropic")
                return ChatAnthropic(
                    model=want or cls.default_model("anthropic"),
                    api_key=key,
                    temperature=0.0,
                )
            
            case "openai":
                key = cls._need_key("openai", api_key, "OpenAI")
                return cls._openai_compat(want or cls.default_model("openai"), key, None)
            
            case "openrouter":
                key = cls._need_key("openrouter", api_key, "OpenRouter")
                base = url or os.environ.get("OPENROUTER_BASE_URL", "").strip() or cls.OPENROUTER_BASE_URL
                headers: dict = {}
                site = os.environ.get("OPENROUTER_SITE_URL", "").strip()
                if site:
                    headers["HTTP-Referer"] = site
                title = os.environ.get("OPENROUTER_APP_NAME", "").strip()
                if title:
                    headers["X-Title"] = title
                return cls._openai_compat(want or cls.default_model("openrouter"), key, base, headers or None)
            
            case "deepseek":
                key = cls._need_key("deepseek", api_key, "DeepSeek")
                base = url or os.environ.get("DEEPSEEK_BASE_URL", "").strip() or cls.DEEPSEEK_BASE_URL
                return cls._openai_compat(want or cls.default_model("deepseek"), key, base)
            
            case "openai_compat":
                base = url or os.environ.get("OPENAI_COMPAT_BASE_URL", "").strip()
                if not base:
                    raise SystemExit(
                        "ABORT: no OpenAI-compatible endpoint. Set "
                        "OPENAI_COMPAT_BASE_URL (or pass base_url) plus "
                        "OPENAI_COMPAT_MODEL / OPENAI_COMPAT_API_KEY."
                    )
                resolved = want or cls.default_model("openai_compat")
                if not resolved:
                    raise SystemExit(
                        "ABORT: no model for the OpenAI-compatible endpoint. Set "
                        "OPENAI_COMPAT_MODEL or pass the model id explicitly."
                    )
                key = cls._resolve_key("openai_compat", api_key) or "not-needed"
                return cls._openai_compat(resolved, key, base)
            
            case _:
                raise ValueError(f"unknown chat backend {backend!r}")

    @staticmethod
    def is_cloud_backend(backend: str) -> bool:
        """Whether the backend id selects a stock cloud model.

        Local legs (qwen, bonsai, ...) read False everywhere.
        ..
        """
        return str(backend or "").lower().strip() in ChatModel.BACKENDS

    @staticmethod
    def chat_text(content) -> str:
        """LangChain content (str or list blocks) -> plain text.

        List blocks contribute their text parts in order.
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
