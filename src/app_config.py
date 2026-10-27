"""
First-start bootstrap for ~/.config/voicert. Plus file metadata:
contract seeds skills/mcp.yaml/models.yaml/voicert.json once,
dependencies stdlib + yaml + src.paths, who calls it server get_agent.
"""

import json
import os

import yaml

from src.paths import VoicertPaths

__all__ = ["AppConfig"]

DEFAULT_MCP_YAML = """# MCP servers for the voice agent's extra tools.
# Managed at ~/.config/voicert/mcp.yaml (seeded on first start).
# Rules: no secrets here, ever. Keys travel via environment, and
# `env:` below only MERGES over the process environment.

tool_name_prefix: false

servers:
  voice-tools:
    enabled: true
    transport: stdio
    command: null
    args: ["-m", "src.agent.mcp.mcp"]
    cwd: "."
    env: {}

  # Remote connector shelf (the composer's "Get better answers from your
  # apps" row). They ship disabled: each needs its own OAuth grant, handled
  # by the MCP client, before you flip `enabled: true`.
  slack:
    enabled: false
    transport: streamable_http
    url: "https://mcp.slack.com/mcp"
  gmail:
    enabled: false
    transport: streamable_http
    url: "https://gmailmcp.googleapis.com/mcp/v1"
  calcom:
    enabled: false
    transport: streamable_http
    url: "https://mcp.cal.com/mcp"
  google-calendar:
    enabled: false
    transport: streamable_http
    url: "https://calendarmcp.googleapis.com/mcp/v1"
  google-drive:
    enabled: false
    transport: streamable_http
    url: "https://drivemcp.googleapis.com/mcp/v1"
  microsoft-teams:
    enabled: false
    transport: streamable_http
    url: "https://microsoft365.mcp.claude.com/mcp"
  dropbox:
    enabled: false
    transport: streamable_http
    url: "https://mcp.dropbox.com/mcp"
  stripe:
    enabled: false
    transport: streamable_http
    url: "https://mcp.stripe.com"
  asana:
    enabled: false
    transport: streamable_http
    url: "https://mcp.asana.com/v2/mcp"
  box:
    enabled: false
    transport: streamable_http
    url: "https://mcp.box.com"
  github:
    enabled: false
    transport: streamable_http
    url: "https://api.githubcopilot.com/mcp/"
  jira:
    enabled: false
    transport: streamable_http
    url: "https://mcp.atlassian.com/v2/mcp"
  linear:
    enabled: false
    transport: streamable_http
    url: "https://mcp.linear.app/mcp"
  salesforce:
    enabled: false
    transport: streamable_http
    url: "https://api.salesforce.com/platform/mcp/v1/platform/sobject-reads"
"""

DEFAULT_BACKENDS = (
    "groq",
    "gemini",
    "anthropic",
    "openai",
    "openrouter",
    "deepseek",
    "openai_compat",
)

DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3.5-flash",
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
    "openrouter": "deepseek/deepseek-chat",
    "deepseek": "deepseek-chat",
    "openai_compat": "",
}

DEFAULT_SETTINGS = {
    "permissions": {
        "read": "allow",
        "write": "ask",
        "edit": "ask",
        "exec": "ask",
        "network": "ask",
        "system": "deny",
    },
    "defaults": {
        "model_backend": "groq",
        "delegate_enabled": True,
        "worker_eager": False,
    },
    "tool_outputs": {
        "spill_chars": 8000,
    },
}


class AppConfig:
    """Seed and load the global config dir.

    ..
    """

    @staticmethod
    def ensure_global() -> str:
        """Create config dir + seed the four entries once.

        ..
        """
        cfg = VoicertPaths.config_dir()
        os.makedirs(cfg, exist_ok=True)
        os.makedirs(VoicertPaths.skills_dir(), exist_ok=True)
        AppConfig._ensure_mcp()
        AppConfig._ensure_models()
        AppConfig._ensure_settings()
        return cfg

    @staticmethod
    def _write_if_absent(path: str, text: str) -> bool:
        """Write text when path is missing, True when seeded.

        ..
        """
        if os.path.isfile(path):
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return True

    @staticmethod
    def _ensure_mcp() -> bool:
        """Seed mcp.yaml from the builtin default.

        ..
        """
        dest = VoicertPaths.mcp_path()
        if os.path.isfile(dest):
            return False
        AppConfig._write_if_absent(dest, DEFAULT_MCP_YAML)
        return True

    @staticmethod
    def _ensure_models() -> bool:
        """Seed models.yaml from builtin defaults.

        ..
        """
        dest = VoicertPaths.models_path()
        if os.path.isfile(dest):
            return False
        payload = {"default_backend": "groq", "models": dict(DEFAULT_MODELS)}
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=True)
            fh.write("\n")
        return True

    @staticmethod
    def _ensure_settings() -> bool:
        """Seed voicert.json with permissions and defaults.

        ..
        """
        dest = VoicertPaths.settings_path()
        if os.path.isfile(dest):
            return False
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(DEFAULT_SETTINGS, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return True

    @staticmethod
    def load_models() -> dict:
        """Global model map from models.yaml or builtins.

        ..
        """
        path = VoicertPaths.models_path()
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                try:
                    raw = yaml.safe_load(fh) or {}
                except yaml.YAMLError:
                    raw = {}
            if isinstance(raw, dict) and isinstance(raw.get("models"), dict):
                return {str(k): str(v or "") for k, v in raw["models"].items()}
            if isinstance(raw, dict) and raw and "models" not in raw:
                return {str(k): str(v or "") for k, v in raw.items()}
        return dict(DEFAULT_MODELS)

    @staticmethod
    def save_model(backend: str, model: str) -> str:
        """Save one backend default into models.yaml.

        ..
        """
        name = str(backend or "").lower().strip()
        if name not in DEFAULT_BACKENDS:
            raise ValueError(f"unknown chat backend {backend!r}")
        value = str(model or "").strip()
        if not value:
            raise ValueError("refusing to save an empty model id")
        data = AppConfig.load_models()
        data[name] = value
        dest = VoicertPaths.models_path()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        payload = {"default_backend": AppConfig.default_backend(), "models": data}
        with open(dest, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=True)
            fh.write("\n")
        return dest

    @staticmethod
    def default_backend() -> str:
        """Configured default backend id or groq.

        ..
        """
        path = VoicertPaths.models_path()
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                try:
                    raw = yaml.safe_load(fh) or {}
                except yaml.YAMLError:
                    raw = {}
            if isinstance(raw, dict):
                hit = str(raw.get("default_backend", "") or "").strip()
                if hit:
                    return hit
        return "groq"

    @staticmethod
    def load_settings() -> dict:
        """Voicert settings with builtin fallback, never raises.

        ..
        """
        try:
            with open(VoicertPaths.settings_path(), encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return dict(DEFAULT_SETTINGS)
        if not isinstance(raw, dict):
            return dict(DEFAULT_SETTINGS)
        return raw

    @staticmethod
    def spill_chars() -> int:
        """Tool-output spill threshold from voicert.json.

        ..
        """
        raw = AppConfig.load_settings()
        try:
            hit = int(((raw.get("tool_outputs") or {}).get("spill_chars")) or 8000)
        except (TypeError, ValueError, AttributeError):
            return 8000
        return max(1000, hit)
