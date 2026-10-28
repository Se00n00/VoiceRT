"""
Single local setup owner: XDG dirs, config seeds and session stores.

contract idempotent ensure()/check() for every file the app needs,
dependencies stdlib + yaml + src.paths + src.app_config + src.agent.store,
who calls it server get_agent, provider seed and `python -m src.setup`.
"""

import argparse
import json
import os

import yaml

from src.agent.store import VoicertStore
from src.app_config import DEFAULT_MCP_YAML, DEFAULT_MODELS, DEFAULT_SETTINGS
from src.paths import VoicertPaths

__all__ = ["VoicertSetup", "main"]


class VoicertSetup:
    """Create and verify the complete local config and data layout."""

    @staticmethod
    def ensure() -> dict:
        """Create every config file, directory and store. Idempotent.

        ..
        """
        created: list[str] = []
        cfg = VoicertSetup._dir(VoicertPaths.config_dir(), created)
        VoicertSetup._dir(VoicertPaths.skills_dir(), created)
        VoicertSetup._file(VoicertPaths.mcp_path(), DEFAULT_MCP_YAML, created)
        models = yaml.safe_dump(
            {"default_backend": "groq", "models": dict(DEFAULT_MODELS)},
            sort_keys=True,
        ) + "\n"
        VoicertSetup._file(VoicertPaths.models_path(), models, created)
        settings = json.dumps(
            DEFAULT_SETTINGS, indent=2, sort_keys=True
        ) + "\n"
        VoicertSetup._file(VoicertPaths.settings_path(), settings, created)
        data = VoicertSetup._dir(VoicertPaths.data_dir(), created)
        VoicertSetup._dir(VoicertPaths.sessions_dir(), created)
        VoicertSetup._dir(VoicertPaths.tg_dir(), created)
        VoicertSetup._dir(VoicertPaths.memory_dir(), created)
        VoicertSetup._dir(VoicertPaths.tool_outputs_dir(), created)
        if not os.path.isfile(VoicertPaths.db_path()):
            created.append(VoicertPaths.db_path())
        VoicertStore().ensure()
        return {"config_dir": cfg, "data_dir": data, "created": created}

    @staticmethod
    def check() -> list[str]:
        """Paths still missing from the local setup (empty when ready).

        ..
        """
        wanted = [
            VoicertPaths.config_dir(),
            VoicertPaths.skills_dir(),
            VoicertPaths.mcp_path(),
            VoicertPaths.models_path(),
            VoicertPaths.settings_path(),
            VoicertPaths.data_dir(),
            VoicertPaths.sessions_dir(),
            VoicertPaths.tg_dir(),
            VoicertPaths.memory_dir(),
            VoicertPaths.tool_outputs_dir(),
            VoicertPaths.db_path(),
        ]
        return [path for path in wanted if not os.path.exists(path)]

    @staticmethod
    def _dir(path: str, created: list[str]) -> str:
        """Create one directory when absent and record it.

        ..
        """
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            created.append(path)
        return path

    @staticmethod
    def _file(path: str, text: str, created: list[str]) -> bool:
        """Write one seed file when absent and record it.

        ..
        """
        if os.path.isfile(path):
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        created.append(path)
        return True


def main(argv: list[str] | None = None) -> int:
    """Run setup from the command line, or only check readiness.

    ..
    """
    ap = argparse.ArgumentParser(description="create local voicert state")
    ap.add_argument("--check", action="store_true",
                    help="list missing paths and exit non-zero")
    args = ap.parse_args(argv)
    if args.check:
        missing = VoicertSetup.check()
        if missing:
            print("missing:")
            for path in missing:
                print(f"  {path}")
            return 1
        print("setup ok")
        return 0
    result = VoicertSetup.ensure()
    print(f"config: {result['config_dir']}")
    print(f"data:   {result['data_dir']}")
    if result["created"]:
        for path in result["created"]:
            print(f"created: {path}")
    else:
        print("already set up")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
