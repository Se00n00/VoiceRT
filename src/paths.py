"""
XDG locations for global config and local data. Plus file metadata:
contract pure path resolution with env overrides, dependencies stdlib
only, who calls it provider/mcp/delegate/store/server bootstrap.
"""

import os

__all__ = ["VoicertPaths"]


class VoicertPaths:
    """Global config dir and local data dir for voicert.

    ..
    """

    APP = "voicert"

    @staticmethod
    def config_dir() -> str:
        """Global config dir, XDG_CONFIG_HOME or ~/.config/voicert.

        ..
        """
        override = os.environ.get("VOICERT_CONFIG_DIR", "").strip()
        if override:
            return os.path.expanduser(override)
        base = os.environ.get("XDG_CONFIG_HOME", "").strip()
        if base:
            return os.path.join(os.path.expanduser(base), VoicertPaths.APP)
        return os.path.expanduser(os.path.join("~", ".config", VoicertPaths.APP))

    @staticmethod
    def data_dir() -> str:
        """Local data dir, XDG_DATA_HOME or ~/.local/share/voicert.

        ..
        """
        override = os.environ.get("VOICERT_DATA_DIR", "").strip()
        if override:
            return os.path.expanduser(override)
        base = os.environ.get("XDG_DATA_HOME", "").strip()
        if base:
            return os.path.join(os.path.expanduser(base), VoicertPaths.APP)
        return os.path.expanduser(os.path.join("~", ".local", "share", VoicertPaths.APP))

    @staticmethod
    def skills_dir() -> str:
        """Global skills directory path.

        ..
        """
        return os.path.join(VoicertPaths.config_dir(), "skills")

    @staticmethod
    def mcp_path() -> str:
        """Global mcp.yaml path.

        ..
        """
        return os.path.join(VoicertPaths.config_dir(), "mcp.yaml")

    @staticmethod
    def models_path() -> str:
        """Global models.yaml path.

        ..
        """
        return os.path.join(VoicertPaths.config_dir(), "models.yaml")

    @staticmethod
    def settings_path() -> str:
        """Global voicert.json path.

        ..
        """
        return os.path.join(VoicertPaths.config_dir(), "voicert.json")

    @staticmethod
    def log_path() -> str:
        """Local voicert.log path.

        ..
        """
        return os.path.join(VoicertPaths.data_dir(), "voicert.log")

    @staticmethod
    def db_path() -> str:
        """Local voicert.db path.

        ..
        """
        return os.path.join(VoicertPaths.data_dir(), "voicert.db")

    @staticmethod
    def tool_outputs_dir() -> str:
        """Local tool_outputs directory path.

        ..
        """
        return os.path.join(VoicertPaths.data_dir(), "tool_outputs")
