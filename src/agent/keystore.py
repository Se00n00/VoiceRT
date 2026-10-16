"""
OS-keychain secret store for LLM provider keys.

contract, dependencies, who calls it: get/set/delete/has secrets by account;
depends on keyring (Keychain / Credential Manager / Secret Service); used by
provider key resolution and the /keys bridge routes. Process env always
wins (headless/CI override); the gitignored .env stays as a warned legacy
fallback. Writes fail loudly when no OS store is reachable.
"""

import os
import re

SERVICE = "VoiceRT"
"""Keychain service namespace; every account below lives under it."""

_ACCOUNT_RE = re.compile(r"[a-z0-9_]{1,64}")
"""Allowed account shape (provider ids, never paths or sentences)."""

_available: bool | None = None
"""Cached OS-store probe (None = unprobed)."""


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


def _check_account(account: str) -> str:
    """Validated account id, else raises ValueError (never a keyring call).

    ..
    """
    name = str(account or "").strip().lower()
    if not _ACCOUNT_RE.fullmatch(name):
        raise ValueError(f"unknown secret account {account!r}")
    return name


def store_available() -> bool:
    """Whether an OS keychain backend is reachable (cached probe).

    A missing keyring install or its fail-backend both read as False;
    callers fail loudly on writes and fall back on reads.
    ..
    """
    global _available
    if _available is not None:
        return _available
    try:
        import keyring
        import keyring.backends.fail

        backend = keyring.get_keyring()
        _available = not isinstance(backend, keyring.backends.fail.Keyring)
    except Exception:
        _available = False
    return _available


def store_name() -> str:
    """Human label of the active keychain backend, or "" when none.

    ..
    """
    try:
        import keyring

        backend = keyring.get_keyring()
        return type(backend).__name__
    except Exception:
        return ""


def _unavailable_help() -> str:
    """Setup guidance used when no OS store is reachable.

    ..
    """
    return (
        "no OS keychain reachable (keyring backend unavailable). On Linux "
        "enable Secret Service (gnome-keyring-daemon --start, or kwallet "
        "with Secret Service enabled) and retry; until then set the "
        "provider KEY env var or the gitignored repo .env."
    )


_warned_dotenv: set = set()
"""Accounts already warned about legacy .env reads (warn once each)."""


def get_secret(account: str, *env_names: str) -> str:
    """Env -> OS keychain -> legacy .env for one account. Never raises.

    Empty names are skipped; a legacy .env hit warns once per account
    (it keeps working, but the key should move into Settings -> Keys).
    ..
    """
    name = str(account or "").strip().lower()
    for env in env_names:
        if not env:
            continue
        hit = os.environ.get(env, "").strip()
        if hit:
            return hit
    if store_available():
        try:
            import keyring

            hit = keyring.get_password(SERVICE, name) or ""
            if str(hit).strip():
                return str(hit).strip()
        except Exception:
            pass
    _load_dotenv()
    for env in env_names:
        if not env:
            continue
        hit = os.environ.get(env, "").strip()
        if hit:
            if name not in _warned_dotenv:
                _warned_dotenv.add(name)
                try:
                    print(f"[keys] {name}: read from legacy .env — "
                          f"move it into Settings -> Keys", flush=True)
                except Exception:
                    pass
            return hit
    return ""


def has_secret(account: str) -> bool:
    """Whether the OS keychain holds this account. Never raises.

    Env/.env are intentionally NOT consulted: this answers "is a key
    stored", which is what the UI status dots show.
    ..
    """
    name = _check_account(account)
    if not store_available():
        return False
    try:
        import keyring

        return bool((keyring.get_password(SERVICE, name) or "").strip())
    except Exception:
        return False


def set_secret(account: str, secret: str) -> None:
    """Store a secret in the OS keychain. Raises when unusable.

    Raises RuntimeError when no OS store is reachable (fail loudly, never
    silent plaintext) and ValueError for bad accounts or empty secrets.
    ..
    """
    name = _check_account(account)
    value = str(secret or "").strip()
    if not value:
        raise ValueError("refusing to store an empty secret")
    if not store_available():
        raise RuntimeError(_unavailable_help())
    import keyring

    keyring.set_password(SERVICE, name, value)


def delete_secret(account: str) -> bool:
    """Remove a stored secret. False when absent. Never raises.

    ..
    """
    try:
        name = _check_account(account)
    except ValueError:
        return False
    if not store_available():
        return False
    try:
        import keyring

        if not (keyring.get_password(SERVICE, name) or "").strip():
            return False
        keyring.delete_password(SERVICE, name)
        return True
    except Exception:
        return False
