"""Config resolution and the domain list, standard library only.

Kept free of Qdrant and model imports so lightweight entry points (the
``kipp-capture`` hook, issue #71) can read config in milliseconds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DOMAINS = ["shared", "dev", "personal"]


def _xdg_config_home() -> Path:
    """Base config dir per the XDG spec.

    ``$XDG_CONFIG_HOME`` when set to an absolute path, else ``~/.config``. The
    spec says a relative value is invalid and must be ignored, so we fall back.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg and os.path.isabs(xdg):
        return Path(xdg)
    return Path.home() / ".config"


# Explicit config path env vars, highest precedence first. ENTITY_MEMORY_CONFIG
# is the pre-rename name, still honoured.
CONFIG_ENV_VARS = ("KIPP_CONFIG", "ENTITY_MEMORY_CONFIG")


def _config_search_paths() -> list[Path]:
    """Default config file locations, highest precedence first.

    Consulted only when no config env var is set. The pre-rename
    ``entity-memory`` directory and the legacy ``~/.openclaw/memory.json`` path
    are kept after the KIPP location so existing installs keep resolving
    unchanged.
    """
    return [
        _xdg_config_home() / "kipp" / "config.json",
        _xdg_config_home() / "entity-memory" / "config.json",
        Path.home() / ".openclaw" / "memory.json",
    ]


def load_config() -> dict:
    """Resolve Qdrant config from env vars and standard file locations.

    Precedence, highest first:

    1. ``QDRANT_URL`` env var — point at any Qdrant with no config file at all.
    2. ``KIPP_CONFIG`` (or the pre-rename ``ENTITY_MEMORY_CONFIG``) env var —
       explicit path to a config JSON. When
       set it is the *only* file consulted; if it's missing we raise rather than
       silently using a different Qdrant — unless ``QDRANT_URL`` is also set,
       which is higher precedence and supplies the url without needing a file.
    3. ``~/.config/kipp/config.json`` — default config location.
    4. ``~/.config/entity-memory/config.json`` — pre-rename location.
    5. ``~/.openclaw/memory.json`` — legacy OpenClaw path, kept for back-compat.
    6. Built-in defaults (localhost Qdrant).

    The chosen config file's top-level keys overlay the defaults; ``QDRANT_URL``
    then overrides the resolved url, so a deployment can point at a different
    Qdrant without editing any file.
    """
    config: dict = {
        "qdrant": {"url": "http://127.0.0.1:6333", "api_key_env": "QDRANT_API_KEY"},
    }
    url_override = os.environ.get("QDRANT_URL")
    explicit_var = next(
        (v for v in CONFIG_ENV_VARS if os.environ.get(v)), None,
    )
    explicit = os.environ.get(explicit_var) if explicit_var else None
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            with open(explicit_path) as f:
                config.update(json.load(f))
        elif not url_override:
            # Operator pointed at a config we can't read and gave no QDRANT_URL
            # to fall back on — abort rather than silently using another Qdrant.
            raise FileNotFoundError(
                f"{explicit_var} points at {explicit!r}, which does not "
                "exist; set QDRANT_URL or fix the path."
            )
        # else: QDRANT_URL (higher precedence) supplies the url below, so a
        # missing explicit file is non-fatal.
    else:
        for path in _config_search_paths():
            if path.exists():
                with open(path) as f:
                    config.update(json.load(f))
                break
    if url_override:
        config.setdefault("qdrant", {})["url"] = url_override
    return config
