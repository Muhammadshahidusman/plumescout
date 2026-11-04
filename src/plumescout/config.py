"""Load YAML configs and `.env` secrets.

Paths in YAML are resolved relative to the repo root, so code never hard-codes paths.
Secrets come only from environment variables (populated from `.env` if present).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = REPO_ROOT / "configs"

SECRET_KEYS: tuple[str, ...] = (
    "EARTHDATA_USERNAME",
    "EARTHDATA_PASSWORD",
    "FIRMS_MAP_KEY",
    "HF_TOKEN",
    "GEE_PROJECT",
)


def load_env(env_path: Path | None = None) -> bool:
    """Load `.env` into `os.environ` without overriding existing variables.

    Returns True if a file was found and loaded.
    """
    path = env_path or REPO_ROOT / ".env"
    if not path.is_file():
        logger.debug("No .env file at %s", path)
        return False
    load_dotenv(path, override=False)
    return True


def get_secret(key: str, required: bool = True) -> str | None:
    """Return a secret from the environment, loading `.env` first.

    Raises KeyError with a setup hint if `required` and the value is missing or empty.
    """
    load_env()
    value = os.environ.get(key) or None
    if value is None and required:
        raise KeyError(f"{key} is not set. Add it to .env (see .env.example).")
    return value


def missing_secrets() -> list[str]:
    """List the secret keys from SECRET_KEYS that are unset or empty."""
    load_env()
    return [k for k in SECRET_KEYS if not os.environ.get(k)]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict. Relative names resolve against `configs/`."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIG_DIR / p
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top level of {p} must be a mapping, got {type(data).__name__}")
    return data


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path against the repo root if it is relative."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
