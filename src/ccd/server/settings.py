from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RemoteProviderSettings:
    provider: str
    base_url: str
    api_key: str
    chat_path: str = "/v1/chat/completions"
    timeout_s: float = 60.0
    max_retries: int = 3


_DOTENV_LOADED = False


def _try_load_dotenv() -> None:
    """Best-effort load of .env so users don't need to `source .env`.

    - Only loads once per process.
    - Does not override existing environment variables.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    def find_env(start: Path) -> Optional[Path]:
        cur = start
        for _ in range(12):
            candidate = cur / ".env"
            if candidate.exists() and candidate.is_file():
                return candidate
            if cur.parent == cur:
                break
            cur = cur.parent
        return None

    env_path = find_env(Path.cwd())
    if env_path is None:
        # Also try relative to this file (useful when CWD is not repo root)
        env_path = find_env(Path(__file__).resolve().parent)
    if env_path is None:
        return

    try:
        load_dotenv(dotenv_path=str(env_path), override=False)
    except Exception:
        # Never fail hard on dotenv parsing
        return


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


def load_remote_provider_settings(provider: str) -> RemoteProviderSettings:
    """Load remote provider config from env.

    Supported patterns:
    - provider == "openai_compatible": use CCD_REMOTE_BASE_URL / CCD_REMOTE_API_KEY / CCD_REMOTE_CHAT_PATH
    - otherwise: use CCD_{PROVIDER}_BASE_URL / CCD_{PROVIDER}_API_KEY / CCD_{PROVIDER}_CHAT_PATH
    """
    _try_load_dotenv()
    p = (provider or "").strip() or "openai_compatible"
    if p in {"openai_compatible", "remote"}:
        prefix = "CCD_REMOTE"
    else:
        prefix = f"CCD_{p.upper()}"

    base_url = _get_env(f"{prefix}_BASE_URL")
    api_key = _get_env(f"{prefix}_API_KEY")
    chat_path = _get_env(f"{prefix}_CHAT_PATH") or _get_env("CCD_REMOTE_CHAT_PATH") or "/v1/chat/completions"

    timeout_s_raw = _get_env(f"{prefix}_TIMEOUT_S") or _get_env("CCD_REMOTE_TIMEOUT_S") or "60"
    retries_raw = _get_env(f"{prefix}_MAX_RETRIES") or _get_env("CCD_REMOTE_MAX_RETRIES") or "3"
    try:
        timeout_s = float(timeout_s_raw)
    except Exception:
        timeout_s = 60.0
    try:
        max_retries = int(retries_raw)
    except Exception:
        max_retries = 3

    if not base_url:
        raise ValueError(
            f"Remote provider '{p}' missing env: {prefix}_BASE_URL. "
            f"Tip: create a .env file (ignored by git) or export env vars before starting the server."
        )
    if not api_key:
        raise ValueError(
            f"Remote provider '{p}' missing env: {prefix}_API_KEY. "
            f"Tip: create a .env file (ignored by git) or export env vars before starting the server."
        )
    return RemoteProviderSettings(
        provider=p,
        base_url=base_url,
        api_key=api_key,
        chat_path=chat_path,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
