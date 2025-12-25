from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ccd.inference.engine import (
    GenerationConfig,
    PromptConfig,
    build_prompt,
    extract_response,
    load_system_prompt,
)
from ccd.server.settings import RemoteProviderSettings


@dataclass
class RemoteChatResult:
    candidates: List[str]
    decoded: List[str]
    usage: Optional[Dict[str, Any]] = None


def _join_url(base_url: str, path: str) -> str:
    b = (base_url or "").strip().rstrip("/")
    p = (path or "").strip()
    if not b:
        return p

    # Normalize path to no leading slash for easier de-duplication.
    p0 = p.lstrip("/")

    # Common OpenAI-compatible pattern: base ends with /v1, path starts with v1/...
    # Avoid producing .../v1/v1/...
    if b.endswith("/v1") and (p0 == "v1" or p0.startswith("v1/")):
        p0 = p0[3:]  # drop leading 'v1'
        p0 = p0.lstrip("/")

    if not p0:
        return b
    return b + "/" + p0


def _safe_error_text(text: str, secrets: List[str]) -> str:
    out = text
    for s in secrets:
        if s:
            out = out.replace(s, "***")
    return out


def _strip_bearer(value: str) -> str:
    v = (value or "").strip()
    if v.lower().startswith("bearer "):
        return v[len("bearer ") :].strip()
    return v


def _format_authorization(api_key: str, *, mode: str) -> str:
    """Format Authorization header.

    PoloAPI docs show both patterns in different places:
    - Batch example: Authorization: Bearer <API_KEY>
    - OpenAPI spec header example: Authorization: sk-

    To maximize compatibility we:
    - default to bearer (works with OpenAI and most proxies)
    - optionally fallback to raw key when upstream returns 401/403
    """
    token = _strip_bearer(api_key)
    if mode == "raw":
        return token
    # bearer (default)
    return f"Bearer {token}" if token else ""


class OpenAICompatClient:
    """Minimal OpenAI-compatible Chat Completions client.

    - Key comes from env via RemoteProviderSettings
    - Retries on 429/5xx
    - Never returns error strings containing the API key
    """

    def __init__(self, settings: RemoteProviderSettings):
        self._settings = settings
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> "OpenAICompatClient":
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=self._settings.timeout_s, headers=headers)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None

    def generate(self, *, model: str, input_text: str, prompt_cfg: PromptConfig, gen_cfg: GenerationConfig) -> RemoteChatResult:
        if self._client is None:
            raise RuntimeError("OpenAICompatClient must be used as a context manager")

        system_prompt = load_system_prompt(prompt_cfg)
        # For OpenAI-compatible chat, send system prompt in a dedicated system-role message
        # and keep the user message free of embedded system headers.
        user_prompt = build_prompt(input_text, "")

        messages: List[Dict[str, str]] = []
        if (system_prompt or "").strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt})

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(gen_cfg.temperature if gen_cfg.do_sample else 0.0),
            "top_p": float(gen_cfg.top_p),
            "max_tokens": int(gen_cfg.max_new_tokens),
            "n": int(gen_cfg.num_return_sequences or 1),
            "stream": False,
        }
        # Note: do NOT send non-standard fields (like `seed`) by default.
        # Some proxies enforce strict schemas and will reject unknown fields.

        url = _join_url(self._settings.base_url, self._settings.chat_path)
        secrets = [self._settings.api_key]

        def _do_post(auth_mode: str) -> Dict[str, Any]:
            assert self._client is not None
            headers = {"Authorization": _format_authorization(self._settings.api_key, mode=auth_mode)}
            last_err_local: Optional[str] = None
            for attempt in range(int(self._settings.max_retries) + 1):
                resp = self._client.post(url, json=body, headers=headers)
                if 200 <= resp.status_code < 300:
                    return resp.json()

                # Capture upstream error body (sanitized later)
                last_err_local = f"HTTP {resp.status_code}: {resp.text[:1000]}"

                if resp.status_code in (429, 500, 502, 503, 504) and attempt < int(self._settings.max_retries):
                    time.sleep(min(8.0, 0.8 * (2 ** attempt)))
                    continue

                raise httpx.HTTPStatusError(
                    message=f"Upstream returned HTTP {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            raise RuntimeError(last_err_local or "remote request failed")

        last_err: Optional[str] = None
        try:
            # Default: Bearer
            data = _do_post("bearer")
        except httpx.HTTPStatusError as e:
            # If auth fails, try raw key once (covers proxies that expect Authorization: sk-xxx)
            try:
                status = int(getattr(e.response, "status_code", 0) or 0)
            except Exception:
                status = 0
            if status in (401, 403):
                try:
                    data = _do_post("raw")
                except Exception as e2:
                    last_err = str(e2)
                    data = None  # type: ignore
            else:
                try:
                    r = e.response
                    last_err = f"HTTP {r.status_code}: {r.text[:1000]}"
                except Exception:
                    last_err = str(e)
                data = None  # type: ignore
        except Exception as e:
            last_err = str(e)
            data = None  # type: ignore

        if data is None:
            safe = _safe_error_text(last_err or "remote request failed", secrets)
            raise RuntimeError(safe)

        try:
                choices = data.get("choices") or []
                decoded: List[str] = []
                candidates: List[str] = []
                for ch in choices:
                    msg = ch.get("message") if isinstance(ch, dict) else None
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        txt = msg.get("content")
                    elif isinstance(ch, dict) and isinstance(ch.get("text"), str):
                        txt = ch.get("text")
                    else:
                        txt = ""
                    decoded.append(txt)
                    candidates.append(extract_response(txt))
                if not candidates:
                    # Some proxies might return a single string field
                    txt = data.get("text") if isinstance(data, dict) else None
                    if isinstance(txt, str) and txt:
                        decoded = [txt]
                        candidates = [extract_response(txt)]

                usage = data.get("usage") if isinstance(data, dict) else None
                return RemoteChatResult(candidates=candidates, decoded=decoded, usage=usage)
        except Exception as e:
            safe = _safe_error_text(str(e), secrets)
            raise RuntimeError(safe)
