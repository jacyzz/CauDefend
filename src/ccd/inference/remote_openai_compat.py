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
            "Authorization": f"Bearer {self._settings.api_key}",
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
        # Some providers support seed; safe to include (ignored otherwise)
        body["seed"] = int(gen_cfg.seed)

        url = _join_url(self._settings.base_url, self._settings.chat_path)
        secrets = [self._settings.api_key]

        last_err: Optional[str] = None
        for attempt in range(int(self._settings.max_retries) + 1):
            try:
                resp = self._client.post(url, json=body)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    if attempt < int(self._settings.max_retries):
                        time.sleep(min(8.0, 0.8 * (2 ** attempt)))
                        continue
                resp.raise_for_status()
                data = resp.json()
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
                last_err = str(e)
                if attempt < int(self._settings.max_retries):
                    time.sleep(min(8.0, 0.8 * (2 ** attempt)))
                    continue
                break

        safe = _safe_error_text(last_err or "remote request failed", secrets)
        raise RuntimeError(safe)
