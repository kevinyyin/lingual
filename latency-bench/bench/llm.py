"""Minimal streaming LLM client for the cascade translator, the EVS aligner and the judge.

provider: openai     -> POST {base_url}/chat/completions (any OpenAI-compatible host:
                        OpenAI, Groq, Cerebras, Together, a local server, ...)
provider: anthropic  -> POST {base_url}/messages (Anthropic Messages API, SSE)
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx

from bench.config import LLMCfg


class LLMClient:
    def __init__(self, cfg: LLMCfg, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.key = os.environ.get(cfg.api_key_env, "")
        self._own = client is None
        self.http = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    @property
    def available(self) -> bool:
        return bool(self.key)

    def _base(self) -> str:
        if self.cfg.provider == "anthropic" and "openai.com" in self.cfg.base_url:
            return "https://api.anthropic.com/v1"
        return self.cfg.base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        if self.cfg.provider == "anthropic":
            return {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

    async def warm(self) -> None:
        """Open the TCP+TLS connection ahead of time (a real product keeps it warm)."""
        try:
            await self.http.get(f"{self._base()}/models", headers=self._headers())
        except Exception:
            pass

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        if self.cfg.provider == "anthropic":
            body = {"model": self.cfg.model, "max_tokens": self.cfg.max_tokens, "system": system, "stream": True,
                    "temperature": self.cfg.temperature, "messages": [{"role": "user", "content": user}],
                    **self.cfg.extra_body}
            async with self.http.stream("POST", f"{self._base()}/messages", headers=self._headers(), json=body) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"LLM HTTP {r.status_code}: {(await r.aread())[:300]!r}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    ev = json.loads(line[5:])
                    if ev.get("type") == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                        yield ev["delta"]["text"]
                    elif ev.get("type") == "error":
                        raise RuntimeError(f"LLM error: {ev.get('error')}")
            return
        tok_field = "max_completion_tokens" if "api.openai.com" in self._base() else "max_tokens"
        body = {"model": self.cfg.model, "stream": True, "temperature": self.cfg.temperature,
                tok_field: self.cfg.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                **self.cfg.extra_body}
        async with self.http.stream("POST", f"{self._base()}/chat/completions", headers=self._headers(), json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"LLM HTTP {r.status_code}: {(await r.aread())[:300]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                for ch in ev.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        yield piece

    async def complete(self, system: str, user: str) -> str:
        return "".join([t async for t in self.stream(system, user)])

    async def aclose(self) -> None:
        if self._own:
            await self.http.aclose()
