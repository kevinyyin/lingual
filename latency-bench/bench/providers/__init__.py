"""Provider registry (lazy imports so a missing optional dependency only affects its provider)."""

from __future__ import annotations

import importlib
from typing import Any

from bench.providers.base import Provider

_REGISTRY = {
    "mock": ("bench.providers.mock", "MockProvider"),
    "openai": ("bench.providers.openai_translate", "OpenAITranslateProvider"),
    "gemini": ("bench.providers.gemini_translate", "GeminiTranslateProvider"),
    "cascade": ("bench.providers.cascade", "CascadeProvider"),
    "hibiki": ("bench.providers.hibiki_local", "HibikiLocalProvider"),
}

PROVIDER_NAMES = list(_REGISTRY)


def provider_class(name: str) -> type[Provider]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown provider '{name}' (known: {', '.join(_REGISTRY)})")
    mod, cls = _REGISTRY[name]
    return getattr(importlib.import_module(mod), cls)


def make_provider(name: str, cfg: dict[str, Any]) -> Provider:
    return provider_class(name)(cfg)
