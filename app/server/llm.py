"""One structured-JSON call to a fast Claude model, shared by compose.py and gloss.py."""

import json
from typing import Any

import anthropic

MODEL = "claude-haiku-5-5"
TIMEOUT_S = 4.0


class LLMError(Exception):
    pass


async def ask_json(client, system: str | list, user: str, schema: dict | None, max_tokens: int = 400) -> dict[str, Any]:
    """schema=None skips constrained decoding (faster; the prompt must ask for JSON). Raises LLMError on timeout, API error, refusal or unparseable output; callers retry once then fall back."""
    try:
        r = await client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "disabled"},  # latency budget is 2 s end to end
            output_config={"effort": "low", **({"format": {"type": "json_schema", "schema": schema}} if schema else {})},
            timeout=TIMEOUT_S,
        )
    except (anthropic.APIError, TypeError) as e:  # API/timeout/connection errors; TypeError = no credentials configured
        raise LLMError(f"{type(e).__name__}: {e}") from e
    if r.stop_reason == "refusal":
        raise LLMError("refusal")
    text = "".join(b.text for b in r.content if b.type == "text")
    try:
        out = parse_object(text)
    except json.JSONDecodeError as e:  # a cut-off reply (stop max_tokens) is never worth repairing: the plan is partial
        raise LLMError(f"bad JSON (stop {r.stop_reason}): {text[:80]!r}") from e
    if not isinstance(out, dict):
        raise LLMError("JSON is not an object")
    return out


def parse_object(text: str) -> Any:
    """The first JSON value in the reply: schema-free replies sometimes wrap it in a ```json fence or follow it with a
    remark, which json.loads rejects although the object itself is whole."""
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("no JSON object", text, 0)
    return json.JSONDecoder().raw_decode(text, start)[0]
