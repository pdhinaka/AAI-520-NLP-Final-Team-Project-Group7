"""Single entry point for LLM calls (Anthropic Claude).

Everything in the project calls the model through this module, so
switching models is a one-line change to LLM_MODEL in .env.

    from finagent import llm
    text = llm.complete("Summarize ...", system="You are ...")
    data = llm.complete_json("Classify ... Return {\"label\": ...}")
    msg = llm.create_message(messages, tools=specs)  # raw, for tool use
"""

import json
import re
from functools import lru_cache
from typing import Any

import anthropic

from finagent import config
from finagent.cache import cached

# Running token totals, handy for reporting cost in the notebook.
USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    headers = {}
    if config.ANTHROPIC_WORKSPACE_ID:
        headers["anthropic-workspace-id"] = config.ANTHROPIC_WORKSPACE_ID
    return anthropic.Anthropic(
        api_key=config.require("ANTHROPIC_API_KEY"),
        default_headers=headers or None,
    )


def _system_blocks(system: str | None):
    if not system:
        return anthropic.NOT_GIVEN
    # Mark the system prompt cacheable; repeated calls with the same
    # prompt are billed at the cache-hit rate. Short prompts are simply
    # not cached, no error.
    return [
        {
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _track(msg) -> None:
    USAGE["calls"] += 1
    USAGE["input_tokens"] += msg.usage.input_tokens
    USAGE["output_tokens"] += msg.usage.output_tokens


def create_message(
    messages: list[dict],
    system: str | None = None,
    tools: list[dict] | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
):
    """Raw Messages API call. Use this for tool-use loops.

    Returns the anthropic Message object (check .stop_reason and
    .content for tool_use blocks).
    """
    kwargs: dict[str, Any] = {
        "model": model or config.LLM_MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
        "system": _system_blocks(system),
    }
    if tools:
        kwargs["tools"] = tools
    if temperature is not None:
        kwargs["temperature"] = temperature
    msg = _client().messages.create(**kwargs)
    _track(msg)
    return msg


def text_of(msg) -> str:
    """Join the text blocks of a Message."""
    return "".join(b.text for b in msg.content if b.type == "text")


def complete(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
) -> str:
    """Send one user prompt and return the reply text."""
    msg = create_message(
        [{"role": "user", "content": prompt}],
        system=system,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return text_of(msg)


def extract_json(text: str) -> Any:
    """Pull a JSON object or array out of a model reply.

    Handles ```json fences and stray prose around the JSON.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("No JSON found in reply")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[start:end + 1])


def complete_json(
    prompt: str,
    system: str | None = None,
    retries: int = 2,
    **kwargs,
) -> Any:
    """Like complete(), but parses and returns JSON.

    If the reply isn't valid JSON, the error is sent back to the model
    and it gets another try.
    """
    messages = [
        {
            "role": "user",
            "content": prompt + "\n\nRespond with valid JSON only.",
        }
    ]
    last_err = None
    for _ in range(retries + 1):
        msg = create_message(messages, system=system, **kwargs)
        reply = text_of(msg)
        try:
            return extract_json(reply)
        except (ValueError, json.JSONDecodeError) as err:
            last_err = err
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": f"That was not valid JSON ({err}). "
                    "Reply again with only the JSON.",
                },
            ]
    raise ValueError(f"Model did not return valid JSON: {last_err}")


def complete_json_cached(
    prompt: str,
    system: str | None = None,
    namespace: str = "llm",
    ttl_hours: float = 24 * 365,
    **kwargs,
) -> Any:
    """complete_json() with the reply saved to the disk cache.

    The cache key is the model, prompt, system prompt, and settings, so
    the same input returns the same output without another API call.
    With CACHE_MODE=offline this replays earlier replies, which keeps
    the final notebook run reproducible.
    """
    params = {
        "model": kwargs.get("model") or config.LLM_MODEL,
        "system": system,
        "prompt": prompt,
        **{k: v for k, v in kwargs.items() if k != "model"},
    }
    return cached(
        namespace,
        params,
        lambda: complete_json(prompt, system=system, **kwargs),
        ttl_hours=ttl_hours,
    )
