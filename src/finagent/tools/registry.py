"""Registry of tools the agent can call.

Workstream B uses this to give the LLM a menu of tools:

    from finagent.tools import registry
    specs = registry.tool_specs()          # pass as tools= to the LLM
    result = registry.call("get_news", {"ticker": "AAPL"})

The docstring of each tool is its description, so keep them accurate.
"""

import inspect
import json
import types
import typing

from finagent.tools import earnings, filings, macro, market_data, news

TOOLS = {
    fn.__name__: fn
    for fn in (
        market_data.get_company_info,
        market_data.get_price_summary,
        market_data.get_price_history,
        market_data.get_financials,
        news.get_news,
        macro.get_macro_snapshot,
        macro.get_series,
        filings.get_recent_filings,
        filings.get_filing_text,
        earnings.get_earnings,
    )
}

_JSON_TYPES = {str: "string", int: "integer", float: "number",
               bool: "boolean"}


def _json_type(hint) -> str:
    # Unwrap "str | None" and Optional[str]
    if isinstance(hint, types.UnionType) or \
            typing.get_origin(hint) is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        hint = args[0] if args else str
    return _JSON_TYPES.get(hint, "string")


def _schema(fn) -> dict:
    hints = typing.get_type_hints(fn)
    props, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        props[name] = {"type": _json_type(hints.get(name, str))}
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            props[name]["default"] = param.default
    return {"type": "object", "properties": props, "required": required}


def tool_specs(names: list[str] | None = None) -> list[dict]:
    """Tool definitions in the Anthropic Messages API format."""
    names = names or list(TOOLS)
    return [
        {
            "name": name,
            "description": inspect.getdoc(TOOLS[name]),
            "input_schema": _schema(TOOLS[name]),
        }
        for name in names
    ]


def describe() -> str:
    """One line per tool, handy for a planning prompt."""
    lines = []
    for name, fn in TOOLS.items():
        first = (inspect.getdoc(fn) or "").split("\n")[0]
        lines.append(f"- {name}: {first}")
    return "\n".join(lines)


def call(name: str, args: dict | None = None) -> dict | list:
    """Run a tool by name. Errors come back as {"error": ...} so the
    agent can see what went wrong and try something else."""
    if name not in TOOLS:
        return {"error": f"Unknown tool {name}"}
    try:
        return TOOLS[name](**(args or {}))
    except Exception as err:  # noqa: BLE001 - reported to the agent
        return {"error": f"{type(err).__name__}: {err}"}


def call_json(name: str, args: dict | None = None,
              max_chars: int = 20000) -> str:
    """call(), serialized and truncated for a tool_result message."""
    text = json.dumps(call(name, args), default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + ' ..."[truncated]"'
    return text
