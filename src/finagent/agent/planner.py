"""Plans research steps for a ticker."""

import logging

from finagent import llm
from finagent.tools import registry

log = logging.getLogger(__name__)

DEFAULT_PLAN = [
    "Check the latest price trend and 52-week context",
    "Review the company's fundamentals and valuation",
    "Assess recent earnings and operating momentum",
    "Read recent news and sentiment for catalysts and risks",
    "Review macro conditions affecting the stock",
    "Evaluate the key risks that could invalidate the thesis",
    "Draft a clear investment thesis and recommendation",
]

MAX_STEPS = 7

PLANNER_SYSTEM = (
    "You are an expert investment researcher. Plans should be specific "
    "and actionable, and every research step should name the tools "
    "that will answer it."
)


def _normalize_plan(raw) -> list[str] | None:
    """Normalize the model output to a clean list of strings."""
    if isinstance(raw, list):
        steps = []
        for item in raw:
            if isinstance(item, dict):
                item = item.get("step") or item.get("description") or ""
            if str(item).strip():
                steps.append(str(item).strip())
        return steps or None

    if isinstance(raw, dict):
        for key in ("steps", "plan", "research_plan"):
            if key in raw:
                item = _normalize_plan(raw[key])
                if item:
                    return item

    if isinstance(raw, str):
        cleaned = [
            piece.strip() for piece in raw.splitlines() if piece.strip()
        ]
        if cleaned:
            return cleaned

    return None


def _step_tools(raw) -> list[list[str]]:
    """Tools named for each step, keeping only real tool names."""
    steps = raw.get("steps") if isinstance(raw, dict) else None
    if not isinstance(steps, list):
        return []
    out = []
    for item in steps:
        tools = item.get("tools", []) if isinstance(item, dict) else []
        if isinstance(tools, str):
            tools = [tools]
        out.append([t for t in tools if t in registry.TOOLS])
    return out


def _notes_text(notes) -> str:
    """Prior notes grouped by category. Accepts strings or memory
    records ({"note", "category"})."""
    groups: dict[str, list[str]] = {}
    for n in notes or []:
        if isinstance(n, dict):
            groups.setdefault(n.get("category", "finding"), []).append(
                n["note"])
        else:
            groups.setdefault("finding", []).append(str(n))
    titles = {
        "finding": "Findings from earlier runs",
        "lesson": "Lessons: gaps flagged in earlier reports",
        "tool": "Tool lessons",
    }
    parts = []
    for cat in ("finding", "lesson", "tool"):
        if groups.get(cat):
            lines = "\n".join(f"- {n}" for n in groups[cat])
            parts.append(f"{titles[cat]}:\n{lines}")
    return "\n\n".join(parts)


def make_plan_detailed(ticker: str, notes=None) -> dict:
    """Plan research steps with the LLM and say how memory shaped it.

    Returns {steps, step_tools, memory_used, source, error}. source is
    "llm" or "fallback" (DEFAULT_PLAN, used if the call fails or the
    reply has no usable steps; the reason is in error and logged).
    """
    notes_text = _notes_text(notes)
    memory_part = (
        "\n\nNotes from earlier runs on this stock:\n"
        f"{notes_text}\n\n"
        "Use these notes. Address every gap flagged in a lesson (for "
        "example by fetching the missing data earlier), avoid tools "
        "that failed before, and check whether earlier findings still "
        "hold. In memory_used, list each note that changed the plan "
        "and how it changed it."
        if notes_text else
        "\n\nThere are no notes from earlier runs; memory_used is []."
    )
    prompt = (
        f"Create a research plan for {ticker}: 5-7 steps that together "
        "cover price and momentum, company fundamentals and valuation, "
        "recent earnings, news and sentiment, the macro backdrop, and "
        "the catalysts and risks behind a Buy, Hold, or Sell call.\n\n"
        f"Available tools:\n{registry.describe()}"
        f"{memory_part}\n\n"
        "Return JSON: {\"steps\": [{\"step\": \"<what to find out>\", "
        "\"tools\": [\"<tool name>\", ...]}, ...], "
        "\"memory_used\": [{\"note\": \"<note>\", \"effect\": "
        "\"<how it changed the plan>\"}, ...]}"
    )

    try:
        result = llm.complete_json_cached(prompt, system=PLANNER_SYSTEM,
                                          max_tokens=1024)
    except Exception as err:  # noqa: BLE001 - logged, then fall back
        log.warning("Planner LLM call failed, using DEFAULT_PLAN: %r", err)
        return _fallback(f"{type(err).__name__}: {err}")

    steps = _normalize_plan(result)
    if not steps:
        log.warning("Planner reply had no usable steps, using DEFAULT_PLAN")
        return _fallback("reply had no usable steps")

    tools = _step_tools(result)[:MAX_STEPS]
    used = result.get("memory_used") if isinstance(result, dict) else []
    return {
        "steps": steps[:MAX_STEPS],
        "step_tools": tools,
        "memory_used": [u for u in used or [] if u] if notes_text else [],
        "source": "llm",
        "error": None,
    }


def _fallback(reason: str) -> dict:
    return {"steps": DEFAULT_PLAN[:], "step_tools": [],
            "memory_used": [], "source": "fallback", "error": reason}


def make_plan(ticker: str, notes: list[str] | None = None) -> list[str]:
    """Create a research plan using the LLM or a safe fallback."""
    return make_plan_detailed(ticker, notes)["steps"]


def plan(ticker: str, notes: list[str] | None = None) -> list[str]:
    """Backward-compatible alias for older notebook calls."""
    return make_plan(ticker, notes)
