"""Plans research steps for a ticker."""

import logging

from finagent import llm

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


def _normalize_plan(raw) -> list[str] | None:
    """Normalize the model output to a clean list of strings."""
    if isinstance(raw, list):
        steps = [str(item).strip() for item in raw if str(item).strip()]
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


def make_plan(ticker: str, notes: list[str] | None = None) -> list[str]:
    """Create a research plan using the LLM or a safe fallback."""

    context = ""
    if notes:
        context = "\n\nPrior knowledge from previous runs:\n"
        context += "\n".join(f"- {n}" for n in notes)

    prompt = f"""You are an investment research planner. Create a concise
    research plan for {ticker}.
    The plan should have 5-7 steps covering:
        - Price and momentum
        - Company fundamentals
        - Recent news/sentiment
        - Macro context
        - Catalysts and risks
    Format as a JSON array of strings (steps).{context}"""

    system = (
        "You are an expert investment researcher. Plans should be "
        "specific and actionable."
    )

    try:
        result = llm.complete_json(prompt, system=system)
    except Exception as err:  # noqa: BLE001 - logged, then fall back
        log.warning("Planner LLM call failed, using DEFAULT_PLAN: %r", err)
        return DEFAULT_PLAN[:]

    steps = _normalize_plan(result)
    if steps:
        return steps[:7]
    log.warning("Planner reply had no usable steps, using DEFAULT_PLAN")
    return DEFAULT_PLAN[:]


def plan(ticker: str, notes: list[str] | None = None) -> list[str]:
    """Backward-compatible alias for older notebook calls."""
    return make_plan(ticker, notes)
