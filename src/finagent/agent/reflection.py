"""Self-reflection on the agent's own output."""

from finagent import llm

PLACEHOLDER_TEXT = (
    "research completed",
    "based on gathered data",
    "hold pending full analysis",
    "no clear catalyst",
)


def _heuristic_gaps(report: dict) -> list[str]:
    """Add concrete checklist-style gaps even if the LLM is weak."""
    gaps: list[str] = []
    text = "\n".join(
        str(report.get(key, ""))
        for key in (
            "summary", "investment_thesis", "recommendation", "upside",
            "risks", "catalysts", "outlook",
        )
    ).lower()

    if any(token in text for token in PLACEHOLDER_TEXT):
        gaps.append(
            "Missing a clear investment thesis and explicit recommendation"
        )
    if not report.get("catalysts"):
        gaps.append("No concrete catalyst path was identified")
    if not report.get("recommendation"):
        gaps.append("Recommendation is missing or not explicit")
    if not report.get("investment_thesis"):
        gaps.append("Investment thesis is not written out")
    generic = ["Based on gathered data"]
    if not report.get("upside") or report.get("upside") == generic:
        gaps.append("Upside section is too generic to be actionable")
    if not report.get("risks") or report.get("risks") == generic:
        gaps.append("Risk section is too generic to be actionable")

    return gaps


def reflect(report: dict) -> dict:
    """Score the report against a rubric and add concrete gaps."""

    report = dict(report)

    report_text = f"""
        Ticker: {report.get('ticker', 'Unknown')}
        Summary: {report.get('summary', 'N/A')}
        Key Facts: {report.get('key_facts', [])}
        Upside: {report.get('upside', [])}
        Risks: {report.get('risks', [])}
        Catalysts: {report.get('catalysts', [])}
    """

    prompt = f"""Score this investment research report on 4 criteria
        (0-10 each):
        1. **Coverage**: Does it address company, market, and macro factors?
        2. **Evidence**: Are claims backed by data (specific numbers, sources)?
        3. **Recency**: Are facts fresh (last quarter or month)?
        4. **Balance**: Does it present both upside and downside?

        Report:
        {report_text}

        Respond as JSON:
        {{
            "coverage_score": <int>,
            "evidence_score": <int>,
            "recency_score": <int>,
            "balance_score": <int>,
            "overall_score": <int 0-40, then divide by 4>,
            "feedback": "<string with main observations>",
            "gaps": ["<gap 1>", "<gap 2>", ...]
        }}"""

    system = (
        "You are a critical evaluator of investment research. Be "
        "constructive but honest."
    )

    try:
        result = llm.complete_json(prompt, system=system)
    except Exception:
        result = {}

    heuristic_gaps = _heuristic_gaps(report)
    if not isinstance(result, dict):
        result = {}

    # Compute overall from components if not provided
    scores = [
        result.get("coverage_score", 6),
        result.get("evidence_score", 6),
        result.get("recency_score", 6),
        result.get("balance_score", 6),
    ]
    if "overall_score" not in result or result["overall_score"] == 0:
        result["overall_score"] = sum(scores) / len(scores)

    result["gaps"] = list(result.get("gaps", [])) + heuristic_gaps
    result["gaps"] = list(dict.fromkeys(result["gaps"]))

    return {
        "score": result["overall_score"],
        # Use empty string as default to indicate "no LLM feedback available"
        # Heuristic gaps still provide actionable feedback
        "feedback": result.get("feedback", ""),
        "gaps": result["gaps"],
    }
