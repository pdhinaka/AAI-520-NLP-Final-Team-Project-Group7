"""Self-reflection on the agent's own output.

reflect() combines three signals: an LLM critic scores four criteria
(coverage, evidence, recency, balance) from 0 to 10, a Python grounding
check looks up every number in the report in the gathered facts, and
heuristic checks flag missing or placeholder sections. The overall
score is the mean of the four criteria, computed here, not by the LLM.
"""

import json
import re

from finagent import llm
from finagent.workflows.evaluator_optimizer import grounding_check

CRITERIA = ("coverage", "evidence", "recency", "balance")
DEFAULT_SCORE = 6  # used when the critic leaves a criterion out
FACTS_CHARS = 9000  # facts shown to the critic (same as the writer)

PLACEHOLDER_TEXT = (
    "research completed",
    "based on gathered data",
    "hold pending full analysis",
    "no clear catalyst",
)

# Report fields the model writes (key_facts are computed in Python).
WRITTEN_FIELDS = ("summary", "investment_thesis", "upside", "risks",
                  "catalysts", "outlook")

# "52-week high", "12-month target": the number names a period, not a
# fact, so it is removed before the grounding check.
_PERIOD = re.compile(
    r"\b\d+[- ](?:day|week|month|quarter|year)s?\b", re.IGNORECASE)

REFLECT_SYSTEM = (
    "You are a critical evaluator of investment research. Be "
    "constructive but honest."
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


def _as_text(value) -> str:
    if isinstance(value, list):
        return "\n".join(f"- {v}" for v in value)
    return str(value or "")


def report_text(report: dict) -> str:
    """The report as plain text, in the order a reader sees it."""
    parts = [f"Ticker: {report.get('ticker', 'Unknown')}"]
    for key, label in (
        ("summary", "Summary"),
        ("key_facts", "Key facts"),
        ("investment_thesis", "Investment thesis"),
        ("recommendation", "Recommendation"),
        ("upside", "Upside"),
        ("risks", "Risks"),
        ("catalysts", "Catalysts"),
        ("outlook", "Outlook"),
    ):
        parts.append(f"{label}:\n{_as_text(report.get(key))}")
    return "\n\n".join(parts)


def _written_text(report: dict) -> str:
    """Only what the model wrote, for the grounding check."""
    text = "\n".join(_as_text(report.get(k)) for k in WRITTEN_FIELDS)
    return _PERIOD.sub(" ", text)


def _score(raw) -> int:
    try:
        return int(min(10, max(0, round(float(raw)))))
    except (TypeError, ValueError):
        return DEFAULT_SCORE


def _previous_part(previous: list[str] | None) -> str:
    """The fixable gaps on the draft this one revises. Without them the
    critic found new small items each round and revisions never
    settled (same fix as the evaluator-optimizer's grader)."""
    if not previous:
        return ""
    lines = "\n".join(f"- {g}" for g in previous)
    return (
        "\n\nThis is a revision. The fixable gaps you listed on the "
        f"previous draft were:\n{lines}\nFirst check whether each was "
        "fixed. List a fixable gap only if one of these is still open or "
        "the revision introduced an error.")


def reflect(report: dict, facts: dict | None = None,
            previous: list[str] | None = None) -> dict:
    """Score the report against a rubric and add concrete gaps.

    facts are the data the report was written from (the agent passes
    its compact facts). With facts, the critic sees them and every
    number the model wrote is checked against them. previous is the
    fixable list from the review of the draft this one revises.

    Returns {score, coverage_score, evidence_score, recency_score,
    balance_score, grounding, feedback, gaps, fixable, needs_data};
    scores are 0-10. fixable gaps can be fixed from the facts (the
    agent revises for them); needs_data gaps need data the run didn't
    gather (the agent saves them as lessons for the next run). gaps is
    both lists together.
    """
    report = dict(report)
    facts_part = ""
    if facts:
        facts_json = json.dumps(facts, default=str)[:FACTS_CHARS]
        facts_part = f"\n\nFacts the report was written from:\n{facts_json}"

    prompt = f"""Score this investment research report on 4 criteria.
Give each an integer from 0 to 10.
1. coverage: Does it address the company, the market, and macro
   factors?
2. evidence: Are claims backed by specific numbers from the facts?
3. recency: Are the facts fresh (latest quarter, recent news), and does
   the report say how recent they are?
4. balance: Does it give both upside and downside fair weight?

Report:
{report_text(report)}{facts_part}{_previous_part(previous)}

Check each number's period and unit against the facts (quarter over
quarter vs year over year, a percentage vs a multiple). Then sort the
problems you find into two lists:
- fixable: the fix uses only the facts above, and you name the fact
  (a number with the wrong period or unit, a claim the facts
  contradict, a point the facts support but the report leaves out).
- needs_data: anything that needs data not in the facts (a filing
  detail, an estimate, a peer comparison, verification elsewhere).
Leave fixable empty if the report already uses the facts well.

Respond as JSON:
{{"coverage_score": <int 0-10>, "evidence_score": <int 0-10>,
  "recency_score": <int 0-10>, "balance_score": <int 0-10>,
  "feedback": "<main observations, 2-3 sentences>",
  "fixable": ["<the problem and how to fix it from the facts>", ...],
  "needs_data": ["<missing item and the data that would fill it>",
                 ...]}}"""

    try:
        result = llm.complete_json_cached(prompt, system=REFLECT_SYSTEM,
                                          max_tokens=1024)
    except Exception:  # noqa: BLE001 - heuristics still apply
        result = {}
    if not isinstance(result, dict):
        result = {}

    scores = {c: _score(result.get(f"{c}_score")) for c in CRITERIA}
    grounding = (grounding_check(_written_text(report), facts)
                 if facts else None)

    # "gaps" is the older single list; it counts as fixable.
    fixable = [str(g) for g in
               (result.get("fixable") or result.get("gaps") or [])]
    needs_data = [str(g) for g in result.get("needs_data") or []]
    if grounding and grounding["ungrounded"]:
        shown = ", ".join(grounding["ungrounded"][:8])
        fixable.insert(0, "These numbers are not in the gathered facts; "
                          f"correct or remove them: {shown}")
    fixable = list(dict.fromkeys(fixable + _heuristic_gaps(report)))
    gaps = list(dict.fromkeys(fixable + needs_data))

    return {
        "score": round(sum(scores.values()) / len(scores), 2),
        **{f"{c}_score": scores[c] for c in CRITERIA},
        "grounding": grounding,
        # Empty when the LLM gave no feedback; gaps still apply.
        "feedback": str(result.get("feedback", "")),
        "gaps": gaps,
        "fixable": fixable,
        "needs_data": needs_data,
    }
