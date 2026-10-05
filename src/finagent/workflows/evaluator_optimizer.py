"""Evaluator-optimizer: generate an analysis, evaluate it, refine it.

Workflow Pattern 3. A generator drafts an investment brief from the
routing specialists' facts and views, an evaluator scores it against a
fixed rubric and writes feedback, and an optimizer revises the draft
using that feedback. The loop stops when the score passes the threshold
or after max_iters drafts:

    from finagent.workflows import evaluator_optimizer as eo

    ctx = eo.context_from_routing(routing.run_routing("AAPL"))
    result = eo.run_loop(ctx)
    result["final"]                    # best brief (markdown)
    [h["overall"] for h in result["history"]]   # score per iteration

The evaluator combines two signals. An LLM grader scores five criteria
from 1 to 5. A Python grounding check pulls every number out of the
brief and looks for it in the source facts; numbers it can't find are
listed in the feedback, which catches invented figures that an LLM
grader tends to miss. The overall score is the mean of all six
criteria divided by 5, so 0.8 means an average of 4 out of 5.

Generator, evaluator, and optimizer replies are cached on disk, so with
CACHE_MODE=offline a rerun replays the same drafts and scores.

Owner: Workstream C.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone

from finagent import config, llm
from finagent.cache import cached

log = logging.getLogger(__name__)

# What the LLM grader scores, 1-5 each. Order is the column order in
# the notebook's score table.
CRITERIA = {
    "coverage": "covers price and valuation, earnings and fundamentals, "
                "recent news, and the macro backdrop",
    "evidence": "every claim is backed by a specific number or a cited "
                "item id from the facts",
    "balance": "gives both the bull case and the bear case fair weight",
    "actionability": "states a clear Buy, Hold, or Sell view with "
                     "conviction and says what would change it",
    "clarity": "well organized, concise, no filler or repetition",
}
# Scored in Python, not by the LLM.
GROUNDING = "grounding"

MAX_FEEDBACK = 6  # feedback items passed to the optimizer
BRIEF_TOKENS = 1500

# Numbers in the brief, with an optional unit: 12.5%, $94.9B, 3.1 bn
_NUM = re.compile(
    r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)"
    r"(\s?(?:%|[kmbt]n?\b|billion\b|million\b|trillion\b))?",
    re.IGNORECASE,
)
_CITATION = re.compile(r"\[[^\]]*\]")  # [news-3], [2]
# The brief's last section holds hypothetical triggers ("downgrade if
# growth falls below 4%"), not claims of fact, so grounding skips it.
_TRIGGERS = re.compile(r"^#+\s*What would change.*", re.IGNORECASE
                       | re.MULTILINE | re.DOTALL)
_SCALES = (1, 100, 1e-3, 1e-6, 1e-9, 1e-12)  # fraction -> %, K, M, B, T


# ------------------------------------------------------------- context

def context_from_routing(result: dict) -> dict:
    """Shrink a routing.run_routing result to what the generator needs:
    each specialist's computed facts and its view, plus the news
    digest headline."""
    return {
        "ticker": result["ticker"],
        "news_headline": (result.get("digest") or {}).get("headline", ""),
        "specialists": {
            name: {k: a[k] for k in ("signal", "confidence", "summary",
                                     "key_points", "risks", "facts")}
            for name, a in result["analyses"].items()
        },
    }


def _context_json(context: dict) -> str:
    return json.dumps(context, indent=1, default=str)


def _ask_text(prompt: str, system: str, max_tokens: int = BRIEF_TOKENS,
              model: str | None = None) -> str:
    """One cached free-text LLM call (complete_json_cached's twin)."""
    params = {"model": model or config.LLM_MODEL, "system": system,
              "prompt": prompt, "max_tokens": max_tokens}
    return cached(
        "llm", params,
        lambda: llm.complete(prompt, system=system, model=model,
                             max_tokens=max_tokens).strip(),
        ttl_hours=24 * 365,
    )


# ------------------------------------------------------------ generator

GENERATOR_SYSTEM = (
    "You are a senior equity research analyst. You write short, "
    "evidence-based investment briefs for portfolio managers. Use only "
    "the facts provided and never invent numbers."
)

BRIEF_FORMAT = (
    "Format (markdown, at most 350 words):\n"
    "## {ticker} investment brief\n"
    "**View:** Buy, Hold, or Sell, with conviction (low, medium, high)\n"
    "### Thesis\n"
    "### Evidence (price and valuation, earnings and fundamentals, "
    "news, macro)\n"
    "### Risks\n"
    "### What would change our view\n\n"
    "Returns and margins in the facts are fractions (0.05 = 5%); show "
    "them as percentages. Cite news items by id, like [news-3]."
)


def generate(context: dict, model: str | None = None) -> str:
    """Draft an investment brief from the specialists' facts and views."""
    ticker = context.get("ticker", "")
    prompt = (
        f"Write an investment brief for {ticker} from the specialist "
        "team's findings below.\n\n"
        + BRIEF_FORMAT.format(ticker=ticker)
        + f"\n\nFindings:\n{_context_json(context)}"
    )
    return _ask_text(prompt, GENERATOR_SYSTEM, model=model)


# ------------------------------------------------------------ evaluator

def _parse_numbers(text: str) -> list[tuple[str, float, int, bool]]:
    """(as written, value, decimals, has_unit) for each number."""
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        has_unit = bool(m.group(2)) or m.group(0).startswith("$")
        out.append((m.group(0).strip(), value, decimals, has_unit))
    return out


def _numbers_in(text: str) -> list[tuple[str, float, int]]:
    """Numbers in the brief worth checking.

    Skips citations, years, and small whole numbers (list counts like
    "3 of 4 quarters"). Digits glued to letters (Q3, FY25) never match.
    """
    return [
        (written, value, decimals)
        for written, value, decimals, has_unit
        in _parse_numbers(_CITATION.sub(" ", text))
        if decimals or has_unit
        or not (value <= 10 or 1900 <= value <= 2100)
    ]


def _context_values(obj) -> list[float]:
    """Every number in the context, including ones inside strings like
    an extracted figure "$94.9B"."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [abs(float(obj))]
    if isinstance(obj, str):
        return [v for _, v, _, _ in _parse_numbers(obj)]
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, list):
        return [v for x in obj for v in _context_values(x)]
    return []


def grounding_check(analysis: str, context: dict) -> dict:
    """Share of the brief's numbers that appear in the source facts.

    A number counts as grounded if some context value, at any common
    scale (fraction to %, raw to K/M/B/T), rounds to it at the
    precision it was written with. Numbers the brief derived itself
    (for example a ratio of two facts) will show as ungrounded, which
    is the conservative choice. The "What would change our view"
    section is skipped. Returns {ratio, checked, ungrounded}.
    """
    found = _numbers_in(_TRIGGERS.sub("", analysis))
    if not found:
        return {"ratio": 1.0, "checked": 0, "ungrounded": []}
    candidates = {round(v * s, 6) for v in _context_values(context)
                  for s in _SCALES}
    ungrounded = []
    for written, value, decimals in found:
        tol = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(c - value) <= tol for c in candidates):
            ungrounded.append(written)
    ratio = 1 - len(ungrounded) / len(found)
    return {"ratio": round(ratio, 3), "checked": len(found),
            "ungrounded": ungrounded}


EVALUATOR_SYSTEM = (
    "You are a strict research director reviewing an analyst's "
    "investment brief before it goes to a portfolio manager. You grade "
    "harshly: a 5 means nothing to improve. Your feedback is specific "
    "and actionable."
)


def evaluate(analysis: str, context: dict,
             model: str | None = None) -> dict:
    """Score the brief against the rubric and return feedback.

    Returns {scores: {criterion: 1-5}, overall: 0-1, feedback: [str],
    strengths: [str], grounding: {ratio, checked, ungrounded}}.
    """
    rubric = "\n".join(f"{name}: {desc}"
                       for name, desc in CRITERIA.items())
    prompt = (
        "Grade this investment brief against the source facts.\n\n"
        f"Criteria (score each 1-5):\n{rubric}\n\n"
        "Return JSON with:\n"
        'scores: {"coverage": 1-5, "evidence": 1-5, ...}\n'
        "strengths: 1-3 short strings\n"
        "feedback: up to 5 specific fixes, most important first, each "
        "saying exactly what to add, cut, or change\n\n"
        # Without this the grader flags "6.43%" against 0.0643 as a
        # mismatch and evidence never improves; exact number checks are
        # done in code (grounding_check).
        "The facts store returns, growth, and margins as fractions "
        "(0.0643 = 6.43%) and money in raw units (416161000000 = "
        "$416.16B); treat those as matching. Number accuracy is checked "
        "separately, so judge evidence on whether claims are backed by "
        "facts, not on number formatting.\n\n"
        f"Brief:\n{analysis}\n\n"
        f"Source facts:\n{_context_json(context)}"
    )
    reply = llm.complete_json_cached(prompt, system=EVALUATOR_SYSTEM,
                                     max_tokens=1024, model=model)
    reply = reply if isinstance(reply, dict) else {}
    raw_scores = reply.get("scores") or {}

    scores = {}
    for name in CRITERIA:
        try:
            scores[name] = min(5.0, max(1.0, float(raw_scores[name])))
        except (KeyError, TypeError, ValueError):
            scores[name] = 1.0  # missing score counts against the draft
    grounding = grounding_check(analysis, context)
    scores[GROUNDING] = round(1 + 4 * grounding["ratio"], 2)

    feedback = [str(f) for f in reply.get("feedback") or []]
    if grounding["ungrounded"]:
        shown = ", ".join(grounding["ungrounded"][:8])
        feedback.insert(0, "These numbers do not appear in the source "
                           f"facts; correct or remove them: {shown}")
    overall = sum(scores.values()) / len(scores) / 5
    return {
        "scores": scores,
        "overall": round(overall, 3),
        "feedback": feedback[:MAX_FEEDBACK],
        "strengths": [str(s) for s in reply.get("strengths") or []],
        "grounding": grounding,
    }


# ------------------------------------------------------------ optimizer

OPTIMIZER_SYSTEM = (
    "You are a senior equity research analyst revising your own brief "
    "after review. Address every point of feedback, keep what the "
    "reviewer liked, and use only the facts provided."
)


def refine(analysis: str, feedback: list[str] | str, context: dict,
           model: str | None = None) -> str:
    """Revise the brief using the evaluator's feedback."""
    if isinstance(feedback, str):
        feedback = [feedback]
    ticker = context.get("ticker", "")
    notes = "\n".join(f"- {f}" for f in feedback)
    prompt = (
        f"Revise this investment brief for {ticker}.\n\n"
        f"Reviewer feedback to address:\n{notes}\n\n"
        f"Current brief:\n{analysis}\n\n"
        + BRIEF_FORMAT.format(ticker=ticker)
        + "\nReturn only the revised brief."
        + f"\n\nFindings:\n{_context_json(context)}"
    )
    return _ask_text(prompt, OPTIMIZER_SYSTEM, model=model)


# ----------------------------------------------------------------- loop

def passes(review: dict, threshold: float, min_score: float) -> bool:
    """A draft passes on a high average with no weak criterion, so one
    strong area can't hide a poor one."""
    return (review["overall"] >= threshold
            and min(review["scores"].values()) >= min_score)


def run_loop(context: dict, max_iters: int = 3, threshold: float = 0.8,
             min_score: float = 4.0, model: str | None = None,
             evaluator_model: str | None = None,
             save: bool = True) -> dict:
    """Generate, then evaluate and refine until the score passes.

    Stops when a draft passes (overall score at least threshold and
    every criterion at least min_score) or after max_iters drafts. A
    revision can score lower than the draft it revised, so the best
    draft is returned, not the last one. evaluator_model lets a
    different (for example stronger) model do the grading.

    Returns {ticker, final, best_iteration, passed, threshold,
    min_score, history, timings, usage, saved_to}; history has one
    entry per draft with its text, scores, overall score, pass flag,
    and feedback.
    """
    assert max_iters >= 1, "max_iters must be at least 1"
    usage_before = dict(llm.USAGE)
    history = []
    start = time.perf_counter()

    analysis = generate(context, model=model)
    for i in range(1, max_iters + 1):
        review = evaluate(analysis, context, model=evaluator_model)
        review["passed"] = passes(review, threshold, min_score)
        history.append({"iteration": i, "analysis": analysis,
                        "words": len(analysis.split()), **review})
        log.info("iteration %d: overall %.3f passed %s %s", i,
                 review["overall"], review["passed"], review["scores"])
        if review["passed"] or i == max_iters:
            break
        analysis = refine(analysis, review["feedback"], context,
                          model=model)

    # Prefer a passing draft, then the higher score.
    best = max(history, key=lambda h: (h["passed"], h["overall"]))
    result = {
        "ticker": context.get("ticker", ""),
        "model": model or config.LLM_MODEL,
        "evaluator_model": evaluator_model or config.LLM_MODEL,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "final": best["analysis"],
        "best_iteration": best["iteration"],
        "passed": best["passed"],
        "threshold": threshold,
        "min_score": min_score,
        "history": history,
        "timings": {"total": round(time.perf_counter() - start, 2)},
        "usage": {k: llm.USAGE[k] - usage_before[k] for k in llm.USAGE},
        "saved_to": None,
    }
    if save:
        runs = config.DATA_DIR / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = runs / f"eval_opt_{result['ticker']}_{stamp}.json"
        result["saved_to"] = str(path)
        path.write_text(json.dumps(result, indent=2, default=str))
        log.info("saved run to %s", path)
    return result
