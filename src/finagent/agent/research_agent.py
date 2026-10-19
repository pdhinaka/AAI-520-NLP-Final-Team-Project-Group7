"""Main agent loop: plan, act with tools, reflect, remember.

    from finagent.agent import research_agent
    result = research_agent.run("AAPL")
    result["recommendation"], result["plan"]["steps"], result["trace"]

run() loads notes from earlier runs, asks the planner for a plan, works
through it in a tool-use loop (the model picks the tools), has the
model write the report from the gathered facts, reflects on the report
and revises it when the critique is weak, then saves new notes for the
next run. Every LLM call is disk-cached, so with CACHE_MODE=offline a
run replays exactly.
"""

import json
import logging
import time
from datetime import datetime

from finagent import config, llm
from finagent.tools import registry
from finagent.workflows.routing import (
    FINANCIAL_SECTORS,
    _eps_facts,
    _fundamental_facts,
    _pct_change,
)

from . import memory, planner, reflection

log = logging.getLogger(__name__)

MAX_TURNS = 8
TOOL_RESULT_CHARS = 6000  # per tool result sent back to the model
FACTS_CHARS = 9000  # compact facts sent to the report writer
REVISE_BELOW = 7.5  # revise when the reflection score is under this
MAX_REVISIONS = 2
RATINGS = ("Buy", "Hold", "Sell")
CONVICTIONS = ("low", "medium", "high")

# Evidence areas and the tools that cover them, for the coverage check.
AREAS = {
    "price": ("get_price_summary", "get_price_history"),
    "fundamentals": ("get_company_info", "get_financials"),
    # Statements carry reported EPS and net income, so they stand in
    # for get_earnings (without the analyst estimate) when it fails.
    "earnings": ("get_earnings", "get_financials"),
    "news": ("get_news_digest", "get_news"),
    "macro": ("get_macro_snapshot", "get_series"),
    "filings": ("get_recent_filings", "get_filing_text"),
}

AGENT_SYSTEM = (
    "You are an equity research agent. Work through the research plan "
    "using the tools. For each step pick the tool or tools that answer "
    "it. Call several tools in the same turn when they don't depend on "
    "each other. If a tool returns an error, try a different tool or "
    "different arguments instead of repeating the call, and never "
    "repeat a call that already succeeded. When every step is covered, "
    "stop calling tools and reply with two or three sentences on what "
    "you gathered and any step you could not cover."
)

WRITER_SYSTEM = (
    "You are a senior equity research analyst. You write concise, "
    "evidence-based reports and never invent numbers."
)

REPORT_FORMAT = (
    "Return JSON with these keys:\n"
    "summary: 2-3 sentences\n"
    "investment_thesis: 3-4 sentences\n"
    "rating: Buy, Hold, or Sell\n"
    "conviction: low, medium, or high\n"
    "upside: 2-4 strings, each backed by a number from the facts\n"
    "risks: 2-4 strings, each backed by a number or a news item\n"
    "catalysts: 2-3 strings, with timing when the facts give it\n"
    "outlook: one sentence\n\n"
    "Returns, growth, and margins in the facts are fractions (0.05 = "
    "5%); write them as percentages. Money is in raw units "
    "(416161000000 = $416.16B). Use only numbers that appear in the "
    "facts. Say how recent the data is (use the as_of and reported "
    "dates)."
)


# ---------------------------------------------------------- formatting

def _money(value) -> str:
    v = float(value)
    for scale, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(v) >= scale:
            return f"${v / scale:.2f}{unit}"
    return f"${v:,.2f}"


def _pct(fraction) -> str:
    return f"{float(fraction) * 100:+.1f}%"


def _is_quarters(value) -> bool:
    return (isinstance(value, list) and bool(value)
            and isinstance(value[0], dict) and "reported_eps" in value[0])


def _is_macro(value) -> bool:
    return (isinstance(value, dict) and bool(value)
            and all(isinstance(v, dict) and "label" in v
                    for v in value.values()))


# ---------------------------------------------------------------- facts

def _fact_stats(facts: dict) -> list[str]:
    """Pull key facts out of the tool results into simple sentences.

    Works on the shape of each result, so it doesn't matter which tool
    call produced it. Uses the keys the tools really return
    (get_price_summary, yfinance camelCase info, Alpha Vantage
    quarters, FRED snapshot).
    """
    stats: list[str] = []
    for key, value in facts.items():
        if key in {"ticker", "gathered_at"}:
            continue

        if _is_quarters(value):
            q = value[0]
            line = f"latest EPS: {q['reported_eps']}"
            if q.get("estimated_eps") is not None:
                line += f" vs {q['estimated_eps']} estimated"
            if q.get("surprise_pct") is not None:
                line += f" ({q['surprise_pct']:+.2f}% surprise)"
            stats.append(f"{line}, quarter ending "
                         f"{q.get('fiscal_date_ending')}")
            continue
        if _is_macro(value):
            for series in ("FEDFUNDS", "DGS10"):
                s = value.get(series) or {}
                if s.get("latest") is not None:
                    stats.append(f"{s['label']}: {s['latest']}%")
            continue
        if not isinstance(value, dict):
            continue

        if "last_close" in value:
            line = f"last close: {value['last_close']}"
            if value.get("as_of"):
                line += f" (as of {value['as_of']})"
            stats.append(line)
        ret_1y = value.get("return_1y")
        if ret_1y is not None:
            stats.append(f"1-year return: {ret_1y:+.1%}")
        if value.get("low_52w") is not None and value.get("high_52w"):
            stats.append(f"52-week range: {value['low_52w']:.2f} to "
                         f"{value['high_52w']:.2f}")
        if value.get("marketCap"):
            stats.append(f"market cap: {_money(value['marketCap'])}")
        if value.get("trailingPE"):
            stats.append(f"trailing P/E: {value['trailingPE']:.1f}")
        if value.get("revenueGrowth") is not None:
            stats.append(f"revenue growth (yoy): "
                         f"{_pct(value['revenueGrowth'])}")
        if value.get("targetMeanPrice"):
            line = f"analyst mean target: {value['targetMeanPrice']:.2f}"
            if value.get("numberOfAnalystOpinions"):
                n = int(value["numberOfAnalystOpinions"])
                line += f" ({n} analysts)"
            stats.append(line)

    return stats[:10]


def _financial_facts(value: dict, sector: str | None = None) -> dict:
    """Routing's fundamentals summary, with the growth period named.

    _fundamental_facts compares the two latest periods, which for
    quarterly statements is quarter over quarter. Left as
    "revenue_growth", the report writer called it year over year.
    """
    if sector in FINANCIAL_SECTORS:
        value = {**value, "sector": sector}
    out = _fundamental_facts(value)
    if not out:
        return value
    if not value.get("quarterly"):
        return {**out, "period_type": "fiscal year"}
    out["period_type"] = "quarter"
    out["revenue_growth_qoq"] = out.pop("revenue_growth")
    income = value.get("income_statement") or {}
    periods = sorted(income, reverse=True)
    if len(periods) >= 5:
        out["revenue_growth_yoy"] = _pct_change(
            income[periods[0]].get("Total Revenue"),
            income[periods[4]].get("Total Revenue"))
    return out


def _base(key: str) -> str:
    return key.split("(")[0]


def compact_facts(facts: dict) -> dict:
    """The gathered facts, trimmed for an LLM prompt.

    Long lists are cut, computed summaries replace raw statements, and
    a few derived numbers (upside to target, 52-week position) are
    added so the report can cite them and grounding can find them.
    """
    out: dict = {}
    sector = next((v.get("sector") for k, v in facts.items()
                   if _base(k) == "get_company_info"
                   and isinstance(v, dict)), None)
    for key, value in facts.items():
        if key in {"ticker", "gathered_at"}:
            continue
        name = _base(key)
        if name == "get_earnings" and _is_quarters(value):
            out[key] = {"summary": _eps_facts(value),
                        "recent_quarters": value[:4]}
        elif name == "get_financials" and isinstance(value, dict):
            out[key] = _financial_facts(value, sector)
        elif name == "get_company_info" and isinstance(value, dict):
            info = dict(value)
            info["longBusinessSummary"] = (
                info.get("longBusinessSummary") or "")[:300]
            # yfinance reports debt/equity in percent (78.4 = 0.784x);
            # the writer read it as a multiple.
            if "debtToEquity" in info:
                info["debtToEquity_pct"] = info.pop("debtToEquity")
            out[key] = info
        elif name == "get_news_digest" and isinstance(value, dict):
            out[key] = {k: value.get(k) for k in (
                "headline", "summary", "catalysts", "risks",
                "sentiment_counts", "topic_counts")}
        elif name == "get_news" and isinstance(value, list):
            out[key] = [{k: a.get(k) for k in
                         ("title", "source", "published_at")}
                        for a in value[:8]]
        elif name == "get_price_history" and isinstance(value, list):
            out[key] = {"days": len(value),
                        "first": value[0] if value else None,
                        "last": value[-1] if value else None}
        elif name == "get_series" and isinstance(value, dict):
            out[key] = {**{k: value.get(k) for k in
                           ("series_id", "title", "units")},
                        "recent": (value.get("observations") or [])[-6:]}
        elif name == "get_recent_filings" and isinstance(value, list):
            out[key] = value[:3]
        elif name == "get_filing_text" and isinstance(value, dict):
            out[key] = {k: (str(v)[:1500] if isinstance(v, str) else v)
                        for k, v in value.items()}
        else:
            out[key] = value

    price = next((v for k, v in out.items()
                  if _base(k) == "get_price_summary"), None)
    info = next((v for k, v in out.items()
                 if _base(k) == "get_company_info"), None)
    derived = {}
    if isinstance(price, dict) and price.get("last_close"):
        last = price["last_close"]
        hi, lo = price.get("high_52w"), price.get("low_52w")
        if hi and lo and hi > lo:
            derived["position_in_52w_range"] = round(
                (last - lo) / (hi - lo), 2)
        if isinstance(info, dict) and info.get("targetMeanPrice"):
            derived["upside_to_target"] = round(
                info["targetMeanPrice"] / last - 1, 4)
    if derived:
        out["derived"] = derived
    return out


def _facts_json(facts: dict) -> str:
    text = json.dumps(compact_facts(facts), default=str)
    return text[:FACTS_CHARS]


def coverage(facts: dict) -> dict:
    """Which evidence areas have at least one successful tool result.
    Reported in the run, not used to stop the loop."""
    names = {_base(k) for k in facts}
    return {area: any(t in names for t in tools)
            for area, tools in AREAS.items()}


# ----------------------------------------------------------- tool loop

def _plan_text(plan: dict) -> str:
    lines = []
    tools = plan.get("step_tools") or []
    for i, step in enumerate(plan["steps"], 1):
        hint = tools[i - 1] if i - 1 < len(tools) else []
        suffix = f" (suggested tools: {', '.join(hint)})" if hint else ""
        lines.append(f"{i}. {step}{suffix}")
    return "\n".join(lines)


def _fact_key(facts: dict, name: str, args: dict) -> str:
    """Tool name, or name(args) when the tool already ran with other
    arguments (two get_series calls, for example)."""
    if name not in facts:
        return name
    arg_text = ",".join(f"{k}={v}" for k, v in sorted(args.items()))
    return f"{name}({arg_text})"


def gather(ticker: str, plan: dict, max_turns: int = MAX_TURNS) -> dict:
    """Run the tool-use loop. Returns {facts, trace, turns, stopped,
    final_text, errors}."""
    # Built outside the f-string: a backslash inside {} is a
    # SyntaxError before Python 3.12.
    plan_text = _plan_text(plan)
    messages = [{
        "role": "user",
        "content": (f"Research {ticker} for an investment report.\n\n"
                    f"Research plan:\n{plan_text}\n\n"
                    "Work through every step with the tools."),
    }]
    specs = registry.tool_specs()
    facts: dict = {}
    trace: list[dict] = []
    errors: list[dict] = []
    stopped, final_text, turn = "max_turns", "", 0

    for turn in range(1, max_turns + 1):
        reply = llm.create_message_cached(messages, system=AGENT_SYSTEM,
                                          tools=specs, max_tokens=1024)
        messages.append({"role": "assistant", "content": reply["content"]})
        calls = [b for b in reply["content"] if b["type"] == "tool_use"]
        if not calls:
            stopped, final_text = "model_finished", llm.text_of(reply)
            break

        results = []
        for c in calls:
            args = c.get("input") or {}
            result = registry.call(c["name"], args)
            text = registry.to_json(result, max_chars=TOOL_RESULT_CHARS)
            failed = isinstance(result, dict) and "error" in result
            entry = {"turn": turn, "tool": c["name"], "args": args,
                     "ok": not failed, "chars": len(text),
                     "truncated": text.endswith('"[truncated]"'),
                     "error": result["error"] if failed else None}
            trace.append(entry)
            if failed:
                errors.append(entry)
            else:
                facts[_fact_key(facts, c["name"], args)] = result
            results.append({"type": "tool_result", "tool_use_id": c["id"],
                            "content": text, "is_error": failed})
            log.info("turn %d: %s(%s) %s", turn, c["name"], args,
                     "ok" if not failed else "error")
        messages.append({"role": "user", "content": results})

    return {"facts": facts, "trace": trace, "turns": turn,
            "stopped": stopped, "final_text": final_text,
            "errors": errors}


# --------------------------------------------------------------- report

def _build_summary(ticker: str, facts: dict) -> str:
    """Build a concise summary from the facts we actually gathered."""
    facts_list = _fact_stats(facts)
    if not facts_list:
        return (
            f"{ticker} research is in progress; no tool facts have been "
            "collected yet."
        )

    return f"{ticker} research indicates: {'; '.join(facts_list[:5])}."


def _build_thesis(ticker: str, facts: dict) -> str:
    """Create a simple investment thesis from the collected facts."""
    summary = _fact_stats(facts)
    if not summary:
        return (
            f"{ticker} has not yet generated a thesis because no tool "
            "facts were collected."
        )
    return (
        f"{ticker} remains constructive as long as the recent operating "
        f"facts hold up: {summary[0]}."
    )


def _digest(facts: dict) -> dict:
    return next((v for k, v in facts.items()
                 if _base(k) == "get_news_digest" and isinstance(v, dict)),
                {})


def _build_upside(facts: dict) -> list[str]:
    upside = []
    for key, value in facts.items():
        if not isinstance(value, dict):
            continue
        ret_1y = value.get("return_1y")
        if ret_1y is not None and ret_1y > 0:
            upside.append(
                f"Price momentum is positive at {ret_1y:+.1%} over the "
                "last year."
            )
        if value.get("marketCap"):
            upside.append(
                "The market profile remains large, with market cap at "
                f"{_money(value['marketCap'])}."
            )
        if value.get("revenueGrowth") is not None:
            upside.append(
                "Revenue growth remains supportive at "
                f"{_pct(value['revenueGrowth'])}."
            )
    return upside[:3] or [
        "The latest tool facts suggest a constructive setup."
    ]


def _build_risks(facts: dict) -> list[str]:
    risks = [str(r) for r in _digest(facts).get("risks") or []]
    return risks[:4] or [
        "Macro or competitive pressure remains the main watch item."
    ]


def _build_catalysts(facts: dict) -> list[str]:
    catalysts = [str(c) for c in _digest(facts).get("catalysts") or []]
    return catalysts[:3] or [
        "The next earnings or product update is the main catalyst to "
        "watch."
    ]


def _build_outlook(facts: dict) -> str:
    for key, value in facts.items():
        if not isinstance(value, dict):
            continue
        ret_1y = value.get("return_1y")
        if ret_1y is not None:
            return "Moderately bullish" if ret_1y > 0 else "Neutral"
    return "Neutral"


def _build_recommendation(facts: dict) -> str:
    if any(
        isinstance(value, dict)
        and value.get("return_1y") is not None
        and value["return_1y"] > 0
        for value in facts.values()
    ):
        return "Moderate Buy"
    return "Hold"


def fallback_report(ticker: str, facts: dict) -> dict:
    """Template report from the facts, used if the writer call fails."""
    return {
        "ticker": ticker,
        "summary": _build_summary(ticker, facts),
        "key_facts": _fact_stats(facts),
        "investment_thesis": _build_thesis(ticker, facts),
        "recommendation": _build_recommendation(facts),
        "rating": None,
        "conviction": None,
        "upside": _build_upside(facts),
        "risks": _build_risks(facts),
        "catalysts": _build_catalysts(facts),
        "outlook": _build_outlook(facts),
        "source": "fallback",
    }


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)] if value else []


def _finish(ticker: str, written, facts: dict) -> dict:
    """Merge the model's report with the computed key facts. Missing
    fields fall back to the template builders."""
    if not isinstance(written, dict):
        return fallback_report(ticker, facts)
    base = fallback_report(ticker, facts)
    rating = str(written.get("rating", "")).strip().title()
    conviction = str(written.get("conviction", "")).strip().lower()
    if rating in RATINGS:
        conviction = conviction if conviction in CONVICTIONS else "medium"
        recommendation = f"{rating} ({conviction} conviction)"
    else:
        rating, conviction = None, None
        recommendation = base["recommendation"]
    return {
        "ticker": ticker,
        "summary": str(written.get("summary") or base["summary"]),
        "key_facts": base["key_facts"],
        "investment_thesis": str(written.get("investment_thesis")
                                 or base["investment_thesis"]),
        "recommendation": recommendation,
        "rating": rating,
        "conviction": conviction,
        "upside": _as_list(written.get("upside")) or base["upside"],
        "risks": _as_list(written.get("risks")) or base["risks"],
        "catalysts": (_as_list(written.get("catalysts"))
                      or base["catalysts"]),
        "outlook": str(written.get("outlook") or base["outlook"]),
        "source": "llm",
    }


def write_report(ticker: str, facts: dict, plan: dict) -> dict:
    """Have the model write the report from the gathered facts."""
    steps = "\n".join(f"- {s}" for s in plan["steps"])
    prompt = (
        f"Write an investment research report on {ticker}.\n\n"
        f"Research plan that was followed:\n{steps}\n\n"
        f"{REPORT_FORMAT}\n\nFacts:\n{_facts_json(facts)}"
    )
    try:
        written = llm.complete_json_cached(prompt, system=WRITER_SYSTEM,
                                           max_tokens=1500)
    except Exception as err:  # noqa: BLE001 - template report instead
        log.warning("Report writer failed, using template: %r", err)
        return fallback_report(ticker, facts)
    return _finish(ticker, written, facts)


def _written_part(report: dict) -> dict:
    keys = ("summary", "investment_thesis", "rating", "conviction",
            "upside", "risks", "catalysts", "outlook")
    return {k: report.get(k) for k in keys}


def revise_report(ticker: str, report: dict, review: dict,
                  facts: dict) -> dict:
    """Rewrite the report to fix the gaps the facts can fix."""
    todo = review.get("fixable") or review["gaps"]
    gaps = "\n".join(f"- {g}" for g in todo)
    prompt = (
        f"Revise this investment research report on {ticker}.\n\n"
        f"Critique: {review['feedback']}\n\n"
        f"Gaps to fix:\n{gaps}\n\n"
        "Fix every gap you can with the facts below. If a gap needs data "
        "that is not in the facts, say plainly that it was not available "
        "instead of inventing it. Correct or remove any number flagged "
        "as not in the facts.\n\n"
        f"Current report:\n{json.dumps(_written_part(report))}\n\n"
        f"{REPORT_FORMAT}\n\nFacts:\n{_facts_json(facts)}"
    )
    try:
        written = llm.complete_json_cached(prompt, system=WRITER_SYSTEM,
                                           max_tokens=1500)
    except Exception as err:  # noqa: BLE001 - keep the current draft
        log.warning("Revision failed, keeping the draft: %r", err)
        return report
    return _finish(ticker, written, facts)


def _needs_revision(review: dict) -> bool:
    grounding = review.get("grounding") or {}
    return (review["score"] < REVISE_BELOW
            or bool(grounding.get("ungrounded"))
            or bool(review.get("fixable")))


def reflect_and_revise(ticker: str, report: dict, facts: dict,
                       max_revisions: int = MAX_REVISIONS) -> dict:
    """Draft, reflect, and revise while the critique is weak.

    Returns {best, best_index, drafts, history}; history[i] is the
    review of drafts[i].
    """
    compact = compact_facts(facts)
    drafts = [report]
    history = [reflection.reflect(report, compact)]
    while _needs_revision(history[-1]) and len(drafts) <= max_revisions:
        drafts.append(revise_report(ticker, drafts[-1], history[-1],
                                    facts))
        history.append(reflection.reflect(
            drafts[-1], compact, previous=history[-1].get("fixable")))

    def rank(i):
        # A draft that needs no revision wins, then one with no
        # ungrounded numbers (an NVDA run kept a higher-scoring draft
        # with an invented figure), then the score.
        r = history[i]
        grounding = r.get("grounding") or {}
        return (not _needs_revision(r), not grounding.get("ungrounded"),
                r["score"], grounding.get("ratio", 1.0))

    best = max(range(len(drafts)), key=rank)
    for i, review in enumerate(history):
        review["draft"] = i
        review["revise"] = _needs_revision(review)
    return {"best": drafts[best], "best_index": best, "drafts": drafts,
            "history": history}


# --------------------------------------------------------------- memory

def _short_error(error: str, ticker: str) -> str:
    """Error type and first words, without the ticker or request
    details, so a tool lesson is general."""
    text = error.split(" for {")[0].split(":", 2)
    short = ":".join(text[:2]).replace(ticker, "<ticker>")
    return short[:80].rstrip()


def lessons(ticker: str, report: dict, review: dict, run: dict) -> dict:
    """Notes for the next run: {ticker: [...], general: [...]}."""
    ticker_notes: list[dict] = []

    # finding: what this run concluded, with a few key numbers
    if report.get("recommendation"):
        thesis = str(report.get("investment_thesis", "")).split(". ")[0]
        ticker_notes.append({
            "category": "finding",
            "note": f"Prior view: {report['recommendation']}. "
                    f"Thesis: {thesis.rstrip('.')}.",
        })
    numbers = [s for s in report.get("key_facts", [])
               if not s.startswith("last close")][:3]
    if numbers:
        ticker_notes.append({"category": "finding",
                             "note": "Prior key numbers: "
                                     + "; ".join(numbers)})

    # lesson: data the report lacked (what the next run should fetch),
    # then gaps left unfixed, and areas with no evidence
    left = review.get("needs_data", []) + review.get("fixable", [])
    for gap in (left or review.get("gaps", []))[:3]:
        ticker_notes.append({"category": "lesson",
                             "note": f"Earlier report gap: {gap}"})
    for area, covered in run["coverage"].items():
        if not covered and area != "filings":
            tools = " or ".join(AREAS[area])
            ticker_notes.append({
                "category": "lesson",
                "note": f"Earlier run gathered no {area} data; call "
                        f"{tools} early.",
            })

    # tool: errors and what worked instead (ticker-independent)
    general: list[dict] = []
    for err in run["errors"]:
        area = next((a for a, tools in AREAS.items()
                     if err["tool"] in tools), None)
        later = [t for t in run["trace"]
                 if t["ok"] and t["turn"] >= err["turn"]
                 and t["tool"] != err["tool"]
                 and area and t["tool"] in AREAS[area]]
        fix = (f"{later[0]['tool']} worked instead" if later
               else "no substitute was found")
        general.append({
            "category": "tool",
            "note": f"{err['tool']} returned an error "
                    f"({_short_error(err['error'], ticker)}); {fix}.",
        })
    return {"ticker": ticker_notes, "general": general}


# ------------------------------------------------------------------ run

def run(ticker: str, memory_dir=None, save: bool = True) -> dict:
    """End-to-end research: plan → act → reflect → remember.

    Returns the final report (summary, key_facts, investment_thesis,
    recommendation, upside, risks, catalysts, outlook) plus plan,
    trace, facts_used, coverage, reflection_history, drafts,
    notes_loaded, notes_saved, usage, and timings. With save=True the
    run is written to data/runs/agent_<TICKER>_<stamp>.json.
    """
    ticker = ticker.upper()
    usage_before = dict(llm.USAGE)
    start = time.perf_counter()
    timings = {}

    # Phase 1: Load prior knowledge
    records = memory.load_records(ticker, memory_dir)

    # Phase 2: Create plan
    t = time.perf_counter()
    plan = planner.make_plan_detailed(ticker, records)
    timings["plan"] = round(time.perf_counter() - t, 2)
    log.info("plan for %s (%s): %s", ticker, plan["source"],
             plan["steps"])

    # Phase 3: Gather facts via the tool-use loop
    t = time.perf_counter()
    loop = gather(ticker, plan)
    timings["tools"] = round(time.perf_counter() - t, 2)
    facts = loop["facts"]
    cover = coverage(facts)

    # Phase 4: Draft the report, then reflect and revise
    t = time.perf_counter()
    draft = write_report(ticker, facts, plan)
    critique = reflect_and_revise(ticker, draft, facts)
    timings["report_and_reflect"] = round(time.perf_counter() - t, 2)
    report = critique["best"]
    final_review = critique["history"][critique["best_index"]]

    # Phase 5: Save notes for the next run
    notes = lessons(ticker, report, final_review,
                    {**loop, "coverage": cover})
    saved = memory.save_notes(ticker, notes["ticker"],
                              memory_dir=memory_dir)
    saved += memory.save_general(notes["general"], memory_dir=memory_dir)
    timings["total"] = round(time.perf_counter() - start, 2)

    result = {
        **report,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reflection_score": final_review["score"],
        "reflection_feedback": final_review["feedback"],
        "plan": plan,
        "trace": loop["trace"],
        "turns": loop["turns"],
        "stopped": loop["stopped"],
        "agent_note": loop["final_text"],
        "facts_used": list(facts),
        "facts": facts,
        "coverage": cover,
        "drafts": critique["drafts"],
        "best_draft": critique["best_index"],
        "reflection_history": critique["history"],
        "notes_loaded": records,
        "notes_saved": saved,
        "model": config.LLM_MODEL,
        "usage": {k: llm.USAGE[k] - usage_before[k] for k in llm.USAGE},
        "timings": timings,
        "saved_to": None,
    }
    if save:
        runs = config.DATA_DIR / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = runs / f"agent_{ticker}_{stamp}.json"
        result["saved_to"] = str(path)
        path.write_text(json.dumps(result, indent=2, default=str))
        log.info("saved agent run to %s", path)
    return result
