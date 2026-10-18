"""End-to-end run: the research agent plus the three workflows.

    from finagent import pipeline
    out = pipeline.run_all("AAPL")
    out["comparison"]

Order: the agent researches the ticker (its get_news_digest tool runs
the prompt chain), then routing sends the chain's articles and the
data tools' results to the specialists, then the evaluator-optimizer
writes and refines a brief from the specialists' findings. Routing
uses the chain with its default 7-day window. The agent's news tool
runs the same chain with the window the model asks for (often 30
days), so the chain's LLM calls replay from the cache only when the
windows match. Results computed earlier (for example in the notebook)
can be passed in and are reused as is.
"""

import re
import textwrap
from collections import Counter

from finagent.agent import research_agent
from finagent.workflows import evaluator_optimizer as eo
from finagent.workflows import prompt_chain, routing

_VIEW = re.compile(r"\*\*View:\*\*\s*(Buy|Hold|Sell)", re.IGNORECASE)


def brief_view(brief: str) -> str | None:
    """Buy, Hold, or Sell from the brief's **View:** line."""
    m = _VIEW.search(brief or "")
    return m.group(1).title() if m else None


def compare(agent: dict, routed: dict, loop: dict) -> dict:
    """Agent view vs the evaluator-optimizer brief and the specialists."""
    view = brief_view(loop["final"])
    rating = agent.get("rating")
    return {
        "agent_recommendation": agent.get("recommendation"),
        "agent_rating": rating,
        "agent_reflection_score": agent.get("reflection_score"),
        "brief_view": view,
        "brief_score": loop["history"][loop["best_iteration"] - 1][
            "overall"],
        "brief_passed": loop["passed"],
        "agree": (rating == view) if rating and view else None,
        "specialist_signals": {name: a["signal"] for name, a
                               in routed["analyses"].items()},
    }


def run_all(ticker: str, memory_dir=None, threshold: float = 0.8,
            max_iters: int = 3, save: bool = True,
            agent: dict | None = None, chain: dict | None = None,
            routed: dict | None = None,
            loop: dict | None = None) -> dict:
    """Run (or reuse) every stage and compare the two recommendations.

    Returns {ticker, agent, chain, routing, loop, comparison}.
    """
    ticker = ticker.upper()
    if agent is None:
        agent = research_agent.run(ticker, memory_dir=memory_dir,
                                   save=save)
    if chain is None:
        # Default window: a cache replay only if the agent's news tool
        # was called with the same one.
        chain = prompt_chain.run_chain(ticker, save=save)
    if routed is None:
        routed = routing.run_routing(ticker, chain=chain, save=save)
    if loop is None:
        loop = eo.run_loop(eo.context_from_routing(routed),
                           max_iters=max_iters, threshold=threshold,
                           save=save)
    return {
        "ticker": ticker,
        "agent": agent,
        "chain": chain,
        "routing": routed,
        "loop": loop,
        "comparison": compare(agent, routed, loop),
    }


# ------------------------------------------------------------- summary

def _wrap(text: str, indent: str = "   ", hang: str | None = None) -> str:
    return textwrap.fill(str(text), 78, initial_indent=indent,
                         subsequent_indent=hang or indent)


def _turns(trace: list[dict]) -> list[str]:
    """One line per tool-loop turn: tools in call order, repeats
    counted, errors marked."""
    lines = []
    for turn in sorted({t["turn"] for t in trace}):
        calls = [t for t in trace if t["turn"] == turn]
        names = Counter(t["tool"] for t in calls)
        failed = {t["tool"] for t in calls if not t["ok"]}
        parts = [f"{name}{f' x{k}' if k > 1 else ''}"
                 f"{' (error)' if name in failed else ''}"
                 for name, k in names.items()]
        lines.append(_wrap(f"turn {turn}: {', '.join(parts)}",
                           hang=" " * 11))
    return lines


def format_summary(out: dict) -> str:
    """A run_all result as a short, readable report for a live demo:
    the agent's plan, tools, reflection and memory, then the three
    workflows and the two recommendations side by side."""
    a, chain, routed, loop = (out["agent"], out["chain"],
                              out["routing"], out["loop"])
    c = out["comparison"]
    lines = [f"{out['ticker']} end-to-end research run "
             f"(model {a.get('model', '?')})", ""]

    plan = a["plan"]
    tools = plan.get("step_tools") or []
    lines.append(f"1. Plan ({plan['source']}, {len(plan['steps'])} steps,"
                 f" {len(a['notes_loaded'])} notes loaded)")
    for i, step in enumerate(plan["steps"]):
        hint = ", ".join(tools[i]) if i < len(tools) and tools[i] else ""
        lines.append(_wrap(f"{i + 1}. {step}", hang=" " * 6))
        if hint:
            lines.append(f"      tools: {hint}")
    for u in plan.get("memory_used") or []:
        note = u.get("note", u) if isinstance(u, dict) else u
        effect = u.get("effect", "") if isinstance(u, dict) else ""
        lines.append(_wrap(f"memory: {note} -> {effect}", "     "))

    trace = a["trace"]
    errors = sum(not t["ok"] for t in trace)
    lines += ["", f"2. Tools: {len(trace)} calls, "
                  f"{len({t['tool'] for t in trace})} tools, "
                  f"{a['turns']} turns, {errors} errors"]
    lines += _turns(trace)
    missing = [k for k, v in a["coverage"].items() if not v]
    lines.append("   evidence: all areas covered" if not missing else
                 f"   evidence missing: {', '.join(missing)}")

    lines += ["", "3. Self-reflection (scores 0-10, revise below "
                  f"{research_agent.REVISE_BELOW} or on fixable gaps)",
              "   draft  cov  evid  rec  bal  score  grounded  fixable"]
    for h in a["reflection_history"]:
        g = (h.get("grounding") or {}).get("ratio", 1.0)
        kept = "  <- kept" if h["draft"] == a["best_draft"] else ""
        lines.append(
            f"   {h['draft']:>5}  {h['coverage_score']:>3}  "
            f"{h['evidence_score']:>4}  {h['recency_score']:>3}  "
            f"{h['balance_score']:>3}  {h['score']:>5}  {g:>8.2f}  "
            f"{len(h.get('fixable', [])):>7}{kept}")

    lines += ["", f"4. Agent report: {a['recommendation']}",
              _wrap(a.get("summary", ""))]

    saved = Counter(n.get("category", "?") for n in a["notes_saved"])
    lines += ["", f"5. Memory: saved {len(a['notes_saved'])} notes ("
                  + ", ".join(f"{k} {v}" for k, v in saved.items())
                  + ") for the next run"]

    steps = chain["steps"]
    relevant = sum(x["labels"].get("relevant", False)
                   for x in steps["classify"])
    sent = (chain.get("digest") or {}).get("sentiment_counts", {})
    counts = routed["counts"]
    lines += [
        "", "6. Workflows",
        f"   prompt chain: {len(steps['ingest'])} articles, "
        f"{len(steps['preprocess']['articles'])} kept, {relevant} "
        f"relevant",
        "   news sentiment: " + ", ".join(
            f"{v} {k}" for k, v in sent.items()),
        "   routing: " + ", ".join(
            f"{k} {v}" for k, v in counts["by_specialist"].items())
        + " (" + ", ".join(f"{v} by {k}" for k, v
                           in counts["by_method"].items()) + ")",
        _wrap("specialists: " + ", ".join(
            f"{k} {x['signal']} ({x['confidence']})"
            for k, x in routed["analyses"].items()), hang=" " * 5),
        "   evaluator-optimizer: " + " -> ".join(
            f"{h['overall']:.3f}" for h in loop["history"])
        + f" (passed: {loop['passed']}, kept iteration "
          f"{loop['best_iteration']})",
    ]
    if routed.get("errors"):
        lines.append("   skipped tools: " + ", ".join(
            e["kind"] for e in routed["errors"]))

    agree = {True: "agree", False: "disagree"}.get(c["agree"], "n/a")
    lines += ["", "7. Two views",
              f"   agent: {c['agent_recommendation']}  |  brief: "
              f"{c['brief_view']}  ->  {agree}"]
    return "\n".join(lines)
