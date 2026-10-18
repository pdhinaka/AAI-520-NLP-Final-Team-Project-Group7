"""End-to-end run: the research agent plus the three workflows.

    from finagent import pipeline
    out = pipeline.run_all("AAPL")
    out["comparison"]

Order: the agent researches the ticker (its get_news_digest tool runs
the prompt chain), then routing sends the chain's articles and the
data tools' results to the specialists, then the evaluator-optimizer
writes and refines a brief from the specialists' findings. The full
chain result routing needs is fetched with the same arguments as the
agent's news tool, so its LLM calls replay from the cache instead of
running again. Results computed earlier (for example in the notebook)
can be passed in and are reused as is.
"""

import re

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
        # Same defaults as get_news_digest, so this is a cache replay.
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
