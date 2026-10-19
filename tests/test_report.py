from finagent import llm
from finagent.agent import planner, reflection, research_agent

PRICE = {
    "ticker": "AAPL", "as_of": "2026-10-16", "last_close": 220.5,
    "return_1m": 0.02, "return_3m": 0.05, "return_6m": 0.1,
    "return_1y": 0.18, "high_52w": 240.0, "low_52w": 180.0,
    "volatility_annualized": 0.25,
}
INFO = {
    "ticker": "AAPL", "shortName": "Apple Inc.", "marketCap": 3.0e12,
    "trailingPE": 33.4, "revenueGrowth": 0.061,
    "targetMeanPrice": 250.0, "numberOfAnalystOpinions": 40,
    "recommendationKey": "buy", "longBusinessSummary": "Makes phones.",
}
DIGEST = {
    "headline": "Mixed week.", "summary": "iPhone demand [1].",
    "catalysts": ["iPhone launch"], "risks": ["China demand slowdown"],
    "sentiment_counts": {"positive": 2}, "topic_counts": {"product": 2},
}
TOOL_DATA = {
    "get_price_summary": PRICE,
    "get_company_info": INFO,
    "get_news_digest": DIGEST,
}

REPORT = {
    "summary": "AAPL closed at 220.5, up 18.0% over the year.",
    "investment_thesis": "Revenue growth of 6.1% supports the case.",
    "rating": "buy", "conviction": "medium",
    "upside": ["Trading 13.4% below the 250.0 mean target"],
    "risks": ["China demand slowdown"],
    "catalysts": ["iPhone launch"],
    "outlook": "Constructive.",
}


def tool_reply(*names):
    return {"stop_reason": "tool_use", "usage": {},
            "content": [{"type": "tool_use", "id": f"tu-{n}", "name": n,
                         "input": {"ticker": "AAPL"}} for n in names]}


def text_reply(text="Done."):
    return {"stop_reason": "end_turn", "usage": {},
            "content": [{"type": "text", "text": text}]}


def fake_json(scores=(8, 8, 8, 8)):
    def complete_json(prompt, system=None, **kwargs):
        if system == planner.PLANNER_SYSTEM:
            return {"steps": [{"step": "Get latest price data",
                               "tools": ["get_price_summary"]},
                              {"step": "Summarize investment case",
                               "tools": ["get_news_digest"]}],
                    "memory_used": []}
        if system == research_agent.WRITER_SYSTEM:
            return REPORT
        if system == reflection.REFLECT_SYSTEM:
            return dict(zip(("coverage_score", "evidence_score",
                             "recency_score", "balance_score"), scores),
                        feedback="Solid.", gaps=[])
        raise AssertionError(f"unexpected system prompt {system!r}")
    return complete_json


def test_make_plan_falls_back_to_default_when_llm_reply_is_invalid(
    monkeypatch,
):
    """The planner should still return a usable plan if the model is
    noisy."""

    monkeypatch.setattr(
        planner.llm,
        "complete_json",
        lambda *args, **kwargs: {"unexpected": "shape"},
    )

    plan = planner.make_plan("AAPL")

    assert isinstance(plan, list)
    assert len(plan) >= 5
    assert any("price" in step.lower() for step in plan)
    assert any("risk" in step.lower() for step in plan)
    detailed = planner.make_plan_detailed("AAPL")
    assert detailed["source"] == "fallback"


def test_report_is_fact_driven(monkeypatch):
    """The final brief should use actual tool facts, not placeholders."""

    replies = iter([
        tool_reply("get_price_summary", "get_company_info"),
        tool_reply("get_news_digest"),
        text_reply(),
    ])
    monkeypatch.setattr(llm, "create_message_cached",
                        lambda *a, **k: next(replies))
    monkeypatch.setattr(llm, "complete_json", fake_json())
    monkeypatch.setattr(research_agent.registry, "call",
                        lambda name, args=None: TOOL_DATA[name])

    report = research_agent.run("AAPL", save=False)

    assert report["ticker"] == "AAPL"
    assert report["plan"]["source"] == "llm"
    assert "220.5" in report["summary"]
    assert report["recommendation"] == "Buy (medium conviction)"
    assert any("market cap: $3.00T" == s for s in report["key_facts"])
    assert any("revenue growth (yoy): +6.1%" == s
               for s in report["key_facts"])
    assert report["upside"] != ["Based on gathered data"]
    assert report["risks"] != ["Based on gathered data"]
    assert report["catalysts"]
    assert report["outlook"] != "Hold pending full analysis"
    # two tool turns, then the model stopped on its own
    assert [t["turn"] for t in report["trace"]] == [1, 1, 2]
    assert report["stopped"] == "model_finished"
    assert report["reflection_history"][0]["score"] == 8.0


def test_report_handles_missing_1y_return():
    """Reports should skip missing 1Y returns instead of crashing on None."""

    facts = {
        "ticker": "AAPL",
        "get_price_summary": {**PRICE, "return_1y": None},
        "get_company_info": INFO,
    }

    stats = research_agent._fact_stats(facts)
    summary = research_agent._build_summary("AAPL", facts)
    upside = research_agent._build_upside(facts)
    outlook = research_agent._build_outlook(facts)

    assert any("last close" in s for s in stats)
    assert not any("1-year return" in s for s in stats)
    assert "AAPL" in summary
    assert isinstance(upside, list)
    assert outlook in {"Neutral", "Moderately bullish"}
