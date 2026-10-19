from finagent import pipeline

AGENT = {"recommendation": "Buy (medium conviction)", "rating": "Buy",
         "reflection_score": 8.0}
CHAIN = {"digest": {"headline": "h"}, "steps": {"classify": []}}
ROUTED = {"analyses": {"earnings": {"signal": "bullish"},
                       "news": {"signal": "neutral"}}}
LOOP = {"final": "## AAPL\n**View:** Buy, with medium conviction",
        "best_iteration": 2, "passed": True,
        "history": [{"overall": 0.7}, {"overall": 0.84}]}


def test_run_all_wires_stages_and_compares(monkeypatch):
    calls = {}
    monkeypatch.setattr(pipeline.research_agent, "run",
                        lambda t, memory_dir=None, save=True: AGENT)
    monkeypatch.setattr(pipeline.prompt_chain, "run_chain",
                        lambda t, save=True: CHAIN)

    def run_routing(t, chain=None, save=True):
        calls["chain"] = chain
        return ROUTED

    def run_loop(ctx, max_iters, threshold, save=True):
        calls["threshold"] = threshold
        return LOOP

    monkeypatch.setattr(pipeline.routing, "run_routing", run_routing)
    monkeypatch.setattr(pipeline.eo, "context_from_routing", lambda r: {})
    monkeypatch.setattr(pipeline.eo, "run_loop", run_loop)

    out = pipeline.run_all("aapl", save=False)

    assert calls == {"chain": CHAIN, "threshold": 0.8}
    c = out["comparison"]
    assert c["brief_view"] == "Buy" and c["agree"] is True
    assert c["brief_score"] == 0.84
    assert c["specialist_signals"] == {"earnings": "bullish",
                                       "news": "neutral"}


def test_run_all_reuses_given_results(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should be reused, not rerun")

    for mod, name in ((pipeline.research_agent, "run"),
                      (pipeline.prompt_chain, "run_chain"),
                      (pipeline.routing, "run_routing"),
                      (pipeline.eo, "run_loop")):
        monkeypatch.setattr(mod, name, boom)
    out = pipeline.run_all("AAPL", agent={**AGENT, "rating": "Hold"},
                           chain=CHAIN, routed=ROUTED, loop=LOOP)
    assert out["comparison"]["agree"] is False


def test_brief_view_parsing():
    assert pipeline.brief_view("**View:** hold (low)") == "Hold"
    assert pipeline.brief_view("no view line") is None


def test_format_summary_reads_like_a_report():
    agent = {
        "ticker": "AAPL", "model": "m", "recommendation": "Hold (low "
        "conviction)", "rating": "Hold", "reflection_score": 8.0,
        "summary": "Short summary.", "turns": 2, "best_draft": 1,
        "plan": {"source": "llm", "steps": ["Check price", "Read news"],
                 "step_tools": [["get_price_summary"], []],
                 "memory_used": [{"note": "gap: macro",
                                  "effect": "added a macro step"}]},
        "notes_loaded": [{"note": "gap: macro"}],
        "trace": [{"turn": 1, "tool": "get_earnings", "ok": False},
                  {"turn": 1, "tool": "get_filing_text", "ok": True},
                  {"turn": 1, "tool": "get_filing_text", "ok": True}],
        "coverage": {"price": True, "macro": False},
        "reflection_history": [
            {"draft": d, "coverage_score": 7, "evidence_score": 7,
             "recency_score": 7, "balance_score": 7, "score": 7.0,
             "grounding": {"ratio": 1.0}, "fixable": ["x"] * (1 - d)}
            for d in (0, 1)],
        "notes_saved": [{"category": "lesson"}, {"category": "finding"}],
    }
    chain = {"steps": {"ingest": [{}] * 3,
                       "preprocess": {"articles": [{}] * 2},
                       "classify": [{"labels": {"relevant": True}},
                                    {"labels": {}}]},
             "digest": {"sentiment_counts": {"positive": 1}}}
    routed = {"counts": {"by_specialist": {"news": 2},
                         "by_method": {"rule": 2}},
              "analyses": {"news": {"signal": "neutral",
                                    "confidence": "low"}},
              "errors": [{"kind": "earnings"}]}
    loop = {"history": [{"overall": 0.7}, {"overall": 0.85}],
            "passed": True, "best_iteration": 2,
            "final": "**View:** Buy"}
    out = {"ticker": "AAPL", "agent": agent, "chain": chain,
           "routing": routed, "loop": loop,
           "comparison": pipeline.compare(agent, routed, loop)}
    text = pipeline.format_summary(out)
    assert "1. Plan (llm, 2 steps, 1 notes loaded)" in text
    assert "memory: gap: macro -> added a macro step" in text
    assert "turn 1: get_earnings (error), get_filing_text x2" in text
    assert "evidence missing: macro" in text
    assert "<- kept" in text.splitlines()[[
        i for i, line in enumerate(text.splitlines())
        if line.strip().startswith("1 ")][0]]
    assert "0.700 -> 0.850" in text
    assert "skipped tools: earnings" in text
    assert "agent: Hold (low conviction)  |  brief: Buy  ->  disagree" \
        in text
    assert max(len(line) for line in text.splitlines()) <= 80
