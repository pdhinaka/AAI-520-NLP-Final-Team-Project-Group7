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
