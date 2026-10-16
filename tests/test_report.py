from types import SimpleNamespace

from finagent.agent import planner, research_agent


def test_make_plan_falls_back_to_default_when_llm_reply_is_invalid(monkeypatch):
    """The planner should still return a usable research plan if the model is noisy."""

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


def test_report_is_fact_driven(monkeypatch):
    """The final brief should synthesize actual tool facts, not placeholders."""

    responses = iter([
        SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="get_price_summary",
                    input={"ticker": "AAPL"},
                    id="tool-1",
                )
            ],
        ),
        SimpleNamespace(stop_reason="end_turn", content=[]),
    ])

    monkeypatch.setattr(
        research_agent.memory,
        "load_notes",
        lambda ticker: [],
    )
    monkeypatch.setattr(
        research_agent.memory,
        "save_notes",
        lambda ticker, notes: None,
    )
    monkeypatch.setattr(
        research_agent.planner,
        "make_plan",
        lambda ticker, notes: ["Get latest price data", "Summarize investment case"],
    )
    monkeypatch.setattr(
        research_agent.llm,
        "create_message",
        lambda messages, tools=None, **kwargs: next(responses),
    )
    monkeypatch.setattr(
        research_agent.registry,
        "call",
        lambda name, args=None: {
            "ticker": "AAPL",
            "last_close": 220.5,
            "return_1y": 0.18,
            "market_cap": "$3.0T",
            "risk_factors": ["China demand slowdown"],
        },
    )
    monkeypatch.setattr(
        research_agent.reflection,
        "reflect",
        lambda report: {"score": 8.0, "feedback": "Good evidence base", "gaps": []},
    )

    report = research_agent.run("AAPL")

    assert report["ticker"] == "AAPL"
    assert "AAPL" in report["summary"]
    assert "220.5" in report["summary"] or "$3.0T" in report["summary"]
    assert "investment_thesis" in report
    assert "recommendation" in report
    assert report["investment_thesis"]
    assert report["recommendation"]
    assert report["upside"] != ["Based on gathered data"]
    assert report["risks"] != ["Based on gathered data"]
    assert report["catalysts"]
    assert report["outlook"] != "Hold pending full analysis"
