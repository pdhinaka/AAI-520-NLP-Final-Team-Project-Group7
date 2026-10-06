"""Offline tests for routing: the LLM and data tools are faked."""

import pytest

from finagent import config, llm
from finagent.workflows import routing as rt


def article(i, topic, relevant=True, sentiment="neutral"):
    return {"id": i, "title": f"Apple story {i}", "text": "Body " * 200,
            "source": "Reuters", "published_at": "2026-09-27T12:00:00Z",
            "ticker": "AAPL",
            "labels": {"topic": topic, "sentiment": sentiment,
                       "relevant": relevant, "reason": "test"}}


ARTICLES = [
    article(1, "earnings", sentiment="positive"),
    article(2, "legal", sentiment="negative"),
    article(3, "analyst_rating"),
    article(4, "other"),
    article(5, "other"),
    article(6, "product", relevant=False),
]
EXTRACTED = [{"id": 1, "event": "Apple beat Q4 estimates",
              "figures": [{"value": "$94.9B", "context": "revenue"}]}]

PRICE = {"ticker": "AAPL", "as_of": "2026-10-02", "last_close": 250.0,
         "return_1m": 0.05, "return_3m": 0.1, "return_6m": 0.2,
         "return_1y": 0.3, "high_52w": 260.0, "low_52w": 160.0,
         "volatility_annualized": 0.27}
INFO = {"shortName": "Apple Inc.", "trailingPE": 38.0,
        "targetMeanPrice": 275.0, "recommendationKey": "buy"}
EARNINGS = [
    {"fiscal_date_ending": f"2026-0{q}-30", "reported_eps": 1.5 + q / 10,
     "estimated_eps": 1.5, "surprise": s, "surprise_pct": s * 100}
    for q, s in zip(range(6, 0, -1), (0.1, -0.05, 0.2, 0.1, 0.0, 0.3))
]
FINANCIALS = {
    "ticker": "AAPL", "quarterly": False,
    "income_statement": {
        "2025-09-30": {"Total Revenue": 400e9, "Net Income": 100e9},
        "2024-09-30": {"Total Revenue": 380e9, "Net Income": 95e9},
    },
    "cash_flow": {"2025-09-30": {"Free Cash Flow": 110e9}},
}
MACRO = {"DGS10": {"label": "10-year Treasury yield", "latest": 4.1,
                   "change_vs_year_ago": -0.3}}


class FakeLLM:
    """Stands in for llm.complete_json and records the system prompts."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, system=None, **kwargs):
        self.calls.append(system)
        if system == rt.ROUTER_SYSTEM:
            return [{"id": "news-4", "specialist": "market",
                     "reason": "about the share price"},
                    {"id": "news-5", "specialist": "astrology",
                     "reason": "bad answer"}]
        return {"signal": "bullish", "confidence": "very",
                "summary": "Looks fine.", "key_points": ["a", 2],
                "risks": ["r"]}


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "complete_json", fake)
    return fake


@pytest.fixture
def fake_tools(monkeypatch):
    def broken():
        raise RuntimeError("no key")

    monkeypatch.setattr(rt, "_tool_calls", lambda t: {
        "price_summary": lambda: PRICE,
        "company_info": lambda: INFO,
        "financials": lambda: FINANCIALS,
        "earnings": lambda: EARNINGS,
        "macro": broken,
    })


def test_items_from_articles_skips_irrelevant_and_merges_facts():
    items = rt.items_from_articles(ARTICLES, EXTRACTED)
    assert [it["id"] for it in items] == [f"news-{i}" for i in range(1, 6)]
    assert items[0]["event"] == "Apple beat Q4 estimates"
    assert items[0]["topic"] == "earnings"
    assert len(items[0]["text"]) <= rt.ARTICLE_CHARS + 4


def test_items_from_tools_records_failures(fake_tools):
    errors = []
    items = rt.items_from_tools("aapl", errors)
    assert [it["kind"] for it in items] == [
        "price_summary", "company_info", "financials", "earnings"]
    assert errors == [{"kind": "macro", "error": "RuntimeError: no key"}]


def test_rules_route_structured_data_and_labeled_news(fake_llm):
    items = rt.items_from_articles(ARTICLES[:3]) + [
        {"id": "earnings", "kind": "earnings", "data": []},
        {"id": "macro", "kind": "macro", "data": {}},
    ]
    out = rt.route_items(items, "AAPL")
    assert [d["specialist"] for d in out] == [
        "earnings", "news", "market", "earnings", "market"]
    assert {d["method"] for d in out} == {"rule"}
    assert fake_llm.calls == []  # nothing needed the LLM


def test_llm_routes_unlabeled_items_in_one_call(fake_llm):
    items = rt.items_from_articles(ARTICLES[3:5])
    out = rt.route_items(items, "AAPL")
    assert fake_llm.calls == [rt.ROUTER_SYSTEM]
    assert out[0] == {"item_id": "news-4", "kind": "news",
                      "title": "Apple story 4", "specialist": "market",
                      "method": "llm", "reason": "about the share price"}
    # unknown specialist from the model falls back to the default
    assert out[1]["specialist"] == rt.DEFAULT_SPECIALIST
    assert out[1]["method"] == "default"


def test_route_without_llm_uses_default(fake_llm):
    item = rt.items_from_articles(ARTICLES[3:4])[0]
    d = rt.route(item, "AAPL", use_llm=False)
    assert (d["specialist"], d["method"]) == ("news", "default")
    assert fake_llm.calls == []


def test_earnings_facts_computes_beat_record_and_growth():
    facts = rt.earnings_facts([
        {"id": "earnings", "kind": "earnings", "data": EARNINGS},
        {"id": "financials", "kind": "financials", "data": FINANCIALS},
    ])
    eps = facts["eps"]
    assert (eps["quarters"], eps["beats"]) == (6, 4)
    assert eps["beat_rate"] == 0.67
    assert eps["avg_surprise_pct"] == pytest.approx(10.83, abs=0.01)
    assert eps["year_ago_eps"] == EARNINGS[4]["reported_eps"]
    fund = facts["fundamentals"]
    assert fund["period"] == "2025-09-30"
    assert fund["revenue_growth"] == pytest.approx(0.0526, abs=1e-4)
    assert fund["net_margin"] == 0.25
    assert fund["free_cash_flow"] == 110e9


def test_market_facts_range_position_and_upside():
    facts = rt.market_facts([
        {"id": "price_summary", "kind": "price_summary", "data": PRICE},
        {"id": "company_info", "kind": "company_info", "data": INFO},
        {"id": "macro", "kind": "macro", "data": MACRO},
    ])
    assert facts["price"]["position_in_52w_range"] == 0.9
    assert facts["valuation"]["upside_to_target"] == 0.1
    assert facts["macro"]["10-year Treasury yield"]["latest"] == 4.1


def test_analyzer_normalizes_reply_and_skips_empty(fake_llm):
    items = rt.items_from_articles(ARTICLES[1:2])
    out = rt.news_analyzer(items, "AAPL")
    assert out["signal"] == "bullish"
    assert out["confidence"] == "low"  # "very" is not allowed
    assert out["key_points"] == ["a", "2"]
    assert out["facts"]["sentiment_counts"] == {"negative": 1}

    calls = len(fake_llm.calls)
    empty = rt.market_analyzer([], "AAPL")
    assert empty["n_items"] == 0 and empty["signal"] == "neutral"
    assert len(fake_llm.calls) == calls


def test_run_routing_end_to_end(fake_llm, fake_tools, monkeypatch,
                                tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    chain = {"digest": {"headline": "Mixed week."},
             "steps": {"classify": ARTICLES, "extract": EXTRACTED}}
    result = rt.run_routing("aapl", chain=chain)

    assert result["ticker"] == "AAPL"
    assert len(result["decisions"]) == 9  # 5 articles + 4 tools
    assert result["counts"]["by_specialist"] == {
        "earnings": 3, "news": 2, "market": 4}
    assert result["counts"]["by_method"] == {
        "rule": 7, "llm": 1, "default": 1}
    assert set(result["analyses"]) == set(rt.SPECIALISTS)
    assert result["analyses"]["earnings"]["facts"]["eps"]["beats"] == 4
    assert result["errors"][0]["kind"] == "macro"
    assert result["saved_to"].startswith(str(tmp_path))
    # one router call plus one call per specialist
    assert len(fake_llm.calls) == 4
