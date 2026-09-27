"""Offline tests: API responses are mocked, no network or keys needed."""

import json

import pytest

from finagent import cache, config, llm
from finagent.tools import (
    earnings,
    filings,
    macro,
    market_data,
    news,
    registry,
)


def test_cache_roundtrip_and_offline(monkeypatch):
    calls = []

    def fetch():
        calls.append(1)
        return {"x": 1}

    assert cache.cached("t", {"a": 1}, fetch) == {"x": 1}
    assert cache.cached("t", {"a": 1}, fetch) == {"x": 1}
    assert len(calls) == 1

    monkeypatch.setattr(config, "CACHE_MODE", "offline")
    assert cache.cached("t", {"a": 1}, fetch, ttl_hours=0) == {"x": 1}
    with pytest.raises(cache.CacheMissError):
        cache.cached("t", {"a": 2}, fetch)


def test_summarize_prices():
    records = [
        {"date": f"2026-01-{i:02d}", "close": 100 + i} for i in range(1, 31)
    ]
    s = market_data.summarize_prices(records)
    assert s["last_close"] == 130
    assert s["return_1m"] == round(130 / 109 - 1, 4)
    assert s["return_1y"] is None
    assert s["high_52w"] == 130 and s["low_52w"] == 101


def test_parse_newsapi():
    payload = {"status": "ok", "articles": [
        {"source": {"name": "Reuters"}, "title": "Apple beats",
         "description": "Strong quarter", "content": "Body",
         "url": "u", "publishedAt": "2026-09-01T00:00:00Z"},
        {"source": {}, "title": "[Removed]"},
    ]}
    arts = news.parse_newsapi(payload, "AAPL")
    assert len(arts) == 1
    assert arts[0]["source"] == "Reuters"
    assert arts[0]["text"] == "Strong quarter Body"


def test_parse_yahoo_both_formats():
    items = [
        {"content": {"title": "New", "summary": "s",
                     "pubDate": "2026-09-02T10:00:00Z",
                     "provider": {"displayName": "Yahoo"},
                     "canonicalUrl": {"url": "u1"}}},
        {"title": "Old", "publisher": "WSJ", "link": "u2",
         "providerPublishTime": 1788000000},
    ]
    arts = news.parse_yahoo(items, "AAPL")
    assert [a["title"] for a in arts] == ["New", "Old"]
    assert arts[1]["source"] == "WSJ"
    assert arts[1]["published_at"].startswith("2026")


def test_get_news_auto_merges_and_dedupes(monkeypatch):
    a = {"title": "Same", "published_at": "2026-09-01", "text": ""}
    b = {"title": "same ", "published_at": "2026-09-02", "text": ""}
    c = {"title": "Other", "published_at": "2026-09-03", "text": ""}
    monkeypatch.setattr(news, "fetch_newsapi", lambda *a_, **k: [a, c])
    monkeypatch.setattr(news, "fetch_yahoo", lambda t: [b])
    out = news.get_news("aapl")
    assert [x["title"] for x in out] == ["Other", "Same"]


def test_local_news_fallback(monkeypatch):
    config.LOCAL_NEWS_CSV.write_text(
        "Headline,Date,Stock\nApple news,2026-01-01,AAPL\n"
        "Other,2026-01-01,MSFT\n"
    )
    monkeypatch.setattr(config, "NEWSAPI_KEY", "")

    def boom(t):
        raise RuntimeError("offline")

    monkeypatch.setattr(news, "fetch_yahoo", boom)
    out = news.get_news("AAPL")
    assert [x["title"] for x in out] == ["Apple news"]
    assert out[0]["provider"] == "local"


def test_fred_series_and_snapshot(monkeypatch):
    def fake_get_json(url, params=None, headers=None):
        if url.endswith("/series"):
            return {"seriess": [{"title": "T", "units": "Percent",
                                 "frequency": "Monthly"}]}
        return {"observations": [
            {"date": "2025-08-01", "value": "4.0"},
            {"date": "2025-09-01", "value": "."},
            {"date": "2026-08-01", "value": "4.4"},
        ]}

    monkeypatch.setattr(macro, "get_json", fake_get_json)
    s = macro.get_series("unrate")
    assert s["observations"][1]["value"] is None
    snap = macro.get_macro_snapshot()
    assert snap["UNRATE"]["latest"] == 4.4
    assert snap["UNRATE"]["change_vs_year_ago"] == 0.4
    assert snap["CPIAUCSL"]["yoy_pct_change"] == 10.0


def test_sec_filings(monkeypatch):
    def fake_get_json(url, params=None, headers=None):
        assert headers["User-Agent"] == "test"
        if "company_tickers" in url:
            return {"0": {"cik_str": 320193, "ticker": "AAPL",
                          "title": "Apple Inc."}}
        return {"filings": {"recent": {
            "form": ["8-K", "10-Q", "10-K"],
            "accessionNumber": ["0001-26-1", "0001-26-2", "0001-25-3"],
            "primaryDocument": ["a.htm", "b.htm", "c.htm"],
            "filingDate": ["2026-09-01", "2026-08-01", "2025-11-01"],
            "reportDate": ["", "2026-06-30", "2025-09-30"],
        }}}

    monkeypatch.setattr(filings, "get_json", fake_get_json)
    out = filings.get_recent_filings("aapl", form="10-K,10-Q")
    assert [f["form"] for f in out] == ["10-Q", "10-K"]
    assert out[1]["url"].endswith("/320193/0001253/c.htm")


def test_filing_text_section(monkeypatch):
    html = ("<html><style>x{}</style><p>Contents: Risk Factors ... </p>"
            "<p>Item 1A. Risk Factors</p><p>Supply chain risk.</p></html>")
    monkeypatch.setattr(filings, "get_text", lambda url, headers: html)
    out = filings.get_filing_text("u", section="Risk Factors",
                                  max_chars=60)
    assert out["section_found"]
    assert out["text"].startswith("Risk Factors Supply chain")
    assert "x{}" not in out["text"]


def test_earnings_parse_and_rate_limit():
    out = earnings.parse_earnings({"quarterlyEarnings": [
        {"fiscalDateEnding": "2026-06-30", "reportedEPS": "1.5",
         "estimatedEPS": "1.4", "surprise": "0.1",
         "surprisePercentage": "7.1", "reportedDate": "2026-07-30"},
    ]}, quarters=4)
    assert out[0]["reported_eps"] == 1.5
    with pytest.raises(RuntimeError):
        earnings.parse_earnings({"Note": "rate limit"}, quarters=4)


def test_registry_specs_and_errors():
    specs = registry.tool_specs()
    assert {s["name"] for s in specs} == set(registry.TOOLS)
    for s in specs:
        assert s["description"]
        json.dumps(s)  # must be serializable for the API
    news_spec = next(s for s in specs if s["name"] == "get_news")
    assert news_spec["input_schema"]["required"] == ["ticker"]
    assert news_spec["input_schema"]["properties"]["days"]["type"] == \
        "integer"
    assert "error" in registry.call("nope")


def test_extract_json():
    assert llm.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.extract_json('Sure! [1, 2] done') == [1, 2]
    with pytest.raises(ValueError):
        llm.extract_json("no json here")


def test_missing_key_message(monkeypatch):
    monkeypatch.setattr(config, "FRED_API_KEY", "")
    with pytest.raises(config.MissingKeyError, match=".env"):
        macro.get_series("UNRATE")
