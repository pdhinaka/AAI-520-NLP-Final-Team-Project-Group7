"""Offline tests for the news source fixes (NewsAPI domains, Yahoo
search fallback, empty results not cached)."""

from finagent import cache
from finagent.tools import news


def test_cache_skips_empty_results_when_asked():
    calls = []

    def fetch():
        calls.append(1)
        return []

    assert cache.cached("t", {"a": 1}, fetch, cache_empty=False) == []
    assert cache.cached("t", {"a": 1}, fetch, cache_empty=False) == []
    assert len(calls) == 2
    cache.cached("t", {"a": 2}, fetch)
    cache.cached("t", {"a": 2}, fetch)
    assert len(calls) == 3


def _payload(*urls):
    return {"status": "ok", "articles": [
        {"source": {"name": "S"}, "title": f"Apple {u}",
         "description": "d", "url": u,
         "publishedAt": "2026-09-28T00:00:00Z"} for u in urls]}


def test_newsapi_uses_domains_and_tops_up(monkeypatch):
    seen = []

    def fake_get_json(url, params=None, headers=None):
        seen.append(params)
        if "domains" in params:
            return _payload("a", "b")
        return _payload("b", "c", "d")

    monkeypatch.setattr(news, "get_json", fake_get_json)
    monkeypatch.setattr(news, "_company_name", lambda t: "Apple")
    out = news.fetch_newsapi("AAPL")
    assert [a["url"] for a in out] == ["a", "b", "c", "d"]
    assert all(p["sortBy"] == "relevancy" for p in seen)
    assert "reuters.com" in seen[0]["domains"]
    assert "pypi.org" in seen[1]["excludeDomains"]


def test_newsapi_skips_open_search_when_domains_suffice(monkeypatch):
    seen = []

    def fake_get_json(url, params=None, headers=None):
        seen.append(params)
        return _payload(*"abcdef")

    monkeypatch.setattr(news, "get_json", fake_get_json)
    monkeypatch.setattr(news, "_company_name", lambda t: "Apple")
    assert len(news.fetch_newsapi("AAPL", limit=4)) == 4
    assert len(seen) == 1


class _FakeYF:
    """Stands in for the yfinance module."""

    def __init__(self, ticker_news, search_news):
        self.ticker_news, self.search_news = ticker_news, search_news
        self.search_calls = 0
        fake = self

        class Ticker:
            def __init__(self, t):
                self.news = fake.ticker_news

        class Search:
            def __init__(self, q, **kwargs):
                fake.search_calls += 1
                self.news = fake.search_news

        self.Ticker, self.Search = Ticker, Search


def test_yahoo_falls_back_to_search_and_filters(monkeypatch):
    fake = _FakeYF([], [
        {"title": "Apple up", "publisher": "Reuters", "link": "u1",
         "providerPublishTime": 1790000000, "relatedTickers": ["AAPL"]},
        {"title": "Msft up", "publisher": "Reuters", "link": "u2",
         "providerPublishTime": 1790000000, "relatedTickers": ["MSFT"]},
    ])
    monkeypatch.setattr(news, "yf", fake)
    out = news.fetch_yahoo("AAPL")
    assert [a["title"] for a in out] == ["Apple up"]
    assert out[0]["provider"] == "yahoo"


def test_yahoo_empty_is_not_cached(monkeypatch):
    fake = _FakeYF([], [])
    monkeypatch.setattr(news, "yf", fake)
    assert news.fetch_yahoo("AAPL") == []
    assert news.fetch_yahoo("AAPL") == []
    assert fake.search_calls == 2


def test_cache_retries_a_stored_empty_entry():
    cache.cached("t", {"a": 3}, lambda: [])  # stored the old way
    assert cache.cached("t", {"a": 3}, lambda: [1],
                        cache_empty=False) == [1]


def test_get_news_shares_limit_across_providers(monkeypatch):
    def arts(provider, day, n):
        return [{"title": f"{provider} {i}", "provider": provider,
                 "published_at": f"2026-09-{day:02d}T{i:02d}:00:00Z",
                 "text": ""} for i in range(n)]

    monkeypatch.setattr(news, "fetch_newsapi",
                        lambda *a, **k: arts("newsapi", 25, 10))
    monkeypatch.setattr(news, "fetch_yahoo", lambda t: arts("yahoo", 29, 10))
    out = news.get_news("AAPL", limit=6)
    providers = [a["provider"] for a in out]
    assert providers.count("newsapi") == 3 and providers.count("yahoo") == 3
    assert out[0]["published_at"] > out[-1]["published_at"]
