"""AS_OF_DATE pins the date windows that end up in cache keys."""

from finagent import config
from finagent.tools import macro, news


def test_newsapi_from_date_follows_as_of(monkeypatch):
    seen = []

    def fake_get_json(url, params=None, headers=None):
        seen.append(params)
        return {"status": "ok", "articles": []}

    monkeypatch.setattr(config, "AS_OF_DATE", "2026-10-18")
    monkeypatch.setattr(news, "get_json", fake_get_json)
    news.fetch_newsapi("AAPL", days=7, query="AAPL")
    assert seen[0]["from"] == "2026-10-11"


def test_fred_start_follows_as_of(monkeypatch):
    seen = []

    def fake_get_json(url, params=None):
        seen.append(params)
        if url.endswith("/series"):
            return {"seriess": [{"title": "t"}]}
        return {"observations": []}

    monkeypatch.setattr(config, "AS_OF_DATE", "2026-10-18")
    monkeypatch.setattr(macro, "get_json", fake_get_json)
    macro.get_series("UNRATE")
    assert seen[1]["observation_start"] == "2024-10-18"


def test_today_defaults_to_real_date(monkeypatch):
    from datetime import date

    monkeypatch.setattr(config, "AS_OF_DATE", "")
    assert config.today() == date.today()
