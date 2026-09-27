import pytest

from finagent import config


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own empty cache and fake keys."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "CACHE_MODE", "normal")
    monkeypatch.setattr(config, "LOCAL_NEWS_CSV", tmp_path / "news.csv")
    for key in ("NEWSAPI_KEY", "FRED_API_KEY", "ALPHAVANTAGE_API_KEY",
                "SEC_USER_AGENT"):
        monkeypatch.setattr(config, key, "test")
    yield
