import pytest

from finagent import config
from finagent.agent import memory


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own empty cache and fake keys."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    # Nothing a test runs may write to the real data/ or memory/.
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "MEMORY_DIR", tmp_path / "memory")
    monkeypatch.setattr(memory, "MEMORY_DIR", None)
    monkeypatch.setattr(config, "CACHE_MODE", "normal")
    monkeypatch.setattr(config, "LOCAL_NEWS_CSV", tmp_path / "news.csv")
    for key in ("NEWSAPI_KEY", "FRED_API_KEY", "ALPHAVANTAGE_API_KEY",
                "SEC_USER_AGENT"):
        monkeypatch.setattr(config, key, "test")
    yield
