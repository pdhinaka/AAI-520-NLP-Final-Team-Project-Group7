"""Financial news for a ticker from NewsAPI, Yahoo Finance, or a local CSV.

NewsAPI needs NEWSAPI_KEY (free developer plan: articles from the last
month, body text truncated). Yahoo needs no key. The local CSV at
data/local_news.csv is the offline fallback (for example a Kaggle
financial news dataset); see load_local_news for the expected columns.
"""

import csv
import logging
from datetime import datetime, timedelta, timezone

import yfinance as yf

from finagent import config
from finagent.cache import cached
from finagent.schemas import Article
from finagent.tools._http import get_json

log = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Business and tech outlets. Searching every source returned mostly noise
# (PyPI package pages, deal forums), so NewsAPI is queried on these first
# and an open search only tops up the results when they return too few.
NEWS_DOMAINS = (
    "reuters.com", "apnews.com", "cnbc.com", "bloomberg.com", "wsj.com",
    "ft.com", "marketwatch.com", "barrons.com", "investors.com",
    "fool.com", "finance.yahoo.com", "seekingalpha.com", "benzinga.com",
    "businessinsider.com", "fortune.com", "forbes.com", "axios.com",
    "nytimes.com", "theverge.com", "techcrunch.com", "arstechnica.com",
    "9to5mac.com", "macrumors.com", "appleinsider.com",
)
# Never useful for company news; excluded from the open search.
NOISE_DOMAINS = ("pypi.org", "ozbargain.com.au", "slickdeals.net")
MIN_DOMAIN_RESULTS = 5

# Suffixes stripped from company names to build a better search query
_NAME_SUFFIXES = (
    ", Inc.", " Inc.", " Inc", " Corporation", " Corp.", " Corp",
    " Company", " Co.", " Holdings", " plc", " Ltd.", " N.V.",
    " & Co.",
)


def _since(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def _company_name(ticker: str) -> str:
    """Best-effort short company name for search queries."""
    try:
        from finagent.tools.market_data import get_company_info

        name = get_company_info(ticker).get("shortName") or ""
    except Exception:  # noqa: BLE001 - name is optional
        return ""
    for suffix in _NAME_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def parse_newsapi(payload: dict, ticker: str) -> list[dict]:
    if payload.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {payload.get('message')}")
    out = []
    for a in payload.get("articles", []):
        if not a.get("title") or a["title"] == "[Removed]":
            continue
        text = " ".join(
            p for p in (a.get("description"), a.get("content")) if p
        )
        out.append(Article(
            title=a["title"],
            source=(a.get("source") or {}).get("name", ""),
            published_at=a.get("publishedAt", ""),
            url=a.get("url", ""),
            text=text,
            ticker=ticker,
            provider="newsapi",
        ).to_dict())
    return out


def fetch_newsapi(ticker: str, days: int = 7, limit: int = 20,
                  query: str | None = None,
                  domains: tuple = NEWS_DOMAINS) -> list[dict]:
    """Most relevant NewsAPI articles, from NEWS_DOMAINS first.

    If those outlets return fewer than MIN_DOMAIN_RESULTS articles, an
    open search (minus NOISE_DOMAINS) fills in the rest.
    """
    key = config.require("NEWSAPI_KEY")
    days = min(days, 29)  # free plan only goes back about a month
    if query is None:
        name = _company_name(ticker)
        query = f'"{name}" OR {ticker}' if name else ticker
    base = {
        "q": query,
        "from": _since(days).strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "searchIn": "title,description",
        "pageSize": min(limit, 100),
    }

    def search(params):
        def fetch():
            payload = get_json(
                NEWSAPI_URL, params=params, headers={"X-Api-Key": key}
            )
            return parse_newsapi(payload, ticker)

        return cached("newsapi", {"ticker": ticker, **params}, fetch,
                      ttl_hours=3)

    out = search({**base, "domains": ",".join(domains)}) if domains \
        else []
    if len(out) < MIN_DOMAIN_RESULTS:
        seen = {a["url"] for a in out}
        more = search({**base, "excludeDomains": ",".join(NOISE_DOMAINS)})
        out += [a for a in more if a["url"] not in seen]
    return out[:limit]


def parse_yahoo(items: list[dict], ticker: str) -> list[dict]:
    """Normalize yfinance news. Handles both the newer nested 'content'
    format and the older flat format."""
    out = []
    for item in items or []:
        c = item.get("content")
        if c:
            published = c.get("pubDate") or c.get("displayTime") or ""
            url = (c.get("canonicalUrl") or {}).get("url") or \
                (c.get("clickThroughUrl") or {}).get("url") or ""
            out.append(Article(
                title=c.get("title", ""),
                source=(c.get("provider") or {}).get("displayName", ""),
                published_at=published,
                url=url,
                text=c.get("summary") or c.get("description") or "",
                ticker=ticker,
                provider="yahoo",
            ).to_dict())
        elif item.get("title"):
            ts = item.get("providerPublishTime")
            published = (
                datetime.fromtimestamp(ts, timezone.utc).isoformat()
                if ts else ""
            )
            out.append(Article(
                title=item["title"],
                source=item.get("publisher", ""),
                published_at=published,
                url=item.get("link", ""),
                text=item.get("summary", ""),
                ticker=ticker,
                provider="yahoo",
            ).to_dict())
    return [a for a in out if a["title"]]


def fetch_yahoo(ticker: str, count: int = 20) -> list[dict]:
    """Yahoo Finance news for the ticker.

    Ticker.news has come back empty for every ticker since late Sep 2026
    (Yahoo's news endpoint errors; yfinance 1.4.1 and 1.7.0 both hit
    it), so when it is empty this falls back to Yahoo's search endpoint,
    keeping only items tagged with the ticker. Empty results are not
    cached, so the next call tries again.
    """
    def fetch():
        items = parse_yahoo(yf.Ticker(ticker).news, ticker)
        if not items:
            found = yf.Search(ticker, max_results=1,
                              news_count=count).news or []
            found = [i for i in found
                     if ticker in (i.get("relatedTickers") or [ticker])]
            items = parse_yahoo(found, ticker)
        return items

    return cached("yahoo_news", {"ticker": ticker}, fetch, ttl_hours=3,
                  cache_empty=False)


_COLUMN_ALIASES = {
    "title": ("title", "headline"),
    "published_at": ("published_at", "publishedat", "date", "datetime",
                     "time"),
    "text": ("text", "description", "summary", "content", "body"),
    "source": ("source", "publisher"),
    "url": ("url", "link"),
    "ticker": ("ticker", "stock", "symbol"),
}


def load_local_news(ticker: str, path=None) -> list[dict]:
    """Read articles from a local CSV (e.g. a Kaggle dataset).

    Needs a title/headline column and a date column. text, source, url,
    and ticker columns are optional. If there is a ticker column, rows
    are filtered on it; otherwise rows mentioning the ticker in the title
    or text are kept.
    """
    path = path or config.LOCAL_NEWS_CSV
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        lower = {h.lower().strip(): h for h in reader.fieldnames or []}
        cols = {}
        for field, aliases in _COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in lower:
                    cols[field] = lower[alias]
                    break
        if "title" not in cols:
            raise ValueError(f"{path} needs a title or headline column")
        out = []
        for row in reader:
            def get(field):
                return row.get(cols[field], "") if field in cols else ""

            if "ticker" in cols:
                if get("ticker").upper() != ticker:
                    continue
            elif ticker not in f"{get('title')} {get('text')}".upper():
                continue
            out.append(Article(
                title=get("title"),
                source=get("source") or "local",
                published_at=get("published_at"),
                url=get("url"),
                text=get("text"),
                ticker=ticker,
                provider="local",
            ).to_dict())
    return out


def get_news(ticker: str, days: int = 7, limit: int = 20,
             source: str = "auto") -> list[dict]:
    """Recent news articles about a company.

    Each article has title, source, published_at, url, text (summary or
    snippet), and provider. Use when you need recent events, sentiment,
    or catalysts for a stock.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        days: How far back to look (NewsAPI free plan caps at ~29 days).
        limit: Max articles to return.
        source: "auto" (NewsAPI + Yahoo, local CSV if both fail),
            "newsapi", "yahoo", or "local".
    """
    ticker = ticker.upper()
    if source == "newsapi":
        return fetch_newsapi(ticker, days, limit)[:limit]
    if source == "yahoo":
        return fetch_yahoo(ticker)[:limit]
    if source == "local":
        return load_local_news(ticker)[:limit]

    articles = []
    if config.NEWSAPI_KEY:
        try:
            articles += fetch_newsapi(ticker, days, limit)
        except Exception as err:  # noqa: BLE001 - fall through
            log.warning("NewsAPI failed for %s: %s", ticker, err)
    try:
        articles += fetch_yahoo(ticker)
    except Exception as err:  # noqa: BLE001 - fall through
        log.warning("Yahoo news failed for %s: %s", ticker, err)
    if not articles:
        articles = load_local_news(ticker)

    # Drop exact duplicate titles; newest first
    seen, unique = set(), []
    for a in articles:
        key = a["title"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(a)
    # Share the limit across providers (round robin, newest first within
    # each), so a feed with more recent timestamps can't crowd the other
    # out, then return the picks newest first.
    by_provider: dict[str, list] = {}
    for a in unique:
        by_provider.setdefault(a.get("provider", ""), []).append(a)
    queues = [
        sorted(q, key=lambda a: a["published_at"] or "", reverse=True)
        for q in by_provider.values()
    ]
    picked = []
    while len(picked) < limit and any(queues):
        for q in queues:
            if q and len(picked) < limit:
                picked.append(q.pop(0))
    picked.sort(key=lambda a: a["published_at"] or "", reverse=True)
    return picked
