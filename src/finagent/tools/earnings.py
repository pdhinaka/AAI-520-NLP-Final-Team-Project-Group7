"""Quarterly earnings (reported vs. estimated EPS) from Alpha Vantage.

Needs ALPHAVANTAGE_API_KEY. The free tier allows about 25 requests a
day, so results are cached for 24 hours.
"""

from finagent import config
from finagent.cache import cached
from finagent.tools._http import get_json

AV_URL = "https://www.alphavantage.co/query"


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_earnings(payload: dict, quarters: int) -> list[dict]:
    for key in ("Note", "Information", "Error Message"):
        if key in payload:
            raise RuntimeError(f"Alpha Vantage: {payload[key]}")
    out = []
    for q in payload.get("quarterlyEarnings", [])[:quarters]:
        out.append({
            "fiscal_date_ending": q.get("fiscalDateEnding"),
            "reported_date": q.get("reportedDate"),
            "reported_eps": _num(q.get("reportedEPS")),
            "estimated_eps": _num(q.get("estimatedEPS")),
            "surprise": _num(q.get("surprise")),
            "surprise_pct": _num(q.get("surprisePercentage")),
        })
    return out


def get_earnings(ticker: str, quarters: int = 8) -> list[dict]:
    """Recent quarterly earnings: reported EPS vs. analyst estimate.

    Each quarter has fiscal_date_ending, reported_date, reported_eps,
    estimated_eps, surprise, and surprise_pct. Use to judge whether a
    company has been beating or missing expectations.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        quarters: Number of recent quarters to return.
    """
    key = config.require("ALPHAVANTAGE_API_KEY")
    ticker = ticker.upper()

    def fetch():
        payload = get_json(AV_URL, params={
            "function": "EARNINGS", "symbol": ticker, "apikey": key,
        })
        return parse_earnings(payload, quarters=40)

    data = cached("earnings", {"ticker": ticker}, fetch, ttl_hours=24)
    return data[:quarters]
