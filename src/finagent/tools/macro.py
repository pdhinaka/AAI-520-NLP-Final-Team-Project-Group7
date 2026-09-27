"""Economic indicators from FRED (Federal Reserve Bank of St. Louis).

Needs FRED_API_KEY.
"""

from datetime import date, timedelta

from finagent import config
from finagent.cache import cached
from finagent.tools._http import get_json

FRED_URL = "https://api.stlouisfed.org/fred"

# series_id -> (label, how to report the change over a year)
SNAPSHOT_SERIES = {
    "FEDFUNDS": ("Effective federal funds rate (%)", "diff"),
    "DGS10": ("10-year Treasury yield (%)", "diff"),
    "CPIAUCSL": ("CPI, all urban consumers (index)", "pct"),
    "UNRATE": ("Unemployment rate (%)", "diff"),
    "VIXCLS": ("VIX volatility index", "diff"),
}


def parse_observations(payload: dict) -> list[dict]:
    out = []
    for obs in payload.get("observations", []):
        raw = obs.get("value", ".")
        value = None if raw in (".", "") else float(raw)
        out.append({"date": obs["date"], "value": value})
    return out


def get_series(series_id: str, start: str | None = None) -> dict:
    """Observations for one FRED economic data series.

    Common series: FEDFUNDS (fed funds rate), DGS10 (10y Treasury),
    CPIAUCSL (CPI), UNRATE (unemployment), GDPC1 (real GDP), VIXCLS
    (VIX). Use when you need the history of a specific indicator.

    Args:
        series_id: FRED series id, e.g. "UNRATE".
        start: Start date YYYY-MM-DD (default: about 2 years ago).
    """
    key = config.require("FRED_API_KEY")
    series_id = series_id.upper()
    start = start or (date.today() - timedelta(days=730)).isoformat()
    base = {"series_id": series_id, "api_key": key, "file_type": "json"}

    def fetch():
        meta = get_json(f"{FRED_URL}/series", params=base)
        info = (meta.get("seriess") or [{}])[0]
        obs = get_json(
            f"{FRED_URL}/series/observations",
            params={**base, "observation_start": start},
        )
        return {
            "series_id": series_id,
            "title": info.get("title", ""),
            "units": info.get("units", ""),
            "frequency": info.get("frequency", ""),
            "observations": parse_observations(obs),
        }

    return cached("fred", {"series_id": series_id, "start": start},
                  fetch, ttl_hours=24)


def latest_vs_year_ago(observations: list[dict], mode: str) -> dict:
    """Latest value and change versus about a year earlier."""
    obs = [o for o in observations if o["value"] is not None]
    if not obs:
        return {"latest": None}
    latest = obs[-1]
    target = date.fromisoformat(latest["date"]) - timedelta(days=365)
    prior = min(
        obs, key=lambda o: abs(date.fromisoformat(o["date"]) - target)
    )
    if mode == "pct":
        change = round((latest["value"] / prior["value"] - 1) * 100, 2)
        key = "yoy_pct_change"
    else:
        change = round(latest["value"] - prior["value"], 2)
        key = "change_vs_year_ago"
    return {
        "latest": latest["value"],
        "as_of": latest["date"],
        "year_ago": prior["value"],
        key: change,
    }


def get_macro_snapshot() -> dict:
    """Current macroeconomic backdrop in one call.

    Latest fed funds rate, 10-year Treasury yield, CPI inflation (year
    over year %), unemployment rate, and VIX, each with the change versus
    a year ago. Use when the analysis needs interest-rate, inflation, or
    market-risk context.
    """
    out = {}
    for series_id, (label, mode) in SNAPSHOT_SERIES.items():
        series = get_series(series_id)
        out[series_id] = {
            "label": label,
            **latest_vs_year_ago(series["observations"], mode),
        }
    return out
