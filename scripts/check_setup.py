"""Check that API keys are set and every data source responds.

Run from the repo root after filling in .env:

    python scripts/check_setup.py

Each check makes one small live call (results are cached in data/).
The Anthropic check costs a fraction of a cent.
"""

import sys

from finagent import config

TICKER = "AAPL"


def check_anthropic():
    from finagent import llm

    reply = llm.complete("Reply with the single word: ok", max_tokens=5)
    return f"model {config.LLM_MODEL} replied {reply.strip()!r}"


def check_yfinance():
    from finagent.tools import market_data

    s = market_data.get_price_summary(TICKER)
    return f"{TICKER} last close {s['last_close']} on {s['as_of']}"


def check_newsapi():
    from finagent.tools import news

    arts = news.get_news(TICKER, days=3, limit=5, source="newsapi")
    return f"{len(arts)} articles"


def check_yahoo_news():
    from finagent.tools import news

    arts = news.get_news(TICKER, limit=5, source="yahoo")
    return f"{len(arts)} articles"


def check_fred():
    from finagent.tools import macro

    s = macro.get_series("UNRATE")
    last = s["observations"][-1]
    return f"UNRATE {last['value']} on {last['date']}"


def check_sec():
    from finagent.tools import filings

    f = filings.get_recent_filings(TICKER, form="10-K", limit=1)
    return f"latest 10-K filed {f[0]['filing_date']}"


def check_alpha_vantage():
    from finagent.tools import earnings

    e = earnings.get_earnings(TICKER, quarters=1)
    return f"last quarter EPS {e[0]['reported_eps']}"


CHECKS = [
    ("Anthropic API", "ANTHROPIC_API_KEY", check_anthropic),
    ("Yahoo Finance prices", None, check_yfinance),
    ("Yahoo Finance news", None, check_yahoo_news),
    ("NewsAPI", "NEWSAPI_KEY", check_newsapi),
    ("FRED", "FRED_API_KEY", check_fred),
    ("SEC EDGAR", "SEC_USER_AGENT", check_sec),
    ("Alpha Vantage", "ALPHAVANTAGE_API_KEY", check_alpha_vantage),
]


def main() -> int:
    env = config.ROOT / ".env"
    print(f".env found: {env.exists()}  ({env})")
    print(f"Cache mode: {config.CACHE_MODE}\n")
    failures = 0
    for label, key, fn in CHECKS:
        if key and not getattr(config, key):
            print(f"[SKIP] {label}: {key} not set in .env")
            failures += 1
            continue
        try:
            print(f"[ OK ] {label}: {fn()}")
        except Exception as err:  # noqa: BLE001
            print(f"[FAIL] {label}: {type(err).__name__}: {err}")
            failures += 1
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
