"""Prices, company profile, and financial statements via yfinance.

No API key needed. Results are cached in data/cache/.
"""

import math

import yfinance as yf

from finagent.cache import cached

INFO_FIELDS = [
    "shortName", "longName", "sector", "industry", "country", "website",
    "currency", "marketCap", "enterpriseValue", "fullTimeEmployees",
    "trailingPE", "forwardPE", "priceToBook", "dividendYield", "beta",
    "profitMargins", "revenueGrowth", "earningsGrowth", "debtToEquity",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "recommendationKey",
    "targetMeanPrice", "numberOfAnalystOpinions", "longBusinessSummary",
]

# Keep the statements to the lines an analyst actually looks at, so the
# output fits comfortably in a prompt.
STATEMENT_LINES = {
    "income_statement": [
        "Total Revenue", "Gross Profit", "Operating Income",
        "Net Income", "Diluted EPS", "EBITDA",
    ],
    "balance_sheet": [
        "Total Assets", "Total Liabilities Net Minority Interest",
        "Stockholders Equity", "Cash And Cash Equivalents", "Total Debt",
    ],
    "cash_flow": [
        "Operating Cash Flow", "Capital Expenditure", "Free Cash Flow",
    ],
}


def _clean(value):
    """NaN/inf to None and numpy scalars to plain Python."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(f) or math.isinf(f):
        return None
    return int(f) if f.is_integer() and abs(f) > 1e3 else round(f, 4)


def get_price_history(ticker: str, period: str = "1y",
                      interval: str = "1d") -> list[dict]:
    """Daily (or other interval) OHLCV price history for a stock.

    Use when you need raw prices, e.g. to chart a stock or compute your
    own statistics. For a quick read on performance use
    get_price_summary instead.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        period: yfinance period: 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max.
        interval: 1d, 1wk, or 1mo.
    """
    ticker = ticker.upper()

    def fetch():
        df = yf.Ticker(ticker).history(
            period=period, interval=interval, auto_adjust=True
        )
        if df.empty:
            raise ValueError(f"No price data for {ticker}")
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else "Datetime"
        return [
            {
                "date": row[date_col].strftime("%Y-%m-%d"),
                "open": _clean(row["Open"]),
                "high": _clean(row["High"]),
                "low": _clean(row["Low"]),
                "close": _clean(row["Close"]),
                "volume": int(row["Volume"]),
            }
            for _, row in df.iterrows()
        ]

    params = {"ticker": ticker, "period": period, "interval": interval}
    return cached("prices", params, fetch, ttl_hours=12)


def summarize_prices(records: list[dict]) -> dict:
    """Return stats from daily price records (oldest first)."""
    closes = [r["close"] for r in records if r["close"] is not None]
    if len(closes) < 2:
        raise ValueError("Need at least two closes")

    def ret(days):
        if len(closes) <= days:
            return None
        return round(closes[-1] / closes[-1 - days] - 1, 4)

    logs = [math.log(b / a) for a, b in zip(closes, closes[1:])]
    mean = sum(logs) / len(logs)
    var = sum((x - mean) ** 2 for x in logs) / max(len(logs) - 1, 1)
    return {
        "as_of": records[-1]["date"],
        "last_close": closes[-1],
        "return_1m": ret(21),
        "return_3m": ret(63),
        "return_6m": ret(126),
        "return_1y": ret(251),
        "high_52w": max(closes[-252:]),
        "low_52w": min(closes[-252:]),
        "volatility_annualized": round(math.sqrt(var * 252), 4),
    }


def get_price_summary(ticker: str) -> dict:
    """Recent stock performance at a glance.

    Returns the last close, 1m/3m/6m/1y returns (as fractions, 0.05 =
    5%), 52-week high and low, and annualized volatility. Use this first
    when assessing how a stock has been trading.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
    """
    summary = summarize_prices(get_price_history(ticker, period="1y"))
    return {"ticker": ticker.upper(), **summary}


def get_company_info(ticker: str) -> dict:
    """Company profile and key valuation metrics.

    Returns name, sector, industry, market cap, P/E ratios, margins,
    growth, beta, analyst recommendation and target price, and a short
    business description. Use at the start of research to understand
    what the company does.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
    """
    ticker = ticker.upper()

    def fetch():
        info = yf.Ticker(ticker).info or {}
        if not info.get("shortName") and not info.get("longName"):
            raise ValueError(f"No company info for {ticker}")
        out = {k: _clean(info.get(k)) for k in INFO_FIELDS}
        summary = out.get("longBusinessSummary") or ""
        out["longBusinessSummary"] = summary[:1200]
        out["ticker"] = ticker
        return out

    return cached("info", {"ticker": ticker}, fetch, ttl_hours=24)


def _statement_to_dict(df, lines: list[str], periods: int) -> dict:
    if df is None or df.empty:
        return {}
    out = {}
    for col in list(df.columns)[:periods]:
        label = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") \
            else str(col)
        out[label] = {
            line: _clean(df.at[line, col]) if line in df.index else None
            for line in lines
        }
    return out


def get_financials(ticker: str, quarterly: bool = False,
                   periods: int = 4) -> dict:
    """Key lines from the income statement, balance sheet, and cash flow.

    Covers revenue, gross profit, operating income, net income, EPS,
    EBITDA, assets, liabilities, equity, cash, debt, operating cash flow,
    capex, and free cash flow for the most recent periods. Use when you
    need fundamentals or growth trends.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        quarterly: True for quarterly statements, False for annual.
        periods: How many recent periods to return (max about 4-5).
    """
    ticker = ticker.upper()

    def fetch():
        t = yf.Ticker(ticker)
        if quarterly:
            frames = {
                "income_statement": t.quarterly_income_stmt,
                "balance_sheet": t.quarterly_balance_sheet,
                "cash_flow": t.quarterly_cashflow,
            }
        else:
            frames = {
                "income_statement": t.income_stmt,
                "balance_sheet": t.balance_sheet,
                "cash_flow": t.cashflow,
            }
        out = {
            name: _statement_to_dict(df, STATEMENT_LINES[name], periods)
            for name, df in frames.items()
        }
        if not any(out.values()):
            raise ValueError(f"No financial statements for {ticker}")
        return out

    params = {"ticker": ticker, "quarterly": quarterly, "periods": periods}
    data = cached("financials", params, fetch, ttl_hours=24)
    return {"ticker": ticker, "quarterly": quarterly, **data}
