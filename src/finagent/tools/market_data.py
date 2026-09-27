"""Price history, financials, and company info via yfinance."""



def get_price_history(ticker: str, period: str = "1y"):
    """Daily OHLCV for a ticker."""
    raise NotImplementedError


def get_financials(ticker: str):
    """Income statement, balance sheet, and cash flow."""
    raise NotImplementedError


def get_company_info(ticker: str):
    """Sector, industry, market cap, and description."""
    raise NotImplementedError
