"""Routing: send each item to the right specialist analyzer.

Owner: Workstream C."""



def route(item: dict):
    """Return the specialist name and the reason for the choice."""
    raise NotImplementedError


def earnings_analyzer(item: dict):
    """Analyze earnings content."""
    raise NotImplementedError


def news_analyzer(item: dict):
    """Analyze general news content."""
    raise NotImplementedError


def market_analyzer(item: dict):
    """Analyze price and market data."""
    raise NotImplementedError
