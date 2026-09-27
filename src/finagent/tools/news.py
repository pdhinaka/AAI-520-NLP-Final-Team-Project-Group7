"""Financial news for a ticker (NewsAPI, Yahoo, or Kaggle)."""



def get_news(ticker: str, days: int = 7):
    """Recent articles as a list of dicts (title, source, date, text)."""
    raise NotImplementedError
