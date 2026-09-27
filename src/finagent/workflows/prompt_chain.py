"""Prompt chaining: ingest -> preprocess -> classify -> extract -> summarize.

Owner: Workstream A."""



def ingest(ticker: str):
    """Fetch raw news articles."""
    raise NotImplementedError


def preprocess(articles: list[dict]):
    """Clean, dedupe, and trim article text."""
    raise NotImplementedError


def classify(articles: list[dict]):
    """Label each article by topic and sentiment."""
    raise NotImplementedError


def extract(articles: list[dict]):
    """Pull entities, figures, dates, and events."""
    raise NotImplementedError


def summarize(extracted: list[dict]):
    """Write a short news digest."""
    raise NotImplementedError


def run_chain(ticker: str):
    """Run every step and return the digest plus intermediate outputs."""
    raise NotImplementedError
