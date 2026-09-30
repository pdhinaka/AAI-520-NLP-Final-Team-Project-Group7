"""Shared data shapes passed between tools, workflows, and the agent.

Keep these small. Tools return plain dicts/lists so results are easy to
cache and to hand to the LLM as JSON; Article is the one shape every
workflow agrees on.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class Article:
    title: str
    source: str
    published_at: str  # ISO 8601
    url: str = ""
    text: str = ""  # description + whatever body text the source gives
    ticker: str = ""
    provider: str = ""  # newsapi, yahoo, local
    # Filled in later by the prompt chain
    labels: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NewsDigest:
    """Output of the prompt chain (Workstream A), also returned by the
    get_news_digest tool. Proposed shape; the team can still change it.

    summary cites articles as [id]; sources maps those ids to titles and
    URLs. Counts are computed in Python from the classify step, not
    written by the model.
    """

    ticker: str
    n_articles: int = 0
    headline: str = ""
    summary: str = ""  # markdown, cites articles as [id]
    catalysts: list = field(default_factory=list)
    risks: list = field(default_factory=list)
    sentiment_counts: dict = field(default_factory=dict)
    topic_counts: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)  # [{id, title, url}]
    unknown_citations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
