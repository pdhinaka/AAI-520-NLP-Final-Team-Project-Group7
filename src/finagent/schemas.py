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
