"""Settings and API key loading.

Keys live in a `.env` file at the repo root (git-ignored). Copy
`.env.example` to `.env` and fill it in. The README lists where to get
each key.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MEMORY_DIR = ROOT / "memory"
LOCAL_NEWS_CSV = DATA_DIR / "local_news.csv"

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Only needed if your key is not scoped to a workspace (the API error
# will say so). Find the ID in the Claude Console under Workspaces.
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5")

# Data sources
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")

# "normal": use the cache when fresh, otherwise fetch.
# "offline": never touch the network, only read the cache. Use this for
# the final notebook run so results don't shift between runs.
CACHE_MODE = os.getenv("CACHE_MODE", "normal").lower()

HTTP_TIMEOUT = 20


class MissingKeyError(RuntimeError):
    """A tool needs a setting that isn't in .env."""


def require(name: str) -> str:
    """Return a setting's value, or raise a clear error if it is empty."""
    value = globals().get(name, "")
    if not value:
        raise MissingKeyError(
            f"{name} is not set. Add it to .env at the repo root "
            "(see .env.example and the Setup section of the README)."
        )
    return value
