"""Settings and API key loading."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MEMORY_DIR = ROOT / "memory"

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
