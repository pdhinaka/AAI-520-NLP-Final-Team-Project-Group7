"""Copy the replay snapshot in demo/data/ into data/.

The snapshot holds the cached API and LLM replies and the saved runs
from our live AAPL run on 2026-10-18, so a fresh clone can execute the
notebook with CACHE_MODE=offline and no real API keys. Run from the
repo root:

    python scripts/restore_demo_data.py

Existing files in data/ are never overwritten.
"""

import shutil

from finagent import config

SNAPSHOT = config.ROOT / "demo" / "data"


def main():
    copied = skipped = 0
    for src in sorted(SNAPSHOT.rglob("*.json")):
        dest = config.DATA_DIR / src.relative_to(SNAPSHOT)
        if dest.exists():
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    print(f"Copied {copied} files into {config.DATA_DIR} "
          f"({skipped} already there).")


if __name__ == "__main__":
    main()
