"""Run the prompt chain for a ticker and print what each step produced.

Run from the repo root:

    python scripts/run_chain.py AAPL
    python scripts/run_chain.py AAPL --offline   # cache only, no network

The full result (every step's output) is saved to data/runs/. LLM
replies are cached in data/cache/llm/, so a second run on the same news
costs nothing, and --offline replays it exactly.
"""

import argparse
import logging

from finagent import config
from finagent.workflows import prompt_chain as pc


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ticker")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offline", action="store_true",
                        help="read only from the cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    for name in ("httpx", "yfinance", "urllib3", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if args.offline:
        config.CACHE_MODE = "offline"

    r = pc.run_chain(args.ticker, days=args.days, limit=args.limit)
    pre = r["steps"]["preprocess"]

    print(f"\n== {r['ticker']} with {r['model']}")
    print(f"ingest: {len(r['steps']['ingest'])} articles; preprocess "
          f"kept {len(pre['articles'])} (company: {pre['company']!r})")
    for d in pre["dropped"]:
        print(f"  dropped ({d['reason']}): {d['title'][:70]}")

    print("\nclassify:")
    for a in r["steps"]["classify"]:
        lab = a["labels"]
        flag = "" if lab["relevant"] else "  [not relevant]"
        print(f"  [{a['id']}] {lab['topic']:<14} {lab['sentiment']:<9}"
              f"{a['title'][:55]}{flag}")

    print("\nextract:")
    for e in r["steps"]["extract"]:
        figs = ", ".join(f.get("value", "") if isinstance(f, dict)
                         else str(f) for f in e["figures"][:3])
        print(f"  [{e['id']}] {e['event'][:80]}")
        if figs:
            print(f"       figures: {figs}")

    d = r["digest"]
    print(f"\ndigest: {d['headline']}\n{d['summary']}")
    print(f"catalysts: {d['catalysts']}\nrisks: {d['risks']}")
    print(f"sentiment: {d['sentiment_counts']}  topics: {d['topic_counts']}")
    if d["unknown_citations"]:
        print(f"WARNING: summary cites unknown ids {d['unknown_citations']}")
    print(f"\ntimings (s): {r['timings']}")
    print(f"usage: {r['usage']}")
    print(f"saved to: {r['saved_to']}")


if __name__ == "__main__":
    main()
