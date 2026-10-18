"""Run the research agent and all three workflows for a ticker.

Run from the repo root:

    python scripts/run_all.py AAPL
    python scripts/run_all.py AAPL --offline   # cache only

The agent plans, calls tools, writes and reflects on a report, and
saves notes to memory/. Then the prompt chain (replayed from the
agent's news tool), routing, and the evaluator-optimizer run, and the
two recommendations are compared. Each stage is saved to data/runs/.
Set AS_OF_DATE=YYYY-MM-DD to replay a run from an earlier day.
"""

import argparse
import logging

from finagent import config, pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ticker")
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--memory-dir", default=None,
                        help="notes folder (default: memory/)")
    parser.add_argument("--offline", action="store_true",
                        help="read only from the cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    for name in ("httpx", "yfinance", "urllib3", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if args.offline:
        config.CACHE_MODE = "offline"

    out = pipeline.run_all(args.ticker, memory_dir=args.memory_dir,
                           threshold=args.threshold,
                           max_iters=args.max_iters)
    a = out["agent"]
    print(f"\n== agent {a['ticker']} ({a['plan']['source']} plan, "
          f"{a['turns']} turns, stopped: {a['stopped']})")
    for i, step in enumerate(a["plan"]["steps"], 1):
        print(f"  {i}. {step}")
    for t in a["trace"]:
        status = "ok" if t["ok"] else f"error: {t['error'][:60]}"
        print(f"  turn {t['turn']}: {t['tool']} {status}")
    for r in a["reflection_history"]:
        print(f"  draft {r['draft']}: score {r['score']} "
              f"grounding {(r['grounding'] or {}).get('ratio')}")
    print(f"recommendation: {a['recommendation']}")
    print(f"notes saved: {len(a['notes_saved'])}")

    loop = out["loop"]
    print(f"\n== evaluator-optimizer: best iteration "
          f"{loop['best_iteration']} (passed: {loop['passed']})")
    print(f"\n== comparison\n{out['comparison']}")
    for stage in ("agent", "chain", "routing", "loop"):
        print(f"{stage}: usage {out[stage]['usage']}, "
              f"saved to {out[stage]['saved_to']}")


if __name__ == "__main__":
    main()
