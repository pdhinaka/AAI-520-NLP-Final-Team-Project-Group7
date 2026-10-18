"""Run the research agent and all three workflows for a ticker.

Run from the repo root:

    python scripts/run_all.py AAPL
    python scripts/run_all.py AAPL --offline   # cache only

The agent plans, calls tools, writes and reflects on a report, and
saves notes to memory/. Then the prompt chain, routing, and the
evaluator-optimizer run, and the two recommendations are compared.
Prints a short summary of each stage; --verbose adds the step logs.
Live runs are saved to data/runs/. Set AS_OF_DATE=YYYY-MM-DD to replay
a run from an earlier day.
"""

import argparse
import logging

from finagent import config, llm, pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ticker")
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--memory-dir", default=None,
                        help="notes folder (default: memory/)")
    parser.add_argument("--offline", action="store_true",
                        help="read only from the cache")
    parser.add_argument("--verbose", action="store_true",
                        help="also print each step's log lines")
    args = parser.parse_args()

    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s: %(message)s")
    for name in ("httpx", "yfinance", "urllib3", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if args.offline:
        config.CACHE_MODE = "offline"

    out = pipeline.run_all(args.ticker, memory_dir=args.memory_dir,
                           threshold=args.threshold,
                           max_iters=args.max_iters,
                           save=not args.offline)
    print(pipeline.format_summary(out))
    calls = llm.USAGE["calls"]
    print(f"\nLive LLM calls this run: {calls}"
          + (" (everything replayed from the cache)" if not calls else
             f" ({llm.USAGE['input_tokens']:,} input, "
             f"{llm.USAGE['output_tokens']:,} output tokens)"))
    if not args.offline:
        print(f"Agent run saved to {out['agent']['saved_to']}")


if __name__ == "__main__":
    main()
