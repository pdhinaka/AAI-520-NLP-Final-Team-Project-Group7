"""Run routing and the evaluator-optimizer for a ticker and print each
step.

Run from the repo root:

    python scripts/run_workflows.py AAPL
    python scripts/run_workflows.py AAPL --offline   # cache only

Routing runs the prompt chain first for the news items. Both results
are saved to data/runs/. LLM replies are cached in data/cache/llm/, so
--offline replays an earlier run exactly.
"""

import argparse
import logging

from finagent import config
from finagent.workflows import evaluator_optimizer as eo
from finagent.workflows import routing


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ticker")
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--offline", action="store_true",
                        help="read only from the cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    for name in ("httpx", "yfinance", "urllib3", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if args.offline:
        config.CACHE_MODE = "offline"

    r = routing.run_routing(args.ticker)
    print(f"\n== routing {r['ticker']} with {r['model']}")
    for d in r["decisions"]:
        print(f"  {d['item_id']:<14} -> {d['specialist']:<8} "
              f"({d['method']}) {d['reason'][:60]}")
    for e in r["errors"]:
        print(f"  skipped {e['kind']}: {e['error'][:70]}")
    print(f"counts: {r['counts']}")
    for name, a in r["analyses"].items():
        print(f"\n[{name}] {a['signal']} ({a['confidence']}), "
              f"{a['n_items']} items\n  {a['summary']}")
        for p in a["key_points"]:
            print(f"  - {p}")

    loop = eo.run_loop(eo.context_from_routing(r),
                       max_iters=args.max_iters, threshold=args.threshold)
    print(f"\n== evaluator-optimizer (threshold {loop['threshold']}, "
          f"min criterion {loop['min_score']})")
    for h in loop["history"]:
        scores = " ".join(f"{k}={v:g}" for k, v in h["scores"].items())
        print(f"  iter {h['iteration']}: overall {h['overall']:.3f} "
              f"passed={h['passed']}  {scores}")
        for f in h["feedback"][:3]:
            print(f"    feedback: {f[:90]}")
    print(f"\nbest draft: iteration {loop['best_iteration']} "
          f"(passed: {loop['passed']})\n\n{loop['final']}")
    print(f"\nusage: routing {r['usage']}, loop {loop['usage']}")
    print(f"saved to: {r['saved_to']}\n          {loop['saved_to']}")


if __name__ == "__main__":
    main()
