"""Routing: send each piece of content to the right specialist analyzer.

Workflow Pattern 2. Content for a ticker (news articles from the prompt
chain plus structured tool output: earnings, financials, prices,
valuation, macro) is turned into items, each item is routed to one
specialist, and each specialist analyzes only what it was sent:

    from finagent.workflows import routing

    result = routing.run_routing("AAPL")
    result["decisions"]            # one {item_id, specialist, reason}
    result["analyses"]["market"]   # the market specialist's view

The router works in two stages. Structured items and articles whose
classify label maps cleanly to a specialist are routed by rule (free,
deterministic, easy to explain). Only articles the chain labeled
"other" go to the LLM router, in one batched call. Every decision
records the method and a reason, so the notebook can show the routing.

Owner: Workstream C.
"""

import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone

from finagent import config, llm
from finagent.tools import earnings, macro, market_data
from finagent.workflows import prompt_chain

log = logging.getLogger(__name__)

SPECIALISTS = ("earnings", "news", "market")

# Structured tool output: the kind of data decides the specialist.
KIND_ROUTES = {
    "earnings": "earnings",
    "financials": "earnings",
    "price_summary": "market",
    "company_info": "market",
    "macro": "market",
}
# Topic labels from the prompt chain's classify step. "other" is left
# out on purpose: those articles need the LLM router.
TOPIC_ROUTES = {
    "earnings": "earnings",
    "analyst_rating": "market",
    "macro": "market",
    "product": "news",
    "legal": "news",
}
DEFAULT_SPECIALIST = "news"

SIGNALS = ("bullish", "neutral", "bearish")
CONFIDENCE = ("low", "medium", "high")

ARTICLE_CHARS = 400  # article text sent to a specialist, per article
INFO_FIELDS = (
    "shortName", "sector", "industry", "marketCap", "trailingPE",
    "forwardPE", "priceToBook", "beta", "profitMargins", "revenueGrowth",
    "recommendationKey", "targetMeanPrice", "numberOfAnalystOpinions",
)


# ----------------------------------------------------------------- items

def items_from_articles(articles: list[dict],
                        extracted: list[dict] | None = None) -> list[dict]:
    """Turn prompt chain articles into routable items.

    articles is the chain's classify output (articles with labels).
    Articles the chain marked not relevant are skipped. If the extract
    output is passed, each item also gets its event and figures.
    """
    facts = {e["id"]: e for e in extracted or []}
    items = []
    for a in articles:
        labels = a.get("labels") or {}
        if labels.get("relevant") is False:
            continue
        fact = facts.get(a.get("id"), {})
        items.append({
            "id": f"news-{a.get('id', len(items) + 1)}",
            "kind": "news",
            "title": a.get("title", ""),
            "text": prompt_chain.trim(a.get("text", ""), ARTICLE_CHARS),
            "source": a.get("source", ""),
            "published_at": a.get("published_at", ""),
            "topic": labels.get("topic", ""),
            "sentiment": labels.get("sentiment", "neutral"),
            "event": fact.get("event", ""),
            "figures": fact.get("figures", []),
        })
    return items


def _tool_calls(ticker: str) -> dict:
    return {
        "price_summary": lambda: market_data.get_price_summary(ticker),
        "company_info": lambda: market_data.get_company_info(ticker),
        "financials": lambda: market_data.get_financials(ticker),
        "earnings": lambda: earnings.get_earnings(ticker, quarters=8),
        "macro": macro.get_macro_snapshot,
    }


def items_from_tools(ticker: str,
                     errors: list | None = None) -> list[dict]:
    """Fetch structured data for the ticker, one item per tool.

    A tool that fails (missing key, rate limit, no data) is skipped and
    noted in errors, so routing still runs on whatever is available.
    """
    errors = errors if errors is not None else []
    items = []
    for kind, fetch in _tool_calls(ticker.upper()).items():
        try:
            data = fetch()
        except Exception as err:  # noqa: BLE001 - one tool can fail
            log.warning("%s failed for %s: %s", kind, ticker, err)
            errors.append({"kind": kind,
                           "error": f"{type(err).__name__}: {err}"})
            continue
        items.append({"id": kind, "kind": kind,
                      "title": f"{ticker.upper()} {kind}", "data": data})
    return items


# --------------------------------------------------------------- routing

ROUTER_SYSTEM = (
    "You are the dispatcher for an equity research team. You send each "
    "piece of content to the one specialist best placed to analyze it."
)

SPECIALIST_GUIDE = (
    "earnings: quarterly results, EPS, revenue, guidance, margins, "
    "financial statements\n"
    "news: company events such as products, legal or regulatory "
    "action, management, partnerships, security incidents\n"
    "market: stock price and trading, valuation, analyst ratings and "
    "price targets, interest rates, macro and sector moves"
)


def _rule(item: dict) -> dict | None:
    """Route by rule, or return None if the item needs the LLM."""
    kind = item.get("kind", "")
    if kind in KIND_ROUTES:
        return {"specialist": KIND_ROUTES[kind], "method": "rule",
                "reason": f"structured {kind} data"}
    topic = item.get("topic", "")
    if topic in TOPIC_ROUTES:
        return {"specialist": TOPIC_ROUTES[topic], "method": "rule",
                "reason": f"classified as {topic} by the prompt chain"}
    return None


def _llm_route(items: list[dict], ticker: str) -> dict:
    """One batched LLM call for every item the rules couldn't place.

    Returns {item_id: {specialist, reason}}. Answers naming an unknown
    specialist are dropped here and get the default later.
    """
    lines = [f"[{it['id']}] {it['title']}\n{it.get('text', '')[:300]}"
             for it in items]
    prompt = (
        f"Route each item about {ticker} to one specialist.\n\n"
        f"Specialists:\n{SPECIALIST_GUIDE}\n\n"
        'Return a JSON list: [{"id": "news-3", "specialist": "...", '
        '"reason": "one short sentence"}]\n\n'
        "Items:\n\n" + "\n\n".join(lines)
    )
    reply = llm.complete_json_cached(prompt, system=ROUTER_SYSTEM,
                                     max_tokens=1024)
    if isinstance(reply, dict):
        reply = next(
            (v for v in reply.values() if isinstance(v, list)), []
        )
    out = {}
    for r in reply or []:
        if not isinstance(r, dict):
            continue
        if r.get("specialist") in SPECIALISTS and r.get("id"):
            out[str(r["id"])] = {
                "specialist": r["specialist"],
                "reason": str(r.get("reason", "")),
            }
    return out


def route_items(items: list[dict], ticker: str = "",
                use_llm: bool = True) -> list[dict]:
    """Route every item. Returns one decision per item, in order.

    Each decision is {item_id, kind, title, specialist, method,
    reason}; method is "rule", "llm", or "default" (the LLM was off or
    gave no usable answer).
    """
    decisions = {}
    pending = []
    for it in items:
        rule = _rule(it)
        if rule:
            decisions[it["id"]] = rule
        else:
            pending.append(it)

    answers = _llm_route(pending, ticker) if pending and use_llm else {}
    for it in pending:
        got = answers.get(it["id"])
        if got:
            decisions[it["id"]] = {**got, "method": "llm"}
        else:
            decisions[it["id"]] = {
                "specialist": DEFAULT_SPECIALIST,
                "method": "default",
                "reason": "no confident route; sent to the generalist "
                          "news analyzer",
            }

    out = [{"item_id": it["id"], "kind": it["kind"],
            "title": it.get("title", ""), **decisions[it["id"]]}
           for it in items]
    for d in out:
        log.info("route %-14s -> %-8s (%s) %s", d["item_id"],
                 d["specialist"], d["method"], d["reason"])
    return out


def route(item: dict, ticker: str = "", use_llm: bool = True) -> dict:
    """Route a single item; see route_items."""
    return route_items([item], ticker, use_llm=use_llm)[0]


# ------------------------------------------------------ facts per analyst
# Numbers are computed here in Python, not by the model. The specialist
# interprets them; the evaluator-optimizer later checks the final brief
# against these same facts.

def _article_fact(it: dict) -> dict:
    fact = {"id": it["id"], "title": it["title"],
            "source": it.get("source", ""),
            "date": (it.get("published_at") or "")[:10],
            "topic": it.get("topic", ""),
            "sentiment": it.get("sentiment", "")}
    if it.get("event"):
        fact["event"] = it["event"]
    if it.get("figures"):
        fact["figures"] = it["figures"]
    else:
        fact["text"] = it.get("text", "")
    return fact


def _pct_change(new, old):
    if new is None or not old:
        return None
    return round(new / old - 1, 4)


def _eps_facts(quarters: list[dict]) -> dict:
    reported = [q for q in quarters if q.get("reported_eps") is not None]
    if not reported:
        return {}
    with_est = [q for q in reported if q.get("surprise") is not None]
    beats = sum(1 for q in with_est if q["surprise"] > 0)
    surprises = [q["surprise_pct"] for q in with_est
                 if q.get("surprise_pct") is not None]
    return {
        "quarters": len(reported),
        "beats": beats,
        "beat_rate": round(beats / len(with_est), 2) if with_est else None,
        "avg_surprise_pct": (round(sum(surprises) / len(surprises), 2)
                             if surprises else None),
        "latest": reported[0],
        "year_ago_eps": (reported[4]["reported_eps"]
                         if len(reported) > 4 else None),
    }


def _fundamental_facts(data: dict) -> dict:
    income = data.get("income_statement") or {}
    cash = data.get("cash_flow") or {}
    periods = sorted(income, reverse=True)
    if not periods:
        return {}
    latest = income[periods[0]]
    prior = income[periods[1]] if len(periods) > 1 else {}
    revenue = latest.get("Total Revenue")
    net = latest.get("Net Income")
    return {
        "period": periods[0],
        "revenue": revenue,
        "revenue_growth": _pct_change(revenue,
                                      prior.get("Total Revenue")),
        "net_income": net,
        "net_margin": round(net / revenue, 4) if net and revenue else None,
        "operating_income": latest.get("Operating Income"),
        "diluted_eps": latest.get("Diluted EPS"),
        "free_cash_flow": (cash.get(periods[0]) or {}).get(
            "Free Cash Flow"),
    }


def earnings_facts(items: list[dict]) -> dict:
    """EPS beat record, latest fundamentals, and earnings news."""
    facts: dict = {}
    for it in items:
        if it["kind"] == "earnings":
            facts["eps"] = _eps_facts(it["data"])
        elif it["kind"] == "financials":
            facts["fundamentals"] = _fundamental_facts(it["data"])
        else:
            facts.setdefault("articles", []).append(_article_fact(it))
    return facts


def market_facts(items: list[dict]) -> dict:
    """Price action, valuation, analyst view, macro backdrop, news."""
    facts: dict = {}
    for it in items:
        data = it.get("data")
        if it["kind"] == "price_summary":
            facts["price"] = dict(data)
            hi, lo = data.get("high_52w"), data.get("low_52w")
            last = data.get("last_close")
            if hi and lo and last is not None and hi > lo:
                facts["price"]["position_in_52w_range"] = round(
                    (last - lo) / (hi - lo), 2)
        elif it["kind"] == "company_info":
            facts["valuation"] = {k: data.get(k) for k in INFO_FIELDS}
        elif it["kind"] == "macro":
            facts["macro"] = {
                v.get("label", k): {kk: vv for kk, vv in v.items()
                                    if kk != "label"}
                for k, v in data.items()
            }
        else:
            facts.setdefault("articles", []).append(_article_fact(it))
    last = (facts.get("price") or {}).get("last_close")
    target = (facts.get("valuation") or {}).get("targetMeanPrice")
    if last and target:
        facts["valuation"]["upside_to_target"] = _pct_change(target, last)
    return facts


def news_facts(items: list[dict]) -> dict:
    """Company-event articles with sentiment and topic counts."""
    articles = [_article_fact(it) for it in items]
    return {
        "articles": articles,
        "sentiment_counts": dict(Counter(a["sentiment"]
                                         for a in articles)),
        "topic_counts": dict(Counter(a["topic"] for a in articles)),
    }


# ----------------------------------------------------------- specialists

SPECIALIST_SYSTEMS = {
    "earnings": (
        "You are an earnings and fundamentals analyst on an equity "
        "research team. You judge the quality and trend of a company's "
        "reported results. Use only the facts provided."
    ),
    "news": (
        "You are a company news analyst on an equity research team. You "
        "judge how recent company events affect the investment case. "
        "Use only the facts provided."
    ),
    "market": (
        "You are a market and valuation analyst on an equity research "
        "team. You judge price action, valuation, analyst sentiment, "
        "and the macro backdrop. Use only the facts provided."
    ),
}

SPECIALIST_FOCUS = {
    "earnings": "Has the company been beating expectations, and are "
                "revenue, margins, and cash flow improving or "
                "deteriorating?",
    "news": "Which recent events matter most for shareholders, and do "
            "they help or hurt the investment case?",
    "market": "Is the stock's momentum, valuation, and analyst view "
              "supportive, and does the macro backdrop help or hurt?",
}

FACT_BUILDERS = {
    "earnings": earnings_facts,
    "news": news_facts,
    "market": market_facts,
}


def _analyze(specialist: str, items: list[dict], ticker: str) -> dict:
    """Shared body of the three specialists: build facts, ask the LLM,
    and normalize the reply so later steps can rely on its shape."""
    facts = FACT_BUILDERS[specialist](items)
    out = {
        "specialist": specialist,
        "n_items": len(items),
        "item_ids": [it["id"] for it in items],
        "facts": facts,
        "signal": "neutral",
        "confidence": "low",
        "summary": "No content was routed to this specialist.",
        "key_points": [],
        "risks": [],
    }
    if not items:
        return out

    prompt = (
        f"Analyze {ticker} from your specialty.\n"
        f"Question: {SPECIALIST_FOCUS[specialist]}\n\n"
        "Return JSON with:\n"
        f"signal: one of {', '.join(SIGNALS)}\n"
        f"confidence: one of {', '.join(CONFIDENCE)}\n"
        "summary: 2-3 sentences\n"
        "key_points: 3-5 short strings, each quoting a specific number "
        "or citing an item id like [news-3]\n"
        "risks: 1-3 short strings\n\n"
        "Returns are fractions (0.05 = 5%). Do not invent numbers.\n\n"
        f"Facts:\n{json.dumps(facts, indent=1, default=str)}"
    )
    reply = llm.complete_json_cached(
        prompt, system=SPECIALIST_SYSTEMS[specialist], max_tokens=1024)
    reply = reply if isinstance(reply, dict) else {}
    signal = reply.get("signal")
    confidence = reply.get("confidence")
    out.update({
        "signal": signal if signal in SIGNALS else "neutral",
        "confidence": confidence if confidence in CONFIDENCE else "low",
        "summary": str(reply.get("summary", "")),
        "key_points": [str(p) for p in reply.get("key_points") or []],
        "risks": [str(r) for r in reply.get("risks") or []],
    })
    return out


def earnings_analyzer(items: list[dict], ticker: str = "") -> dict:
    """Analyze earnings content: EPS surprises, revenue, margins, cash
    flow, and earnings news."""
    return _analyze("earnings", items, ticker)


def news_analyzer(items: list[dict], ticker: str = "") -> dict:
    """Analyze general company news: products, legal, management."""
    return _analyze("news", items, ticker)


def market_analyzer(items: list[dict], ticker: str = "") -> dict:
    """Analyze price and market data: returns, volatility, valuation,
    analyst targets, and macro."""
    return _analyze("market", items, ticker)


ANALYZERS = {
    "earnings": earnings_analyzer,
    "news": news_analyzer,
    "market": market_analyzer,
}


# ------------------------------------------------------------ the router

def run_routing(ticker: str, chain: dict | None = None,
                use_llm: bool = True, save: bool = True) -> dict:
    """Collect content for the ticker, route it, and run each specialist.

    chain is a prompt_chain.run_chain result; if None the chain is run
    here. Returns {ticker, model, items, decisions, counts, analyses,
    errors, timings, usage, saved_to}. With save=True the result is
    written to data/runs/ as JSON.
    """
    ticker = ticker.upper()
    usage_before = dict(llm.USAGE)
    timings = {}

    start = time.perf_counter()
    if chain is None:
        chain = prompt_chain.run_chain(ticker, save=False)
    articles = items_from_articles(chain["steps"]["classify"],
                                   chain["steps"].get("extract"))
    errors: list = []
    items = articles + items_from_tools(ticker, errors)
    timings["collect"] = round(time.perf_counter() - start, 2)

    start = time.perf_counter()
    decisions = route_items(items, ticker, use_llm=use_llm)
    timings["route"] = round(time.perf_counter() - start, 2)

    by_specialist: dict[str, list] = {s: [] for s in SPECIALISTS}
    item_by_id = {it["id"]: it for it in items}
    for d in decisions:
        by_specialist[d["specialist"]].append(item_by_id[d["item_id"]])

    start = time.perf_counter()
    analyses = {name: ANALYZERS[name](by_specialist[name], ticker)
                for name in SPECIALISTS}
    timings["analyze"] = round(time.perf_counter() - start, 2)

    result = {
        "ticker": ticker,
        "model": config.LLM_MODEL,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "digest": chain.get("digest", {}),
        "items": items,
        "decisions": decisions,
        "counts": {
            "by_specialist": dict(Counter(d["specialist"]
                                          for d in decisions)),
            "by_method": dict(Counter(d["method"] for d in decisions)),
        },
        "analyses": analyses,
        "errors": errors,
        "timings": timings,
        "usage": {k: llm.USAGE[k] - usage_before[k] for k in llm.USAGE},
        "saved_to": None,
    }
    if save:
        result["saved_to"] = _save(result, f"routing_{ticker}")
    return result


def _save(result: dict, name: str) -> str:
    runs = config.DATA_DIR / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = runs / f"{name}_{stamp}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    log.info("saved run to %s", path)
    return str(path)
