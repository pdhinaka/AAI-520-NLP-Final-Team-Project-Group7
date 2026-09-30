"""Prompt chaining: ingest -> preprocess -> classify -> extract -> summarize.

Workflow Pattern 1. Each step takes the previous step's output, so the
notebook can run the steps one at a time and show what each produced:

    from finagent.workflows import prompt_chain as pc

    result = pc.run_chain("AAPL")
    result["digest"]["summary"]      # markdown news digest
    result["steps"]["classify"]      # articles with topic/sentiment

Ingest and preprocess are plain Python. Classify, extract, and summarize
are one LLM call each, batched over the articles. LLM replies go through
the disk cache, so with CACHE_MODE=offline a rerun gives the same output.

Owner: Workstream A.
"""

import difflib
import html
import json
import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone

from finagent import config, llm
from finagent.schemas import NewsDigest
from finagent.tools import news

log = logging.getLogger(__name__)

TOPICS = (
    "earnings", "product", "legal", "macro", "analyst_rating", "other",
)
SENTIMENTS = ("positive", "neutral", "negative")

MAX_CHARS = 1500  # article text is trimmed to this before any LLM call
EXTRACT_BATCH = 8  # articles per extract call, keeps replies short
TITLE_SIMILARITY = 0.9  # titles at least this similar count as dupes
MAX_PER_SOURCE = 4  # keeps one outlet from dominating the digest

_TAG = re.compile(r"<[^>]+>")
_NEWSAPI_TAIL = re.compile(r"\[\+\d+ chars\]")
_SPACE = re.compile(r"\s+")
# [3] or [3, 7]
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


# ---------------------------------------------------------------- helpers

def clean_text(text: str) -> str:
    """Unescape HTML, drop tags and NewsAPI's '[+123 chars]' tail, and
    collapse whitespace."""
    text = html.unescape(text or "")
    text = _TAG.sub(" ", text)
    text = _NEWSAPI_TAIL.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def trim(text: str, max_chars: int = MAX_CHARS) -> str:
    """Cut text at a word boundary so it fits in max_chars."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " ..."


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _norm_url(url: str) -> str:
    return url.split("?")[0].rstrip("/").lower()


def _mentions(article: dict, names: list[str]) -> bool:
    """True if the title or text names the company as a whole word.

    Case-sensitive on purpose: "Apple" matches, "apples" does not.
    """
    blob = f"{article['title']} {article['text']}"
    return any(re.search(rf"\b{re.escape(n)}\b", blob) for n in names)


def _article_block(a: dict) -> str:
    date = (a.get("published_at") or "")[:10]
    return f"[{a['id']}] {a['title']} ({a['source']}, {date})\n{a['text']}"


def _ask_json(prompt: str, system: str, max_tokens: int = 2048):
    """One cached JSON call to the LLM. The cache, not a sampling
    setting, is what makes reruns repeatable."""
    return llm.complete_json_cached(
        prompt, system=system, max_tokens=max_tokens
    )


def _by_id(reply) -> dict:
    """Index a model reply (a list, or {"articles": [...]}) by id."""
    if isinstance(reply, dict):
        reply = next(
            (v for v in reply.values() if isinstance(v, list)), []
        )
    out = {}
    for item in reply or []:
        try:
            out[int(item["id"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ------------------------------------------------------------ 1. ingest

def ingest(ticker: str, days: int = 7, limit: int = 30,
           source: str = "auto") -> list[dict]:
    """Fetch raw news articles for the ticker (see tools.news.get_news)."""
    articles = news.get_news(ticker, days=days, limit=limit,
                             source=source)
    by_provider = Counter(a.get("provider", "?") for a in articles)
    log.info("ingest %s: %d articles %s", ticker, len(articles),
             dict(by_provider))
    return articles


# -------------------------------------------------------- 2. preprocess

def preprocess(articles: list[dict], company: str = "",
               max_chars: int = MAX_CHARS,
               dropped: list | None = None) -> list[dict]:
    """Clean, filter, dedupe, and trim article text.

    - Cleans HTML and truncation markers out of titles and text.
    - Drops articles that never name the company or ticker (skipped if
      the company name is unknown).
    - Drops duplicates by URL and by near-identical title, keeping the
      copy with more text.
    - Keeps at most MAX_PER_SOURCE articles (the newest) per outlet.
    - Trims text to max_chars, sorts newest first, and numbers the
      articles 1..n so later steps can cite them.

    If a list is passed as dropped, one {title, reason} entry is added
    for each article removed.
    """
    dropped = dropped if dropped is not None else []
    ticker = articles[0].get("ticker", "") if articles else ""
    names = [n for n in (company, ticker) if n]

    cleaned = []
    for a in articles:
        a = dict(a)
        a["title"] = clean_text(a.get("title", ""))
        a["text"] = clean_text(a.get("text", "")) or a["title"]
        if not a["title"]:
            dropped.append({"title": "", "reason": "no title"})
            continue
        if company and not _mentions(a, names):
            dropped.append({"title": a["title"],
                            "reason": "does not mention the company"})
            continue
        cleaned.append(a)

    # Longest text first, so the copy we keep of a duplicate is the
    # most complete one.
    cleaned.sort(key=lambda a: len(a["text"]), reverse=True)
    kept, seen_urls, seen_titles = [], set(), []
    for a in cleaned:
        url, title = _norm_url(a.get("url", "")), _norm_title(a["title"])
        if url and url in seen_urls:
            dropped.append({"title": a["title"],
                            "reason": "duplicate URL"})
            continue
        if any(difflib.SequenceMatcher(None, title, t).ratio()
               >= TITLE_SIMILARITY for t in seen_titles):
            dropped.append({"title": a["title"],
                            "reason": "duplicate title"})
            continue
        seen_urls.add(url)
        seen_titles.append(title)
        kept.append(a)

    kept.sort(key=lambda a: a.get("published_at") or "", reverse=True)
    per_source = Counter()
    capped = []
    for a in kept:
        per_source[a["source"]] += 1
        if per_source[a["source"]] > MAX_PER_SOURCE:
            dropped.append({"title": a["title"],
                            "reason": f"over {MAX_PER_SOURCE} from "
                                      f"{a['source']}"})
            continue
        capped.append(a)
    kept = capped
    for i, a in enumerate(kept, start=1):
        a["id"] = i
        a["text"] = trim(a["text"], max_chars)
    log.info("preprocess: kept %d, dropped %d", len(kept), len(dropped))
    return kept


# ---------------------------------------------------------- 3. classify

CLASSIFY_SYSTEM = (
    "You are a financial news analyst. You label news articles about a "
    "company for an equity research team. Judge sentiment from the "
    "point of view of the company's shareholders."
)


def classify(articles: list[dict]) -> list[dict]:
    """Label each article with a topic, sentiment, and relevance.

    Labels outside the allowed sets are replaced with "other" and
    "neutral", so later steps can rely on them.
    """
    if not articles:
        return []
    ticker = articles[0].get("ticker", "")
    prompt = (
        f"Label each news article about {ticker}.\n\n"
        f"topic: one of {', '.join(TOPICS)} (legal covers lawsuits, "
        "antitrust, and regulation; product covers products, services, "
        "and security issues)\n"
        f"sentiment: one of {', '.join(SENTIMENTS)} (for {ticker} "
        "shareholders)\n"
        f"relevant: true only if a {ticker} investor would care: company "
        "news, results, product launches, legal or regulatory action, "
        "security issues, analyst views. false for retail deals and "
        "discount listings, software packages or tools that merely "
        "support the company's products, and passing mentions\n"
        "reason: one short sentence\n\n"
        'Return a JSON list: [{"id": 1, "topic": "...", '
        '"sentiment": "...", "relevant": true, "reason": "..."}]\n\n'
        "Articles:\n\n"
        + "\n\n".join(_article_block(a) for a in articles)
    )
    labels = _by_id(_ask_json(prompt, CLASSIFY_SYSTEM))

    out = []
    for a in articles:
        got = labels.get(a["id"], {})
        topic = got.get("topic") if got.get("topic") in TOPICS else "other"
        sentiment = got.get("sentiment")
        if sentiment not in SENTIMENTS:
            sentiment = "neutral"
        relevant = got.get("relevant", True)
        a = dict(a)
        a["labels"] = {
            "topic": topic,
            "sentiment": sentiment,
            "relevant": relevant if isinstance(relevant, bool) else True,
            "reason": got.get("reason", "not labeled by the model"),
        }
        out.append(a)
    log.info("classify: %s", dict(Counter(
        a["labels"]["sentiment"] for a in out)))
    return out


# ----------------------------------------------------------- 4. extract

EXTRACT_SYSTEM = (
    "You extract facts from financial news for an equity research "
    "team. Only use what the article says. Never invent figures."
)


def extract(articles: list[dict]) -> list[dict]:
    """Pull entities, figures, dates, and the key event from each
    relevant article. Articles marked not relevant are skipped."""
    relevant = [a for a in articles if a["labels"].get("relevant", True)]
    facts = {}
    for start in range(0, len(relevant), EXTRACT_BATCH):
        batch = relevant[start:start + EXTRACT_BATCH]
        prompt = (
            "For each article, extract:\n"
            "companies, people, products: names mentioned\n"
            "figures: numbers that matter, each as "
            '{"value": "$94.9B", "context": "Q3 revenue"}\n'
            "dates: dates or periods mentioned (as written)\n"
            "event: one sentence on what happened\n\n"
            'Return a JSON list: [{"id": 1, "companies": [], '
            '"people": [], "products": [], "figures": [], '
            '"dates": [], "event": "..."}]\n\n'
            "Articles:\n\n"
            + "\n\n".join(_article_block(a) for a in batch)
        )
        facts.update(_by_id(_ask_json(prompt, EXTRACT_SYSTEM)))

    out = []
    for a in relevant:
        got = facts.get(a["id"], {})
        out.append({
            "id": a["id"],
            "ticker": a.get("ticker", ""),
            "title": a["title"],
            "source": a["source"],
            "published_at": a.get("published_at", ""),
            "url": a.get("url", ""),
            "topic": a["labels"]["topic"],
            "sentiment": a["labels"]["sentiment"],
            "companies": list(got.get("companies") or []),
            "people": list(got.get("people") or []),
            "products": list(got.get("products") or []),
            "figures": list(got.get("figures") or []),
            "dates": list(got.get("dates") or []),
            "event": got.get("event") or a["title"],
        })
    log.info("extract: %d articles", len(out))
    return out


# --------------------------------------------------------- 5. summarize

SUMMARIZE_SYSTEM = (
    "You write short, factual news digests for equity analysts. Use "
    "only the facts provided and cite every claim with the article id "
    "in square brackets, like [3]."
)


def summarize(extracted: list[dict], ticker: str = "") -> dict:
    """Write a short news digest from the extracted facts.

    Sentiment and topic counts are computed here, not by the model.
    Citations in the summary that don't match an article id are listed
    in unknown_citations, a cheap check on the model's output.
    """
    ticker = ticker or (extracted[0].get("ticker", "") if extracted
                        else "")
    digest = NewsDigest(
        ticker=ticker,
        n_articles=len(extracted),
        sentiment_counts={s: 0 for s in SENTIMENTS},
        topic_counts={},
        sources=[{"id": e["id"], "title": e["title"], "url": e["url"]}
                 for e in extracted],
    )
    digest.sentiment_counts.update(
        Counter(e["sentiment"] for e in extracted))
    digest.topic_counts = dict(
        Counter(e["topic"] for e in extracted).most_common())
    if not extracted:
        digest.headline = f"No relevant recent news found for {ticker}."
        return digest.to_dict()

    facts = [{k: v for k, v in e.items() if k != "url"}
             for e in extracted]
    prompt = (
        f"Write a news digest for {ticker} from these extracted facts.\n"
        f"Sentiment counts: {json.dumps(digest.sentiment_counts)}\n"
        f"Topic counts: {json.dumps(digest.topic_counts)}\n\n"
        "Return JSON with:\n"
        "headline: one sentence on the overall news picture\n"
        "summary: 3-6 markdown bullet points grouped by theme, each "
        "citing article ids like [2]\n"
        "catalysts: list of short strings (possible upside), with ids\n"
        "risks: list of short strings (possible downside), with ids\n\n"
        f"Facts:\n{json.dumps(facts, indent=1)}"
    )
    reply = _ask_json(prompt, SUMMARIZE_SYSTEM)
    reply = reply if isinstance(reply, dict) else {}
    digest.headline = str(reply.get("headline", ""))
    summary = reply.get("summary", "")
    if isinstance(summary, list):
        summary = "\n".join(f"- {s}" for s in summary)
    digest.summary = str(summary)
    digest.catalysts = list(reply.get("catalysts") or [])
    digest.risks = list(reply.get("risks") or [])

    ids = {e["id"] for e in extracted}
    text = " ".join([digest.summary, *map(str, digest.catalysts),
                     *map(str, digest.risks)])
    cited = {int(n) for group in _CITE.findall(text)
             for n in group.split(",")}
    digest.unknown_citations = sorted(cited - ids)
    return digest.to_dict()


# ------------------------------------------------------------- the chain

def run_chain(ticker: str, days: int = 7, limit: int = 30,
              source: str = "auto", save: bool = True) -> dict:
    """Run every step and return the digest plus intermediate outputs.

    Returns {ticker, model, digest, steps, timings, usage, saved_to}.
    steps holds each step's output (preprocess also lists what it
    dropped and why). With save=True the whole result is written to
    data/runs/ as JSON.
    """
    ticker = ticker.upper()
    usage_before = dict(llm.USAGE)
    timings, steps = {}, {}

    def timed(name, fn, *args, **kwargs):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        timings[name] = round(time.perf_counter() - start, 2)
        return out

    raw = timed("ingest", ingest, ticker, days, limit, source)
    steps["ingest"] = raw

    dropped = []
    company = news._company_name(ticker)
    clean = timed("preprocess", preprocess, raw, company=company,
                  dropped=dropped)
    steps["preprocess"] = {"company": company, "articles": clean,
                           "dropped": dropped}

    labeled = timed("classify", classify, clean)
    steps["classify"] = labeled

    extracted = timed("extract", extract, labeled)
    steps["extract"] = extracted

    digest = timed("summarize", summarize, extracted, ticker)

    result = {
        "ticker": ticker,
        "model": config.LLM_MODEL,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "digest": digest,
        "steps": steps,
        "timings": timings,
        "usage": {k: llm.USAGE[k] - usage_before[k] for k in llm.USAGE},
        "saved_to": None,
    }
    if save:
        runs = config.DATA_DIR / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = runs / f"prompt_chain_{ticker}_{stamp}.json"
        result["saved_to"] = str(path)
        path.write_text(json.dumps(result, indent=2, default=str))
        log.info("saved chain run to %s", path)
    return result


def get_news_digest(ticker: str, days: int = 7, limit: int = 30) -> dict:
    """Short digest of recent news for a stock, built by the prompt chain.

    Fetches, cleans, and dedupes recent articles, labels each by topic
    (earnings, product, legal, macro, analyst_rating, other) and
    sentiment, extracts key facts, and summarizes them. Returns
    headline, summary (cites articles as [id]), catalysts, risks,
    sentiment_counts, topic_counts, and sources. Use this instead of
    get_news when you want the news already analyzed.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        days: How far back to look.
        limit: Max articles to fetch before filtering (split across
            news sources).
    """
    return run_chain(ticker, days=days, limit=limit, save=False)["digest"]
