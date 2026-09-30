"""Offline tests for the prompt chain: the LLM and news fetch are faked."""

import pytest

from finagent import config, llm
from finagent.tools import news, registry
from finagent.workflows import prompt_chain as pc


def art(title, text="", url="", date="2026-09-27T12:00:00Z",
        source="Reuters"):
    return {"title": title, "text": text, "url": url, "source": source,
            "published_at": date, "ticker": "AAPL", "provider": "yahoo",
            "labels": {}}


RAW = [
    art("Apple beats Q4 estimates", "Apple reported revenue of $94.9B.",
        "https://x.com/a?utm=1", "2026-09-26T10:00:00Z"),
    art("Apple beats Q4 estimates", "Short copy.", "https://x.com/a",
        "2026-09-26T10:00:00Z"),
    art("Apple Beats Q4 Estimates!", "Apple beat on revenue and more.",
        "https://y.com/b", "2026-09-26T11:00:00Z"),
    art("Chinese Toffee Apples", "Battered apples in caramel.",
        "https://food.com/c"),
    art("Jury finds Apple infringed patents",
        "<p>A jury awarded $5.7B &amp; costs.</p> [+812 chars]",
        "https://z.com/d", "2026-09-27T13:00:00Z"),
]


class FakeLLM:
    """Stands in for llm.complete_json and records each call."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, system=None, **kwargs):
        self.calls.append(system)
        ids = [int(x) for x in pc._CITE.findall(prompt)]
        if system == pc.CLASSIFY_SYSTEM:
            return [{"id": 1, "topic": "legal", "sentiment": "negative",
                     "relevant": True, "reason": "verdict"},
                    {"id": 2, "topic": "rumors", "sentiment": "great",
                     "relevant": "yes"}]
        if system == pc.EXTRACT_SYSTEM:
            return {"articles": [
                {"id": i, "companies": ["Apple"],
                 "figures": [{"value": "$1", "context": "x"}],
                 "event": f"event {i}"} for i in ids]}
        return {"headline": "Mixed week.",
                "summary": ["Verdict hurts [1]", "Beat [2] and [9]",
                            "Both [1, 2, 12]"],
                "catalysts": ["Beat [2]"], "risks": ["Appeal [1]"]}


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "complete_json", fake)
    return fake


def test_clean_text_and_trim():
    raw = "<p>A&amp;B   rose</p>\n[+812 chars]"
    assert pc.clean_text(raw) == "A&B rose"
    assert pc.trim("one two three", 9) == "one two ..."
    assert pc.trim("short", 9) == "short"


def test_preprocess_filters_dedupes_and_numbers():
    dropped = []
    out = pc.preprocess(RAW, company="Apple", dropped=dropped)
    assert [a["title"] for a in out] == [
        "Jury finds Apple infringed patents", "Apple beats Q4 estimates"]
    assert [a["id"] for a in out] == [1, 2]
    # the longer copy of the duplicate story is the one kept
    assert out[1]["text"] == "Apple reported revenue of $94.9B."
    assert out[0]["text"] == "A jury awarded $5.7B & costs."
    reasons = sorted(d["reason"] for d in dropped)
    assert reasons == ["does not mention the company", "duplicate URL",
                       "duplicate title"]


def test_preprocess_without_company_keeps_everything_relevant():
    out = pc.preprocess(RAW[3:4])
    assert len(out) == 1


def test_classify_repairs_bad_labels(fake_llm):
    arts = pc.preprocess(RAW, company="Apple")
    out = pc.classify(arts)
    assert out[0]["labels"]["topic"] == "legal"
    assert out[1]["labels"] == {"topic": "other", "sentiment": "neutral",
                                "relevant": True,
                                "reason": "not labeled by the model"}
    assert pc.classify([]) == []


def test_extract_skips_irrelevant_and_batches(fake_llm, monkeypatch):
    monkeypatch.setattr(pc, "EXTRACT_BATCH", 2)
    arts = [dict(art(f"Apple {i}"), id=i,
                 labels={"topic": "product", "sentiment": "neutral",
                         "relevant": i != 3})
            for i in range(1, 6)]
    out = pc.extract(arts)
    assert [e["id"] for e in out] == [1, 2, 4, 5]
    assert fake_llm.calls.count(pc.EXTRACT_SYSTEM) == 2
    assert out[0]["event"] == "event 1"


def test_summarize_counts_and_checks_citations(fake_llm):
    extracted = [
        {"id": 1, "title": "t1", "url": "u1", "topic": "legal",
         "sentiment": "negative"},
        {"id": 2, "title": "t2", "url": "u2", "topic": "earnings",
         "sentiment": "positive"},
    ]
    d = pc.summarize(extracted, "AAPL")
    assert d["sentiment_counts"] == {"positive": 1, "neutral": 0,
                                     "negative": 1}
    assert d["summary"].startswith("- Verdict hurts [1]\n- Beat [2]")
    assert d["unknown_citations"] == [9, 12]
    empty = pc.summarize([], "AAPL")
    assert empty["n_articles"] == 0 and "No relevant" in empty["headline"]


def test_run_chain_end_to_end_and_llm_cache(fake_llm, monkeypatch,
                                            tmp_path):
    monkeypatch.setattr(news, "get_news", lambda *a, **k: list(RAW))
    monkeypatch.setattr(news, "_company_name", lambda t: "Apple")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    result = pc.run_chain("aapl")
    assert result["ticker"] == "AAPL"
    assert set(result["steps"]) == {"ingest", "preprocess", "classify",
                                    "extract"}
    assert len(result["steps"]["preprocess"]["dropped"]) == 3
    assert result["digest"]["n_articles"] == 2
    assert result["saved_to"].startswith(str(tmp_path))
    n_calls = len(fake_llm.calls)
    assert n_calls == 3

    # Same input again: every LLM reply comes from the disk cache
    again = pc.run_chain("AAPL", save=False)
    assert len(fake_llm.calls) == n_calls
    assert again["digest"] == result["digest"]


def test_digest_tool_is_registered():
    names = [s["name"] for s in registry.tool_specs()]
    assert "get_news_digest" in names


def test_preprocess_caps_articles_per_source(monkeypatch):
    monkeypatch.setattr(pc, "MAX_PER_SOURCE", 2)
    titles = ["Apple ships Vision Pro", "Apple Pay reaches India",
              "Jury rules against Apple", "Apple raises card rates"]
    raw = [art(titles[i],
               url=f"https://m.com/{i}",
               date=f"2026-09-2{i}T00:00:00Z", source="MacRumors")
           for i in range(4)]
    dropped = []
    out = pc.preprocess(raw, company="Apple", dropped=dropped)
    assert [a["published_at"][:10] for a in out] == ["2026-09-23",
                                                     "2026-09-22"]
    assert [d["reason"] for d in dropped] == ["over 2 from MacRumors"] * 2
