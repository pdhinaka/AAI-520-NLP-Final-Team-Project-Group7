"""Offline tests for the evaluator-optimizer: the LLM is faked."""

import pytest

from finagent import config, llm
from finagent.workflows import evaluator_optimizer as eo

CONTEXT = {
    "ticker": "AAPL",
    "news_headline": "Mixed week.",
    "specialists": {
        "market": {"signal": "bullish", "facts": {
            "price": {"last_close": 254.63, "return_3m": 0.0812},
            "valuation": {"marketCap": 3780000000000, "trailingPE": 38.6},
        }},
        "earnings": {"signal": "neutral", "facts": {
            "articles": [{"id": "news-1", "figures": [
                {"value": "$94.9B", "context": "revenue"}]}],
        }},
    },
}

DRAFTS = {
    1: "AAPL trades at $254.63, up 8.1% in 3 months. Margin 12.7%.",
    2: "AAPL trades at $254.63, up 8.1% in 3 months. P/E 38.6x.",
    3: "AAPL at $254.63. Revenue $94.9B [news-1]. Cap $3.78T.",
}


class FakeLLM:
    """Plays generator, optimizer, and evaluator.

    Draft n+1 comes from refining draft n. The grader's scores rise
    with each draft so the loop has something to show.
    """

    def __init__(self, scores=(3, 4, 5)):
        self.scores = dict(zip((1, 2, 3), scores))
        self.calls = []

    def complete(self, prompt, system=None, **kwargs):
        self.calls.append(system)
        if system == eo.GENERATOR_SYSTEM:
            return DRAFTS[1]
        n = next(k for k, v in DRAFTS.items() if v in prompt)
        return DRAFTS[n + 1]

    def complete_json(self, prompt, system=None, **kwargs):
        self.calls.append(system)
        n = next(k for k, v in DRAFTS.items() if v in prompt)
        s = self.scores[n]
        return {"scores": {c: s for c in eo.CRITERIA},
                "strengths": ["clear"],
                "feedback": [f"fix {i}" for i in range(7)]}


@pytest.fixture
def fake_llm(monkeypatch):
    def install(scores=(3, 4, 5)):
        fake = FakeLLM(scores)
        monkeypatch.setattr(llm, "complete", fake.complete)
        monkeypatch.setattr(llm, "complete_json", fake.complete_json)
        return fake
    return install


def test_numbers_in_skips_citations_years_and_counts():
    text = ("In Q3 FY25 (2026) [news-12] 3 of 4 quarters beat; EPS "
            "$1.64, revenue $94.9B, up 8%, 1,250 stores.")
    written = [w for w, _, _ in eo._numbers_in(text)]
    assert written == ["$1.64", "$94.9B", "8%", "1,250"]


def test_grounding_check_matches_scaled_values():
    text = ("Price $254.63, up 8.1% over 3 months; cap $3.78T; P/E "
            "38.6x; revenue $94.9B; invented 12.7% margin and $500M.")
    g = eo.grounding_check(text, CONTEXT)
    assert g["ungrounded"] == ["12.7%", "$500M"]
    assert g["checked"] == 7
    assert g["ratio"] == pytest.approx(5 / 7, abs=1e-3)
    assert eo.grounding_check("No numbers here.", CONTEXT)["ratio"] == 1


def test_evaluate_scores_clamps_and_flags_ungrounded(monkeypatch):
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: {
        "scores": {"coverage": 9, "evidence": 0, "balance": "x"},
        "feedback": ["add risks"]})
    review = eo.evaluate(DRAFTS[1], CONTEXT)
    s = review["scores"]
    assert (s["coverage"], s["evidence"], s["balance"]) == (5, 1, 1)
    assert s["clarity"] == 1  # missing scores count against the draft
    # one of three numbers ($254.63, 8.1%, 12.7%) is made up
    assert s[eo.GROUNDING] == pytest.approx(1 + 4 * 2 / 3, abs=0.01)
    assert review["feedback"][0].startswith("These numbers do not")
    assert "12.7%" in review["feedback"][0]
    assert review["feedback"][1] == "add risks"


def test_loop_refines_until_threshold(fake_llm, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    fake = fake_llm((3, 4, 5))
    result = eo.run_loop(CONTEXT, max_iters=5, threshold=0.95)

    overall = [h["overall"] for h in result["history"]]
    assert len(overall) == 3 and overall == sorted(overall)
    assert result["passed"] and result["best_iteration"] == 3
    assert result["final"] == DRAFTS[3]
    assert len(result["history"][0]["feedback"]) == eo.MAX_FEEDBACK
    assert fake.calls.count(eo.OPTIMIZER_SYSTEM) == 2
    assert result["saved_to"].startswith(str(tmp_path))


def test_loop_returns_best_draft_not_last(fake_llm):
    fake_llm((4, 2, 3))
    result = eo.run_loop(CONTEXT, max_iters=3, threshold=0.99,
                         save=False)
    assert len(result["history"]) == 3
    assert result["best_iteration"] == 1
    assert result["final"] == DRAFTS[1]
    assert not result["passed"]


def test_loop_is_cached(fake_llm):
    fake = fake_llm((3, 4, 5))
    first = eo.run_loop(CONTEXT, threshold=0.95, save=False)
    n_calls = len(fake.calls)
    again = eo.run_loop(CONTEXT, threshold=0.95, save=False)
    assert len(fake.calls) == n_calls
    assert again["final"] == first["final"]


def test_context_from_routing_keeps_views_and_facts():
    result = {"ticker": "AAPL", "digest": {"headline": "h"},
              "analyses": {"news": {
                  "signal": "bearish", "confidence": "low",
                  "summary": "s", "key_points": [], "risks": [],
                  "facts": {"articles": []}, "item_ids": ["news-1"]}}}
    ctx = eo.context_from_routing(result)
    assert ctx["news_headline"] == "h"
    assert "item_ids" not in ctx["specialists"]["news"]
    assert ctx["specialists"]["news"]["signal"] == "bearish"


def test_grounding_skips_hypothetical_triggers():
    text = ("Price $254.63.\n\n### What would change our view\n"
            "Downgrade if growth falls below 4% or margin under 22.5%.")
    g = eo.grounding_check(text, CONTEXT)
    assert (g["checked"], g["ungrounded"]) == (1, [])


def test_one_weak_criterion_blocks_passing(monkeypatch):
    review = {"overall": 0.9, "scores": {"evidence": 3, "clarity": 5}}
    assert not eo.passes(review, threshold=0.8, min_score=4)
    assert eo.passes(review, threshold=0.8, min_score=3)
