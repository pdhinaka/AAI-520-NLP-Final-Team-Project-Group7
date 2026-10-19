from finagent.agent import memory, reflection


def test_reflection_flags_missing_thesis_and_catalysts(monkeypatch):
    """A weak report should self-report the missing thesis and catalysts."""

    def fake_complete_json(prompt, system=None, **kwargs):
        return {
            "coverage_score": 7,
            "evidence_score": 6,
            "recency_score": 8,
            "balance_score": 5,
            "feedback": (
                "The report is weak on the investment thesis and catalysts."
            ),
            "gaps": [],
        }

    monkeypatch.setattr(reflection.llm, "complete_json", fake_complete_json)

    report = {
        "ticker": "AAPL",
        "summary": "Research completed.",
        "key_facts": {"ticker": "AAPL"},
        "upside": ["Based on gathered data"],
        "risks": ["Based on gathered data"],
        "catalysts": [],
        "outlook": "Hold pending full analysis",
    }

    result = reflection.reflect(report)

    assert result["score"] == (7 + 6 + 8 + 5) / 4  # 0-10, mean of four
    assert result["balance_score"] == 5
    assert result["grounding"] is None  # no facts given
    assert any("thesis" in gap.lower() for gap in result["gaps"])
    assert any("catalyst" in gap.lower() for gap in result["gaps"])


def test_memory_persists_and_deduplicates_ticker_notes(tmp_path, monkeypatch):
    """Memory should be durable and not repeat a lesson across runs."""

    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path)

    memory.save_notes("AAPL", ["First note", "Second note"])
    memory.save_notes("AAPL", ["Second note", "Third note"])

    notes = memory.load_notes("AAPL")

    assert notes == ["First note", "Second note", "Third note"]
    assert (tmp_path / "AAPL.json").exists()
    records = memory.load_records("AAPL")
    assert {r["category"] for r in records} == {"finding"}
