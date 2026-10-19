"""Offline tests for the agent's four functions: planning, tool use,
self-reflection, and learning across runs."""

import json
import logging

from test_report import REPORT, TOOL_DATA, fake_json, text_reply, tool_reply

from finagent import llm
from finagent.agent import memory, planner, reflection, research_agent

NOTES = [
    {"note": "Prior view: Buy (medium conviction).",
     "category": "finding"},
    {"note": "Earlier report gap: no segment revenue",
     "category": "lesson"},
    {"note": "get_earnings returned an error (RuntimeError: Alpha "
             "Vantage); get_financials worked instead.",
     "category": "tool"},
]


def scripted(replies):
    it = iter(replies)
    return lambda *a, **k: next(it)


# ------------------------------------------------------------- planning

def test_planner_puts_grouped_notes_in_prompt(monkeypatch):
    seen = {}

    def complete_json(prompt, system=None, **kwargs):
        seen["prompt"] = prompt
        return {"steps": [{"step": "Fetch segment revenue",
                           "tools": ["get_financials", "made_up_tool"]}],
                "memory_used": [{"note": NOTES[1]["note"],
                                 "effect": "added a segment step"}]}

    monkeypatch.setattr(llm, "complete_json", complete_json)
    plan = planner.make_plan_detailed("AAPL", NOTES)

    assert "Lessons: gaps flagged in earlier reports" in seen["prompt"]
    assert "- Earlier report gap: no segment revenue" in seen["prompt"]
    assert "Tool lessons" in seen["prompt"]
    assert "get_news_digest" in seen["prompt"]  # tools are listed
    assert plan["source"] == "llm"
    assert plan["steps"] == ["Fetch segment revenue"]
    assert plan["step_tools"] == [["get_financials"]]
    assert plan["memory_used"][0]["effect"] == "added a segment step"


def test_planner_without_notes_reports_no_memory_use(monkeypatch):
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: {
        "steps": ["a", "b"], "memory_used": [{"note": "invented"}]})
    plan = planner.make_plan_detailed("AAPL", [])
    assert plan["memory_used"] == []
    assert plan["steps"] == ["a", "b"]


def test_planner_fallback_is_logged_and_recorded(monkeypatch, caplog):
    def broken(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(llm, "complete_json", broken)
    with caplog.at_level(logging.WARNING):
        plan = planner.make_plan_detailed("AAPL", NOTES)
    assert plan["source"] == "fallback"
    assert "api down" in plan["error"]
    assert plan["steps"] == planner.DEFAULT_PLAN
    assert "api down" in caplog.text


def test_plan_replays_from_cache(monkeypatch):
    calls = []

    def complete_json(prompt, system=None, **kwargs):
        calls.append(prompt)
        return {"steps": ["a"], "memory_used": []}

    monkeypatch.setattr(llm, "complete_json", complete_json)
    planner.make_plan("AAPL")
    planner.make_plan("AAPL")
    assert len(calls) == 1


# ------------------------------------------------------------- tool use

PLAN = {"steps": ["Price", "News"],
        "step_tools": [["get_price_summary"], ["get_news_digest"]]}


def test_loop_runs_several_turns_and_stops_on_text(monkeypatch):
    seen = []

    def create(messages, **kwargs):
        seen.append(len(messages))
        return replies(messages)

    replies = scripted([
        tool_reply("get_price_summary", "get_company_info"),
        tool_reply("get_news_digest"),
        text_reply("All steps covered."),
    ])
    monkeypatch.setattr(llm, "create_message_cached", create)
    monkeypatch.setattr(research_agent.registry, "call",
                        lambda name, args=None: TOOL_DATA[name])

    out = research_agent.gather("AAPL", PLAN)

    assert out["turns"] == 3 and out["stopped"] == "model_finished"
    assert out["final_text"] == "All steps covered."
    assert set(out["facts"]) == set(TOOL_DATA)
    assert seen == [1, 3, 5]  # history grows by one exchange per turn


def test_loop_records_errors_and_truncates(monkeypatch):
    sent = []

    def create(messages, **kwargs):
        if messages[-1]["role"] == "user" and isinstance(
                messages[-1]["content"], list):
            sent.extend(messages[-1]["content"])
        return replies(messages)

    def call(name, args=None):
        if name == "get_earnings":
            return {"error": "RuntimeError: Alpha Vantage: rate limit"}
        return {"blob": "x" * 20000}

    replies = scripted([tool_reply("get_earnings", "get_financials"),
                        text_reply()])
    monkeypatch.setattr(llm, "create_message_cached", create)
    monkeypatch.setattr(research_agent.registry, "call", call)

    out = research_agent.gather("AAPL", PLAN)

    assert [t["ok"] for t in out["trace"]] == [False, True]
    assert out["errors"][0]["tool"] == "get_earnings"
    assert "get_earnings" not in out["facts"]
    assert sent[0]["is_error"] is True
    assert out["trace"][1]["truncated"]
    assert len(sent[1]["content"]) < research_agent.TOOL_RESULT_CHARS + 50


def test_loop_stops_at_max_turns(monkeypatch):
    monkeypatch.setattr(llm, "create_message_cached",
                        lambda *a, **k: tool_reply("get_price_summary"))
    monkeypatch.setattr(research_agent.registry, "call",
                        lambda name, args=None: TOOL_DATA[name])
    out = research_agent.gather("AAPL", PLAN, max_turns=3)
    assert out["turns"] == 3 and out["stopped"] == "max_turns"
    assert len(out["trace"]) == 3


def test_second_call_with_same_tool_keeps_both_results():
    facts = {"get_series": {"series_id": "UNRATE"}}
    key = research_agent._fact_key(facts, "get_series",
                                   {"series_id": "DGS10"})
    assert key == "get_series(series_id=DGS10)"


# ------------------------------------------------------ self-reflection

FACTS = research_agent.compact_facts(dict(TOOL_DATA))


def test_reflect_returns_all_fields_on_a_0_to_10_scale(monkeypatch):
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: {
        "coverage_score": 15, "evidence_score": "7",
        "recency_score": None, "balance_score": 4,
        "feedback": "ok", "gaps": ["add macro"]})
    report = research_agent._finish("AAPL", REPORT, TOOL_DATA)
    out = reflection.reflect(report, FACTS)

    assert out["coverage_score"] == 10  # clamped
    assert out["evidence_score"] == 7
    assert out["recency_score"] == reflection.DEFAULT_SCORE
    assert out["score"] == round((10 + 7 + 6 + 4) / 4, 2)
    assert out["grounding"]["ungrounded"] == []
    assert out["gaps"][0] == "add macro"


def test_reflect_flags_numbers_missing_from_facts(monkeypatch):
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: {})
    report = research_agent._finish(
        "AAPL", {**REPORT, "summary": "P/E is 51.7 on a 52-week high."},
        TOOL_DATA)
    out = reflection.reflect(report, FACTS)
    assert out["grounding"]["ungrounded"] == ["51.7"]  # "52-week" skipped
    assert "51.7" in out["gaps"][0]


def test_revision_runs_until_the_critique_passes(monkeypatch):
    rounds = iter([((5, 5, 5, 5), ["add risks"]), ((9, 8, 8, 8), [])])
    writes = []

    def complete_json(prompt, system=None, **kwargs):
        if system == reflection.REFLECT_SYSTEM:
            s, fixable = next(rounds)
            return dict(zip(("coverage_score", "evidence_score",
                             "recency_score", "balance_score"), s),
                        feedback="thin", fixable=fixable,
                        needs_data=["peer multiples"])
        writes.append(prompt)
        return {**REPORT, "outlook": "Revised."}

    monkeypatch.setattr(llm, "complete_json", complete_json)
    draft = research_agent._finish("AAPL", REPORT, TOOL_DATA)
    out = research_agent.reflect_and_revise("AAPL", draft, TOOL_DATA)

    assert [h["score"] for h in out["history"]] == [5.0, 8.25]
    assert out["best_index"] == 1
    assert out["best"]["outlook"] == "Revised."
    assert "- add risks" in writes[0]  # fixable gaps go to the reviser
    assert "peer multiples" not in writes[0]
    assert [h["revise"] for h in out["history"]] == [True, False]


def test_fixable_gap_triggers_revision_despite_high_score(monkeypatch):
    calls = iter([["The -1.6% is quarter over quarter, not yoy"], []])

    def complete_json(prompt, system=None, **kwargs):
        if system == reflection.REFLECT_SYSTEM:
            return {"coverage_score": 8, "evidence_score": 8,
                    "recency_score": 8, "balance_score": 8,
                    "fixable": next(calls),
                    "needs_data": ["segment revenue"]}
        return {**REPORT, "outlook": "Revised."}

    monkeypatch.setattr(llm, "complete_json", complete_json)
    draft = research_agent._finish("AAPL", REPORT, TOOL_DATA)
    out = research_agent.reflect_and_revise("AAPL", draft, TOOL_DATA)
    assert [h["revise"] for h in out["history"]] == [True, False]
    assert out["best_index"] == 1
    assert out["history"][1]["gaps"] == ["segment revenue"]


def test_lessons_prefer_missing_data_over_fixed_gaps():
    review = {"gaps": ["x", "segment revenue"], "fixable": [],
              "needs_data": ["segment revenue"]}
    run = {"coverage": {a: True for a in research_agent.AREAS},
           "errors": [], "trace": []}
    notes = research_agent.lessons("AAPL", REPORT, review, run)
    lessons = [n["note"] for n in notes["ticker"]
               if n["category"] == "lesson"]
    assert lessons == ["Earlier report gap: segment revenue"]


def test_compact_facts_name_growth_period_and_debt_unit():
    income = {d: {"Total Revenue": r} for d, r in (
        ("2026-06-30", 99.0), ("2026-03-31", 100.6),
        ("2025-12-31", 120.0), ("2025-09-30", 95.0),
        ("2025-06-30", 90.0))}
    facts = {"get_financials": {"quarterly": True,
                                "income_statement": income,
                                "cash_flow": {}},
             "get_company_info": {"debtToEquity": 78.4}}
    out = research_agent.compact_facts(facts)
    fin = out["get_financials"]
    assert "revenue_growth" not in fin
    assert fin["period_type"] == "quarter"
    assert fin["revenue_growth_qoq"] == round(99.0 / 100.6 - 1, 4)
    assert fin["revenue_growth_yoy"] == round(99.0 / 90.0 - 1, 4)
    assert out["get_company_info"]["debtToEquity_pct"] == 78.4
    assert "debtToEquity" not in out["get_company_info"]


def test_revision_stops_after_max_revisions(monkeypatch):
    def complete_json(prompt, system=None, **kwargs):
        if system == reflection.REFLECT_SYSTEM:
            return {"coverage_score": 3, "evidence_score": 3,
                    "recency_score": 3, "balance_score": 3}
        return {**REPORT, "outlook": f"v{len(prompt)}"}

    monkeypatch.setattr(llm, "complete_json", complete_json)
    draft = research_agent._finish("AAPL", REPORT, TOOL_DATA)
    out = research_agent.reflect_and_revise("AAPL", draft, TOOL_DATA)
    assert len(out["drafts"]) == 1 + research_agent.MAX_REVISIONS


# --------------------------------------------------------------- memory

def test_memory_categories_and_general_file(tmp_path):
    memory.save_notes("AAPL", [NOTES[0], NOTES[1]], memory_dir=tmp_path)
    memory.save_general([NOTES[2]["note"]], memory_dir=tmp_path)
    memory.save_general([NOTES[2]["note"]], memory_dir=tmp_path)

    stored = json.loads((tmp_path / "AAPL.json").read_text())
    assert [r["category"] for r in stored] == ["finding", "lesson"]
    general = json.loads((tmp_path / "general.json").read_text())
    assert len(general) == 1 and general[0]["category"] == "tool"

    records = memory.load_records("aapl", memory_dir=tmp_path)
    assert [r["scope"] for r in records] == ["ticker", "ticker",
                                             "general"]
    # general lessons reach other tickers too
    assert memory.load_notes("NVDA", memory_dir=tmp_path) == [
        NOTES[2]["note"]]


def test_default_memory_dir_is_config_not_cwd(tmp_path, monkeypatch):
    from finagent import config

    cwd = tmp_path / "notebooks"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    memory.save_notes("AAPL", ["a note"])
    assert (config.MEMORY_DIR / "AAPL.json").exists()
    assert not (cwd / "memory").exists()


def test_second_run_plans_with_first_runs_notes(monkeypatch, tmp_path):
    """Learning across runs: run 1 saves notes, run 2's planner prompt
    contains them."""
    prompts = []
    base = fake_json(scores=(6, 6, 6, 6))  # weak, so gaps get saved

    def complete_json(prompt, system=None, **kwargs):
        if system == planner.PLANNER_SYSTEM:
            prompts.append(prompt)
        if system == reflection.REFLECT_SYSTEM:
            return {"coverage_score": 6, "evidence_score": 6,
                    "recency_score": 6, "balance_score": 6,
                    "feedback": "thin", "gaps": ["no macro backdrop"]}
        return base(prompt, system, **kwargs)

    def create(messages, **kwargs):
        if len(messages) == 1:
            return tool_reply("get_price_summary", "get_company_info")
        return text_reply()

    monkeypatch.setattr(llm, "complete_json", complete_json)
    monkeypatch.setattr(llm, "create_message_cached", create)
    monkeypatch.setattr(research_agent.registry, "call",
                        lambda name, args=None: TOOL_DATA[name])

    run1 = research_agent.run("AAPL", memory_dir=tmp_path, save=False)
    run2 = research_agent.run("AAPL", memory_dir=tmp_path, save=False)

    assert run1["notes_loaded"] == []
    assert "no notes from earlier runs" in prompts[0]
    cats = {n["category"] for n in run1["notes_saved"]}
    assert {"finding", "lesson"} <= cats
    assert "Earlier report gap: no macro backdrop" in prompts[1]
    assert "Earlier run gathered no macro data" in prompts[1]
    assert len(run2["notes_loaded"]) == len(run1["notes_saved"])
    assert len(run1["drafts"]) == 1 + research_agent.MAX_REVISIONS


def test_rereflection_sees_previous_fixable_gaps(monkeypatch):
    prompts = []
    fixable = iter([["name the EPS period"], []])

    def complete_json(prompt, system=None, **kwargs):
        if system == reflection.REFLECT_SYSTEM:
            prompts.append(prompt)
            return {"coverage_score": 8, "evidence_score": 8,
                    "recency_score": 8, "balance_score": 8,
                    "fixable": next(fixable)}
        return {**REPORT, "outlook": "Revised."}

    monkeypatch.setattr(llm, "complete_json", complete_json)
    draft = research_agent._finish("AAPL", REPORT, TOOL_DATA)
    research_agent.reflect_and_revise("AAPL", draft, TOOL_DATA)
    assert "This is a revision" not in prompts[0]
    assert "- name the EPS period" in prompts[1]


def test_tool_lesson_names_the_substitute_for_failed_earnings():
    err = {"turn": 1, "tool": "get_earnings", "ok": False,
           "error": "RuntimeError: Alpha Vantage rate limit"}
    trace = [err, {"turn": 2, "tool": "get_financials", "ok": True}]
    run = {"coverage": {a: True for a in research_agent.AREAS},
           "errors": [err], "trace": trace}
    notes = research_agent.lessons("AAPL", {}, {"gaps": []}, run)
    assert notes["general"][0]["note"].endswith(
        "get_financials worked instead.")


def test_grounded_draft_beats_higher_scoring_ungrounded_one(monkeypatch):
    rounds = iter([(7, 7, 7, 7), (9, 9, 8, 8), (8, 8, 8, 8)])

    def complete_json(prompt, system=None, **kwargs):
        if system == reflection.REFLECT_SYSTEM:
            s = next(rounds)
            return {**dict(zip(("coverage_score", "evidence_score",
                                "recency_score", "balance_score"), s)),
                    "fixable": ["still something"]}
        return {**REPORT, "summary": "Margin is 99.9% now."}

    monkeypatch.setattr(llm, "complete_json", complete_json)
    draft = research_agent._finish("AAPL", REPORT, TOOL_DATA)
    out = research_agent.reflect_and_revise("AAPL", draft, TOOL_DATA)
    scores = [h["score"] for h in out["history"]]
    ungrounded = [bool(h["grounding"]["ungrounded"])
                  for h in out["history"]]
    assert all(h["revise"] for h in out["history"])
    best = out["best_index"]
    assert not ungrounded[best]
    assert scores[best] < max(s for s, u in zip(scores, ungrounded) if u)


def test_compact_facts_drop_free_cash_flow_for_a_bank():
    statements = {
        "quarterly": False,
        "income_statement": {"2025-12-31": {"Total Revenue": 181.8,
                                            "Net Income": 57.0}},
        "cash_flow": {"2025-12-31": {"Free Cash Flow": -147.8}},
    }
    bank = research_agent.compact_facts({
        "get_company_info": {"sector": "Financial Services"},
        "get_financials": statements})
    assert bank["get_financials"]["free_cash_flow"] is None
    tech = research_agent.compact_facts({
        "get_company_info": {"sector": "Technology"},
        "get_financials": statements})
    assert tech["get_financials"]["free_cash_flow"] == -147.8
