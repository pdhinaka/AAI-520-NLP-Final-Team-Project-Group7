"""The LLM wrappers reject settings the SDK doesn't support."""

import logging

import pytest

from finagent import llm
from finagent.agent import planner


def _no_client():
    raise AssertionError("the API client should never be reached")


@pytest.mark.parametrize("call", [
    lambda: llm.create_message(
        [{"role": "user", "content": "hi"}], temperature=0.2),
    lambda: llm.complete("hi", temperature=0.2),
    lambda: llm.complete_json("hi", temperature=0.2),
])
def test_temperature_raises_type_error(monkeypatch, call):
    # anthropic 1.9's messages.create has no temperature argument, so
    # the wrappers don't take one either: passing it must fail loudly.
    monkeypatch.setattr(llm, "_client", _no_client)
    with pytest.raises(TypeError, match="temperature"):
        call()


def test_planner_logs_when_it_falls_back(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(planner.llm, "complete_json", broken)
    with caplog.at_level(logging.WARNING, logger=planner.__name__):
        plan = planner.make_plan("AAPL")
    assert plan == planner.DEFAULT_PLAN
    assert "boom" in caplog.text
