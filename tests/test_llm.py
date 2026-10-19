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


class _FakeMessages:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        from anthropic.types import Message

        self.calls += 1
        return Message.model_validate({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": kwargs["model"],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "content": [
                {"type": "text", "text": "Checking the price."},
                {"type": "tool_use", "id": "tu_1",
                 "name": "get_price_summary",
                 "input": {"ticker": "AAPL"}},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


MESSAGES = [{"role": "user", "content": "Research AAPL"}]


def test_create_message_cached_replays_without_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(llm, "_client", lambda: client)
    monkeypatch.setitem(llm.USAGE, "calls", 0)

    first = llm.create_message_cached(MESSAGES, tools=[])
    second = llm.create_message_cached(MESSAGES, tools=[])

    assert client.messages.calls == 1
    assert llm.USAGE["calls"] == 1  # only the live call is counted
    assert first == second
    assert first["stop_reason"] == "tool_use"
    assert first["content"][1] == {
        "type": "tool_use", "id": "tu_1", "name": "get_price_summary",
        "input": {"ticker": "AAPL"},
    }
    assert llm.text_of(first) == "Checking the price."


def test_create_message_cached_accepts_sdk_blocks(monkeypatch):
    # An assistant turn appended as SDK objects keys the same as dicts.
    client = _FakeClient()
    monkeypatch.setattr(llm, "_client", lambda: client)
    msg = client.messages.create(model="m")
    as_objects = MESSAGES + [
        {"role": "assistant", "content": msg.content}]
    as_dicts = MESSAGES + [
        {"role": "assistant",
         "content": [llm._block_dict(b) for b in msg.content]}]

    llm.create_message_cached(as_objects)
    llm.create_message_cached(as_dicts)
    assert client.messages.calls == 2  # one direct, one cache miss


def test_create_message_cached_offline_miss_raises(monkeypatch):
    from finagent import config
    from finagent.cache import CacheMissError

    monkeypatch.setattr(config, "CACHE_MODE", "offline")
    monkeypatch.setattr(llm, "_client", _no_client)
    with pytest.raises(CacheMissError):
        llm.create_message_cached(MESSAGES)
