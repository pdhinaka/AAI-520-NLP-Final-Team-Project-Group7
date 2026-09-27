"""Smoke test: every module imports."""

import importlib

MODULES = [
    "finagent.tools.market_data",
    "finagent.tools.news",
    "finagent.tools.macro",
    "finagent.tools.filings",
    "finagent.workflows.prompt_chain",
    "finagent.workflows.routing",
    "finagent.workflows.evaluator_optimizer",
    "finagent.agent.planner",
    "finagent.agent.reflection",
    "finagent.agent.memory",
    "finagent.agent.research_agent",
]


def test_imports():
    for name in MODULES:
        importlib.import_module(name)
