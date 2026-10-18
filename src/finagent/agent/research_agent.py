"""Main agent loop: plan, act with tools, reflect, remember."""

import json
from datetime import datetime

from finagent import llm
from finagent.tools import registry

from . import memory, planner, reflection


def _fact_stats(facts: dict) -> list[str]:
    """Pull key facts out of the tool results into simple sentences."""
    stats: list[str] = []
    for key, value in facts.items():
        if key in {"ticker", "gathered_at"}:
            continue
        if not isinstance(value, dict):
            continue

        if "last_close" in value:
            stats.append(f"last close: {value['last_close']}")
        ret_1y = value.get("return_1y")
        if ret_1y is not None:
            stats.append(f"1-year return: {ret_1y:+.1%}")
        if "market_cap" in value:
            stats.append(f"market cap: {value['market_cap']}")
        if "revenue_growth" in value:
            stats.append(f"revenue growth: {value['revenue_growth']}")
        if "risk_factors" in value and value["risk_factors"]:
            joined = ", ".join(value["risk_factors"])
            stats.append(f"risk factors: {joined}")

    return stats[:5]


def _build_summary(ticker: str, facts: dict) -> str:
    """Build a concise summary from the facts we actually gathered."""
    facts_list = _fact_stats(facts)
    if not facts_list:
        return (
            f"{ticker} research is in progress; no tool facts have been "
            "collected yet."
        )

    return f"{ticker} research indicates: {'; '.join(facts_list)}."


def _build_thesis(ticker: str, facts: dict) -> str:
    """Create a simple investment thesis from the collected facts."""
    summary = _fact_stats(facts)
    if not summary:
        return (
            f"{ticker} has not yet generated a thesis because no tool "
            "facts were collected."
        )
    return (
        f"{ticker} remains constructive as long as the recent operating facts "
        f"hold up: {summary[0]}."
    )


def _build_upside(facts: dict) -> list[str]:
    upside = []
    for key, value in facts.items():
        if not isinstance(value, dict):
            continue
        ret_1y = value.get("return_1y")
        if ret_1y is not None:
            upside.append(
                f"Price momentum is positive at {ret_1y:+.1%} over the "
                "last year."
            )
        if "market_cap" in value:
            upside.append(
                "The market profile remains large, with market cap at "
                f"{value['market_cap']}."
            )
        if "revenue_growth" in value:
            upside.append(
                "Revenue growth remains supportive at "
                f"{value['revenue_growth']}."
            )
    return upside[:3] or [
        "The latest tool facts suggest a constructive setup."
    ]


def _build_risks(facts: dict) -> list[str]:
    risks = []
    for key, value in facts.items():
        if not isinstance(value, dict):
            continue
        if "risk_factors" in value and value["risk_factors"]:
            for item in value["risk_factors"]:
                risks.append(str(item))
    return risks or [
        "Macro or competitive pressure remains the main watch item."
    ]


def _build_catalysts(facts: dict) -> list[str]:
    catalysts = []
    for key, value in facts.items():
        if not isinstance(value, dict):
            continue
        if "last_close" in value:
            catalysts.append(
                "Recent price action indicates the market is still "
                "pricing in a live setup."
            )
        if "revenue_growth" in value:
            catalysts.append(
                "Operating momentum is a near-term catalyst if growth "
                "continues."
            )
        if "market_cap" in value:
            catalysts.append(
                "Large-cap scale gives the company flexibility to absorb "
                "short-term volatility."
            )
    return catalysts[:3] or [
        "The next earnings or product update is the main catalyst to "
        "watch."
    ]


def _build_outlook(facts: dict) -> str:
    for key, value in facts.items():
        if not isinstance(value, dict):
            continue
        ret_1y = value.get("return_1y")
        if ret_1y is not None:
            return "Moderately bullish" if ret_1y > 0 else "Neutral"
    return "Neutral"


def _build_recommendation(facts: dict) -> str:
    if any(
        isinstance(value, dict)
        and value.get("return_1y") is not None
        and value["return_1y"] > 0
        for value in facts.values()
    ):
        return "Moderate Buy"
    return "Hold"


def _has_sufficient_evidence(facts: dict) -> bool:
    """Stop the tool loop once we have enough facts to form a brief."""
    sources = (
        "get_price_summary",
        "get_company_info",
        "get_financials",
        "get_earnings",
        "get_news_digest",
        "get_macro_snapshot",
    )
    valid_sources = 0
    for name in sources:
        value = facts.get(name)
        if isinstance(value, dict) and "error" not in value:
            valid_sources += 1
    return valid_sources >= 2


def run(ticker: str) -> dict:
    """End-to-end research: plan → act → reflect → remember."""

    # Phase 1: Load prior knowledge
    notes = memory.load_notes(ticker)

    # Phase 2: Create plan
    plan = planner.make_plan(ticker, notes)
    print(f"Plan for {ticker}: {plan}")

    # Phase 3: Gather facts via agentic loop
    facts = {
        "ticker": ticker,
        "gathered_at": datetime.now().isoformat(),
    }

    # Built outside the f-string: a backslash inside {} is a
    # SyntaxError before Python 3.12.
    plan_text = "\n".join(f"- {step}" for step in plan)
    messages = [
        {
            "role": "user",
            "content": f"""Research {ticker}. Follow this plan:
            {plan_text}

            Available tools:
            {registry.describe()}

            Start by gathering key facts.""",
        }
    ]

    max_iterations = 5
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Call Claude with tools
        response = llm.create_message(
            messages=messages,
            tools=registry.tool_specs(),
        )

        # Check if Claude is done
        if response.stop_reason == "end_turn":
            print("Claude signaled completion.")
            break

        # Extract and execute tool calls
        tool_calls = [
            block for block in response.content
            if hasattr(block, "type") and block.type == "tool_use"
        ]

        if not tool_calls:
            print("No tool calls, stopping.")
            break

        # Add assistant response to history
        messages.append({
            "role": "assistant",
            "content": response.content,
        })

        # Execute tools
        tool_results = []
        for tool_call in tool_calls:
            print(f"Calling {tool_call.name}({tool_call.input})")
            result = registry.call(tool_call.name, tool_call.input)

            # Store in facts
            facts[tool_call.name] = result

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": (
                    result if isinstance(result, str)
                    else json.dumps(result, default=str)
                ),
            })

        # Add results back to conversation
        messages.append({
            "role": "user",
            "content": tool_results,
        })

        if _has_sufficient_evidence(facts):
            print("Enough evidence gathered; stopping early.")
            break

    # Phase 4: Draft report from the facts we actually gathered
    draft = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "summary": _build_summary(ticker, facts),
        "key_facts": _fact_stats(facts),
        "investment_thesis": _build_thesis(ticker, facts),
        "recommendation": _build_recommendation(facts),
        "upside": _build_upside(facts),
        "risks": _build_risks(facts),
        "catalysts": _build_catalysts(facts),
        "outlook": _build_outlook(facts),
    }

    # Phase 5: Reflect
    score_result = reflection.reflect(draft)
    draft["reflection_score"] = score_result["score"]
    draft["reflection_feedback"] = score_result["feedback"]

    # Phase 6: Save to memory
    learnings = [
        f"Plan had {len(plan)} steps, completed in {max_iterations} "
        "iterations",
        f"Score: {score_result['score']}/10",
    ]
    learnings.extend(score_result["gaps"][:2])
    memory.save_notes(ticker, learnings)

    return draft
