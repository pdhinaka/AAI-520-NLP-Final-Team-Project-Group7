"""Main agent loop: plan, act with tools, reflect, remember."""

from finagent import llm
from finagent.tools import registry
from . import planner, reflection, memory
import json
from datetime import datetime

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
        "gathered_at": datetime.now().isoformat()
    }
    
    messages = [
        {
            "role": "user",
            "content": f"""Research {ticker}. Follow this plan:
            {chr(10).join(f"- {step}" for step in plan)}

            Available tools:
            {registry.describe()}

            Start by gathering key facts."""
        }
    ]
    
    max_iterations = 5
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Call Claude with tools
        response = llm.create_message(
            messages=messages,
            tools=registry.tool_specs()
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
            "content": response.content
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
                "content": json.dumps(result) if not isinstance(result, str) else result
            })
        
        # Add results back to conversation
        messages.append({
            "role": "user",
            "content": tool_results
        })
    
    # Phase 4: Draft report
    draft = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "summary": facts.get("summary", "Research completed."),
        "key_facts": facts,
        "upside": ["Based on gathered data"],
        "risks": ["Based on gathered data"],
        "catalysts": [],
        "outlook": "Hold pending full analysis"
    }
    
    # Phase 5: Reflect
    score_result = reflection.reflect(draft)
    draft["reflection_score"] = score_result["score"]
    draft["reflection_feedback"] = score_result["feedback"]
    
    # Phase 6: Save to memory
    learnings = [
        f"Plan had {len(plan)} steps, completed in {max_iterations} iterations",
        f"Score: {score_result['score']}/10",
    ]
    learnings.extend(score_result["gaps"][:2])  # Top gaps
    memory.save_notes(ticker, learnings)
    
    return draft
