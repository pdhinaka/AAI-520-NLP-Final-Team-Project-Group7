"""Plans research steps for a ticker."""

from finagent import llm
import json

def make_plan(ticker: str, notes: list[str] | None = None) -> list[str]:
    """Create a research plan using Claude."""
    
    context = ""
    if notes:
        context = f"\n\nPrior knowledge from previous runs:\n" + "\n".join(f"- {n}" for n in notes)
    
    prompt = f"""You are an investment research planner. Create a concise research plan for {ticker}.
    The plan should have 5-7 steps covering:
        - Price and momentum
        - Company fundamentals
        - Recent news/sentiment
        - Macro context
        - Catalysts and risks
    Format as a JSON array of strings (steps).{context}"""

    system = """You are an expert investment researcher. Plans should be specific and actionable."""
    
    result = llm.complete_json(prompt, system=system)
    
    # Result might be {"steps": [...]} or just [...]
    if isinstance(result, dict) and "steps" in result:
        return result["steps"]
    elif isinstance(result, list):
        return result
    else:
        return str(result).split("\n")
