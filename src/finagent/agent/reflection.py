"""Self-reflection on the agent's own output."""

from finagent import llm

def reflect(report: dict) -> dict:
    """Score the report against a rubric."""
    
    # Prepare report summary for Claude
    report_text = f"""
        Ticker: {report.get('ticker', 'Unknown')}
        Summary: {report.get('summary', 'N/A')}
        Key Facts: {report.get('key_facts', [])}
        Upside: {report.get('upside', [])}
        Risks: {report.get('risks', [])}
        Catalysts: {report.get('catalysts', [])}
    """
            
    prompt = f"""Score this investment research report on 4 criteria (0-10 each):
        1. **Coverage**: Does it address company, market, and macro factors?
        2. **Evidence**: Are claims backed by data (specific numbers, sources)?
        3. **Recency**: Are facts fresh (last quarter or month)?
        4. **Balance**: Does it present both upside and downside?

        Report:
        {report_text}

        Respond as JSON:
        {{
            "coverage_score": <int>,
            "evidence_score": <int>,
            "recency_score": <int>,
            "balance_score": <int>,
            "overall_score": <int 0-40, then divide by 4>,
            "feedback": "<string with main observations>",
            "gaps": ["<gap 1>", "<gap 2>", ...]
        }}"""

    system = "You are a critical evaluator of investment research. Be constructive but honest."
    
    result = llm.complete_json(prompt, system=system)
    
    # Compute overall from components if not provided
    if "overall_score" not in result or result["overall_score"] == 0:
        scores = [
            result.get("coverage_score", 5),
            result.get("evidence_score", 5),
            result.get("recency_score", 5),
            result.get("balance_score", 5)
        ]
        result["overall_score"] = sum(scores) / len(scores)
    
    return {
        "score": result.get("overall_score", 5.0),
        "feedback": result.get("feedback", ""),
        "gaps": result.get("gaps", [])
    }
