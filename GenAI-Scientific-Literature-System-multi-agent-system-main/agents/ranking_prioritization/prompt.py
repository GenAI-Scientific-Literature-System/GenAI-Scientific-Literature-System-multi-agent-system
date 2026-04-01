def build_prompt(insights):
    formatted = []

    for i, insight in enumerate(insights, start=1):
        formatted.append(
            f"""
Insight {i}:
Title: {insight.get('title', '')}
Claim: {insight.get('claim', '')}
Evidence Count: {insight.get('evidence_count', 0)}
Reliability Score: {insight.get('reliability_score', 0)}
Agreement Score: {insight.get('agreement_score', 0)}
Conflict Score: {insight.get('conflict_score', 0)}
Novelty Score: {insight.get('novelty_score', 0)}
Paper ID: {insight.get('paper_id', '')}
"""
        )

    joined = "\n".join(formatted)

    return f"""
You are an AI research analyst.

Your task is to rank the following scientific insights across papers based on:
1. Reliability of the source paper
2. Strength and quantity of evidence
3. Agreement with other papers
4. Conflict or contradiction level
5. Novelty / importance of the insight

Return ONLY a valid Python-style list of dictionaries in this exact format:

[
  {{
    "rank": 1,
    "paper_id": "...",
    "title": "...",
    "claim": "...",
    "priority_score": 0.0,
    "reason": "..."
  }}
]

Insights:
{joined}
"""