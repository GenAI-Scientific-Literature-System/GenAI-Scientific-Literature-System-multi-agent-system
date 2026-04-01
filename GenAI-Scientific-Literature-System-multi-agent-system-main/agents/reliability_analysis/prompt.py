def build_prompt(paper_text):
    return f"""
You are an expert scientific reviewer.

Evaluate the reliability of this research paper using:
- Methodology clarity
- Dataset size
- Publication quality
- Recency
- Evidence strength

Return ONLY JSON:
{{
  "reliability_score": float (0 to 1),
  "confidence": "Low" | "Medium" | "High",
  "justification": [list of reasons]
}}

Paper:
{paper_text}
"""