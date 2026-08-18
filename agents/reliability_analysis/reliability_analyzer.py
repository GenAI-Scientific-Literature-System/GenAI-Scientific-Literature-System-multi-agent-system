from transformers import pipeline
from .prompt import build_prompt
from .heuristic import compute_heuristic
from .parser import parse_output


class ReliabilityAnalyzer:

    def __init__(self, model_name="gpt2"):
        try:
            self.generator = pipeline("text-generation", model=model_name, local_files_only=True)
        except Exception:
            self.generator = None

    def llm_evaluate(self, paper_text):
        prompt = build_prompt(paper_text)

        response = self.generator(
            prompt,
            max_length=300,
            do_sample=True
        )[0]["generated_text"]

        return parse_output(response)

    def evaluate(self, paper_text, metadata):
        heuristic_score = compute_heuristic(metadata)

        llm_result = None
        if self.generator is not None:
            try:
                llm_result = self.llm_evaluate(paper_text)
            except Exception:
                llm_result = None

        if llm_result is None:
            return {
                "final_score": round(heuristic_score, 2),
                "llm_score": None,
                "heuristic_score": heuristic_score,
                "confidence": 0.0,
                "justification": "Heuristic reliability fallback; language model unavailable."
            }

        final_score = (
            0.7 * llm_result["reliability_score"] +
            0.3 * heuristic_score
        )

        return {
            "final_score": round(final_score, 2),
            "llm_score": llm_result["reliability_score"],
            "heuristic_score": heuristic_score,
            "confidence": llm_result["confidence"],
            "justification": llm_result["justification"]
        }
