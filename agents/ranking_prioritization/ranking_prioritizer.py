from .prompt import build_prompt
from .heuristic import rank_insights_heuristically

try:
    from transformers import pipeline
except Exception:
    pipeline = None

from .parser import parse_output


class RankingPrioritizer:
    def __init__(self, model_name="gpt2", use_llm=False):
        self.use_llm = use_llm and pipeline is not None
        self.generator = None

        if self.use_llm:
            try:
                self.generator = pipeline("text-generation", model=model_name)
            except Exception:
                self.generator = None
                self.use_llm = False

    def llm_rank(self, insights):
        if not self.generator:
            return []

        prompt = build_prompt(insights)

        try:
            output = self.generator(
                prompt,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.3
            )[0]["generated_text"]

            parsed = parse_output(output)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except Exception:
            return []

        return []

    def heuristic_rank(self, insights):
        return rank_insights_heuristically(insights)

    def rank(self, insights):
        if self.use_llm:
            llm_result = self.llm_rank(insights)
            if llm_result:
                return llm_result

        return self.heuristic_rank(insights)