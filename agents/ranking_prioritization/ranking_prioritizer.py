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

    def _build_insights_from_components(self, claims, reliability, agreements, uncertainties):
        reliability_by_paper = {
            (r.get("paper_id") or ""): r.get("reliability", {})
            for r in (reliability or [])
        }
        agreement_by_paper = {
            (a.get("paper_id") or ""): a
            for a in (agreements or [])
        }
        uncertainty_by_paper = {
            (u.get("paper_id") or ""): u
            for u in (uncertainties or [])
        }

        insights = []
        for c in claims or []:
            paper_id = c.get("focal_paper_id") or c.get("paper_id") or ""
            rel = reliability_by_paper.get(paper_id, {}) or {}
            ag = agreement_by_paper.get(paper_id, {}) or {}
            un = uncertainty_by_paper.get(paper_id, {}) or {}

            supporting = c.get("supporting", []) or []
            contradicting = c.get("contradicting", []) or []
            inconclusive = c.get("inconclusive", []) or []

            insights.append(
                {
                    "paper_id": paper_id,
                    "title": c.get("focal_paper_title") or c.get("title", ""),
                    "claim": c.get("claim", ""),
                    "reliability_score": rel.get("score", rel.get("reliability_score", 5)),
                    "evidence_count": len(supporting) + len(contradicting) + len(inconclusive),
                    "agreement_score": ag.get("agreement_score", 0),
                    "conflict_score": ag.get("conflict_score", 0),
                    "novelty_score": max(0, 10 - un.get("uncertainty_score", 5)),
                }
            )

        return insights

    def rank(self, insights=None, *, claims=None, evidence=None, reliability=None, agreements=None, uncertainties=None):
        if insights is None:
            insights = self._build_insights_from_components(
                claims=claims or evidence or [],
                reliability=reliability or [],
                agreements=agreements or [],
                uncertainties=uncertainties or [],
            )

        if self.use_llm:
            llm_result = self.llm_rank(insights)
            if llm_result:
                return llm_result

        return self.heuristic_rank(insights)