import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ranking_prioritization import RankingPrioritizer


def test_agent6():
    insights = [
        {
            "paper_id": "P1",
            "title": "Deep Learning for X-ray Diagnosis",
            "claim": "CNN improves accuracy",
            "evidence_count": 8,
            "reliability_score": 9,
            "agreement_score": 8,
            "conflict_score": 2,
            "novelty_score": 6
        },
        {
            "paper_id": "P2",
            "title": "Transfer Learning",
            "claim": "Transfer learning reduces data requirements",
            "evidence_count": 5,
            "reliability_score": 8,
            "agreement_score": 7,
            "conflict_score": 3,
            "novelty_score": 7
        },
        {
            "paper_id": "P3",
            "title": "Attention Models",
            "claim": "Attention improves interpretability",
            "evidence_count": 4,
            "reliability_score": 7,
            "agreement_score": 5,
            "conflict_score": 5,
            "novelty_score": 9
        }
    ]

    agent = RankingPrioritizer(use_llm=False)
    result = agent.rank(insights)

    print("\n=== Agent 6 Output ===")
    for r in result:
        print(r)


if __name__ == "__main__":
    test_agent6()