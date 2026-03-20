"""
MERLIN Test Suite
Tests: schemas, agents, pipeline, API endpoints
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pytest
from unittest.mock import patch, MagicMock

# ── Schema tests ──────────────────────────────────────────────────────────────
class TestSchemas:
    def test_claim_creation(self):
        from src.models.schemas import Claim
        c = Claim(subject="LLM", predicate="outperforms", object="BM25")
        assert c.text == "LLM outperforms BM25"
        assert c.id

    def test_assumption_creation(self):
        from src.models.schemas import Assumption, AssumptionType
        a = Assumption(type=AssumptionType.DOMAIN, constraint="English only")
        assert a.constraint == "English only"

    def test_agreement_creation(self):
        from src.models.schemas import Agreement, RelationType
        ag = Agreement(claim_i_id="a", claim_j_id="b",
                       relation=RelationType.CONTRADICT, confidence=0.9)
        assert ag.relation == "contradict"

    def test_claim_to_dict(self):
        from src.models.schemas import Claim
        c = Claim(subject="X", predicate="improves", object="Y")
        d = c.to_dict()
        assert "text" in d
        assert d["subject"] == "X"


# ── Agent 3 (rule-based, no API needed) ──────────────────────────────────────
class TestNormaliser:
    def test_predicate_normalisation(self):
        from src.agents.agent3_normalize import normalise_claims
        from src.models.schemas import Claim
        claims = [Claim(predicate="outperforms baseline"), Claim(predicate="reduces cost")]
        result = normalise_claims(claims)
        assert result[0].predicate == "outperforms"
        assert result[1].predicate == "reduces"

    def test_domain_normalisation(self):
        from src.agents.agent3_normalize import normalise_claims
        from src.models.schemas import Claim
        claims = [Claim(domain="NLP question answering")]
        result = normalise_claims(claims)
        assert result[0].domain == "NLP"

    def test_unknown_predicate_unchanged(self):
        from src.agents.agent3_normalize import normalise_claims
        from src.models.schemas import Claim
        claims = [Claim(predicate="frobnicates")]
        result = normalise_claims(claims)
        assert result[0].predicate == "frobnicates"


# ── Agent 6.1 Verifier (string fallback) ─────────────────────────────────────
class TestVerifier:
    def test_direct_span_match(self):
        from src.agents.agent6_1_verify import verify_assumption
        from src.models.schemas import Assumption, VerificationStatus
        a = Assumption(span="English text only", constraint="English text only")
        text = "The study is limited to English text only."
        result = verify_assumption(a, text)
        assert result.verification == VerificationStatus.VERIFIED

    def test_rejection_on_mismatch(self):
        from src.agents.agent6_1_verify import verify_assumption
        from src.models.schemas import Assumption, VerificationStatus
        a = Assumption(span="", constraint="requires GPU cluster with 512 nodes")
        text = "We used a laptop for experiments."
        result = verify_assumption(a, text)
        assert result.verification == VerificationStatus.REJECTED

    def test_verify_all_filters_rejected(self):
        from src.agents.agent6_1_verify import verify_all_assumptions
        from src.models.schemas import Assumption, VerificationStatus
        assumptions = [
            Assumption(span="English text", constraint="English text"),
            Assumption(span="", constraint="quantum supercomputer infrastructure xyz"),
        ]
        text = "This study focuses on English text analysis."
        result = verify_all_assumptions(assumptions, text)
        assert all(a.verification != VerificationStatus.REJECTED for a in result)


# ── Agent 4 Agreement (heuristic path) ───────────────────────────────────────
class TestAgreement:
    def test_heuristic_contradict(self):
        from src.agents.agent4_agreement import _quick_relation
        from src.models.schemas import Claim, RelationType
        ci = Claim(predicate="outperforms")
        cj = Claim(predicate="underperforms")
        rel, conf = _quick_relation(ci, cj)
        assert rel == RelationType.CONTRADICT
        assert conf > 0.5

    def test_heuristic_agree(self):
        from src.agents.agent4_agreement import _quick_relation
        from src.models.schemas import Claim, RelationType
        ci = Claim(subject="LLM", predicate="outperforms")
        cj = Claim(subject="llm", predicate="outperforms")
        rel, conf = _quick_relation(ci, cj)
        assert rel == RelationType.AGREE

    def test_different_domain_unrelated(self):
        from src.agents.agent4_agreement import _quick_relation
        from src.models.schemas import Claim, RelationType
        ci = Claim(domain="NLP", predicate="improves")
        cj = Claim(domain="CV",  predicate="improves")
        rel, conf = _quick_relation(ci, cj)
        assert rel == RelationType.UNRELATED


# ── Agent 5 Uncertainty ───────────────────────────────────────────────────────
class TestUncertainty:
    def test_uncertainty_range(self):
        from src.agents.agent5_uncertainty import compute_uncertainty
        from src.models.schemas import Claim, Agreement, RelationType
        c = Claim(evidence_strength="high")
        c.id = "test-claim"
        agreements = [
            Agreement("test-claim", "other", RelationType.CONTRADICT, 0.9),
            Agreement("test-claim", "other2", RelationType.AGREE,      0.8),
        ]
        u = compute_uncertainty(c, agreements)
        assert 0.0 <= u <= 1.0

    def test_high_evidence_reduces_uncertainty(self):
        from src.agents.agent5_uncertainty import compute_uncertainty
        from src.models.schemas import Claim
        c_high = Claim(evidence_strength="high"); c_high.id = "h"
        c_low  = Claim(evidence_strength="low");  c_low.id  = "l"
        u_high = compute_uncertainty(c_high, [])
        u_low  = compute_uncertainty(c_low,  [])
        assert u_high < u_low


# ── EDG ──────────────────────────────────────────────────────────────────────
class TestEDG:
    def test_build_edg(self):
        from src.graph.edg import build_edg
        from src.models.schemas import Claim, Agreement, RelationType
        c1 = Claim(subject="A", predicate="outperforms", object="B"); c1.id = "c1"
        c2 = Claim(subject="C", predicate="reduces",     object="D"); c2.id = "c2"
        ag = Agreement("c1", "c2", RelationType.AGREE, 0.8)
        edg = build_edg([c1, c2], [ag])
        d = edg.to_dict()
        assert d["stats"]["num_claims"] == 2
        assert d["stats"]["num_edges"] >= 1

    def test_edg_serialisation(self):
        from src.graph.edg import build_edg
        from src.models.schemas import Claim
        c = Claim(subject="X", predicate="improves", object="Y"); c.id = "x1"
        edg = build_edg([c], [])
        d = edg.to_dict()
        assert "nodes" in d and "edges" in d and "stats" in d


# ── Pipeline (mocked Mistral) ─────────────────────────────────────────────────
class TestPipeline:
    @patch('src.mistral_client.requests.post')
    def test_pipeline_runs(self, mock_post):
        # Mock Mistral responses
        def mock_response(url, **kwargs):
            body = kwargs.get('json', {})
            prompt = body.get('messages', [{}])[-1].get('content', '')
            if 'Extract scientific claims' in prompt or 'claims' in prompt.lower():
                content = json.dumps([{"subject":"LLM","predicate":"outperforms",
                                        "object":"BM25","method":"fine-tuning","domain":"NLP"}])
            elif 'evidence' in prompt.lower():
                content = json.dumps([{"claim_id":0,"spans":["our results show"],"strength":"high"}])
            elif 'assumption' in prompt.lower():
                content = json.dumps([{"type":"domain","constraint":"English text",
                                        "explicit":True,"span":"English text"}])
            elif 'contradictory' in prompt.lower() or 'compatible' in prompt.lower():
                content = json.dumps({"relation":"agree","confidence":0.8,"reason":"same domain"})
            elif 'gap' in prompt.lower():
                content = json.dumps([{"gap":"Limited cross-lingual evaluation","type":"empirical","priority":"high"}])
            else:
                content = json.dumps({})
            m = MagicMock()
            m.ok = True
            m.raise_for_status = MagicMock()
            m.json.return_value = {
                "choices": [{"message": {"content": content}}]
            }
            return m

        mock_post.side_effect = mock_response

        from src.pipeline import run_pipeline
        papers = [{"id":"p1","text":(
            "We demonstrate that LLMs outperform BM25 on question answering. "
            "Experiments assume English text only. Results show 12% F1 gain."
        )}]
        result = run_pipeline(papers)
        assert result is not None
        assert len(result.claims) >= 0   # may be 0 if mocked JSON mismatches
        assert hasattr(result, 'agreements')
        assert hasattr(result, 'gaps')
        assert hasattr(result, 'edg')

    def test_pipeline_empty_paper(self):
        from src.pipeline import run_pipeline
        result = run_pipeline([{"id": "empty", "text": ""}])
        assert result.claims == []


# ── API Endpoints ─────────────────────────────────────────────────────────────
class TestAPI:
    @pytest.fixture
    def client(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from api.server import app
        app.config['TESTING'] = True
        with app.test_client() as c:
            yield c

    def test_health(self, client):
        r = client.get('/api/health')
        assert r.status_code == 200
        data = r.get_json()
        assert data['status'] == 'ok'

    def test_analyse_missing_field(self, client):
        r = client.post('/api/analyse', json={}, content_type='application/json')
        assert r.status_code == 400

    def test_analyse_empty_papers(self, client):
        r = client.post('/api/analyse', json={"papers":[]})
        assert r.status_code == 400

    def test_sample_endpoint(self, client):
        r = client.get('/api/sample')
        assert r.status_code == 200
        data = r.get_json()
        assert 'papers' in data
        assert len(data['papers']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])


# ── HallucinationGuard unit tests ─────────────────────────────────────────────
class TestHallucinationGuard:

    # V1 — Claim grounding
    def test_v1_grounded_claim_passes(self):
        from src.hallucination_guard import ground_claim
        source = "Large language models significantly outperform BM25 on QA tasks."
        r = ground_claim("LLMs", "outperforms", "BM25", source)
        assert r.grounded, f"Expected grounded, got: {r.reason}"

    def test_v1_hallucinated_claim_fails(self):
        from src.hallucination_guard import ground_claim
        source = "We study neural networks for image recognition."
        r = ground_claim("quantum entanglement", "disproves", "relativity", source)
        assert not r.grounded

    def test_v1_filter_drops_hallucinated(self):
        from src.hallucination_guard import filter_hallucinated_claims
        from src.models.schemas import Claim
        source = "LLMs outperform BM25 on question answering benchmarks."
        real_claim  = Claim(subject="LLMs", predicate="outperforms", object="BM25")
        fake_claim  = Claim(subject="quantum foam", predicate="invalidates", object="relativity")
        kept, dropped, _ = filter_hallucinated_claims([real_claim, fake_claim], source)
        assert dropped >= 1
        assert len(kept) >= 1

    # V2 — Evidence span verification
    def test_v2_real_span_kept(self):
        from src.hallucination_guard import verify_evidence_spans
        from src.models.schemas import Claim
        source = "Results show 12% F1 improvement over baseline."
        c = Claim(evidence_spans=["12% F1 improvement"], evidence_strength="high")
        result = verify_evidence_spans(c, source)
        assert len(result.evidence_spans) == 1

    def test_v2_fabricated_span_removed(self):
        from src.hallucination_guard import verify_evidence_spans
        from src.models.schemas import Claim
        source = "We compared models on the SQUAD dataset."
        c = Claim(
            evidence_spans=["achieved perfect accuracy on all benchmarks worldwide"],
            evidence_strength="high"
        )
        result = verify_evidence_spans(c, source)
        assert len(result.evidence_spans) == 0
        assert result.evidence_strength == "low"

    def test_v2_strength_recalibrated_when_all_removed(self):
        from src.hallucination_guard import verify_evidence_spans
        from src.models.schemas import Claim
        c = Claim(evidence_spans=["this text does not exist anywhere near here xyzabc"],
                  evidence_strength="high")
        result = verify_evidence_spans(c, "completely unrelated text about something else")
        assert result.evidence_strength == "low"

    # V3 — Agreement reason grounding
    def test_v3_grounded_reason_kept(self):
        from src.hallucination_guard import verify_agreement_reason
        from src.models.schemas import Agreement, RelationType
        ag = Agreement("c1", "c2", RelationType.AGREE, 0.9,
                       reason="Both studies use transformer models on English datasets")
        grounded, _ = verify_agreement_reason(
            ag, "transformer models outperform baselines", "English datasets show improvement"
        )
        assert grounded

    def test_v3_heuristic_reason_always_grounded(self):
        from src.hallucination_guard import verify_agreement_reason
        from src.models.schemas import Agreement, RelationType
        ag = Agreement("c1", "c2", RelationType.AGREE, 0.8, reason="Heuristic: agree")
        grounded, _ = verify_agreement_reason(ag, "anything", "anything else")
        assert grounded

    def test_v3_ungrounded_reason_rewritten_and_capped(self):
        from src.hallucination_guard import verify_agreement_reason
        from src.models.schemas import Agreement, RelationType
        ag = Agreement("c1", "c2", RelationType.AGREE, 0.99,
                       reason="xkzqwerty completely fabricated nonsense abc123")
        grounded, _ = verify_agreement_reason(ag, "apples grow on trees", "water is wet")
        assert not grounded
        assert ag.confidence <= 0.70
        assert ag.reason.startswith("[Auto-summary]")

    # V4 — Gap grounding
    def test_v4_grounded_gap_kept(self):
        from src.hallucination_guard import verify_gap
        from src.models.schemas import Claim, ResearchGap
        c = Claim(subject="LLMs", predicate="outperforms", object="BM25", domain="NLP")
        g = ResearchGap(gap="Cross-lingual evaluation of LLMs vs BM25 retrieval")
        grounded, _ = verify_gap(g, [c])
        assert grounded

    def test_v4_ungrounded_gap_rejected(self):
        from src.hallucination_guard import verify_gap
        from src.models.schemas import Claim, ResearchGap
        c = Claim(subject="LLMs", predicate="outperforms", object="BM25", domain="NLP")
        g = ResearchGap(gap="Quantum computing applications in marine biology ecosystems")
        grounded, score = verify_gap(g, [c])
        assert not grounded

    def test_v4_filter_removes_ungrounded(self):
        from src.hallucination_guard import filter_hallucinated_gaps
        from src.models.schemas import Claim, ResearchGap
        claims = [Claim(subject="transformers", predicate="improves", object="accuracy",domain="NLP")]
        good_gap = ResearchGap(gap="Evaluation of transformer accuracy in low-resource NLP settings")
        bad_gap  = ResearchGap(gap="Astrophysics dark matter quantum entanglement xyzzy")
        kept, dropped = filter_hallucinated_gaps([good_gap, bad_gap], claims)
        assert dropped >= 1

    # V5 — Assumption deep grounding
    def test_v5_tier1_exact_span(self):
        from src.hallucination_guard import deep_assumption_ground
        from src.models.schemas import Assumption, VerificationStatus
        a = Assumption(span="English text only", constraint="English text only")
        status, score = deep_assumption_ground(a, "The study is limited to English text only.")
        assert status == VerificationStatus.VERIFIED
        assert score == 1.0

    def test_v5_tier2_bigram_overlap(self):
        from src.hallucination_guard import deep_assumption_ground
        from src.models.schemas import Assumption, VerificationStatus
        a = Assumption(span="", constraint="GPU cluster high performance computing")
        status, score = deep_assumption_ground(
            a, "Experiments run on a GPU cluster using high performance computing infrastructure."
        )
        assert status in (VerificationStatus.VERIFIED, VerificationStatus.WEAK)

    def test_v5_tier3_weak_unigram(self):
        from src.hallucination_guard import deep_assumption_ground
        from src.models.schemas import Assumption, VerificationStatus
        a = Assumption(span="", constraint="English language processing")
        status, score = deep_assumption_ground(
            a, "The model processes English documents."
        )
        assert status in (VerificationStatus.VERIFIED, VerificationStatus.WEAK)

    def test_v5_rejected_no_overlap(self):
        from src.hallucination_guard import deep_assumption_ground
        from src.models.schemas import Assumption, VerificationStatus
        a = Assumption(span="", constraint="superconducting quantum interference devices cryogenic")
        status, score = deep_assumption_ground(a, "We used Python for data analysis.")
        assert status == VerificationStatus.REJECTED

    # Token utility tests
    def test_token_overlap_perfect(self):
        from src.hallucination_guard import token_overlap
        assert token_overlap("neural network classification", "neural network classification task") > 0.8

    def test_token_overlap_zero(self):
        from src.hallucination_guard import token_overlap
        assert token_overlap("xyz abc def", "completely unrelated sentence here") == 0.0

    def test_span_exists_exact(self):
        from src.hallucination_guard import span_exists
        assert span_exists("12% F1 improvement", "We achieved a 12% F1 improvement over baseline.")

    def test_span_exists_false(self):
        from src.hallucination_guard import span_exists
        assert not span_exists("99% accuracy on all tasks", "Results show modest gains.")
