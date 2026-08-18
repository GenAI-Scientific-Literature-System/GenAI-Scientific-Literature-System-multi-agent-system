from src.glas_med import attach_provenance, study_reliability, uncertainty_priorities, weighted_agreements
from src.models.schemas import Claim


def _claim(subject, predicate, object_, paper_id):
    return Claim(
        id=paper_id,
        subject=subject,
        predicate=predicate,
        object=object_,
        domain="adults with type 2 diabetes",
        paper_id=paper_id,
        extraction_confidence=1.0,
    )


def test_reliability_uses_paper_eight_factor_adjustments():
    report = study_reliability(
        "A double-blind randomized controlled trial enrolled n=200 participants. "
        "The protocol was pre-registered at ClinicalTrials.gov NCT12345678 and reported 95% confidence intervals."
    )

    assert report["tier"] == 2
    assert report["score"] == 1.0
    assert report["factors"]["blinding_randomisation"] == 0.12
    assert report["factors"]["preregistration"] == 0.08


def test_low_reliability_study_is_quarantined():
    report = study_reliability("A case report described one patient.")

    assert report["tier"] == 5
    assert report["quarantined"] is True


def test_case_series_uses_case_based_evidence_tier():
    from src.glas_med import evidence_tier

    assert evidence_tier("A case series described outcomes for 20 patients.") == 4


def test_pico_overlap_requires_meaningful_intervention_and_outcome():
    from src.glas_med import _pico_overlap

    generic_a = Claim(subject="adults with obesity", predicate="reduce", object="obesity")
    generic_a.pico = {"intervention": "adults with obesity", "outcome": "reduce obesity"}
    generic_b = Claim(subject="adolescent obesity", predicate="associated with", object="type 2 diabetes")
    generic_b.pico = {"intervention": "adolescent obesity", "outcome": "type 2 diabetes"}
    assert _pico_overlap(generic_a, generic_b) is False

    semaglutide_a = Claim(subject="semaglutide", predicate="reduces", object="body weight")
    semaglutide_a.pico = {"intervention": "semaglutide", "outcome": "body weight"}
    semaglutide_b = Claim(subject="semaglutide", predicate="improves", object="weight loss")
    semaglutide_b.pico = {"intervention": "semaglutide", "outcome": "weight loss"}
    assert _pico_overlap(semaglutide_a, semaglutide_b) is True


def test_normalisation_preserves_harmful_risk_direction():
    from src.agents.agent3_normalize import normalise_claims

    claim = Claim(predicate="increases", object="risk of bleeding")
    assert normalise_claims([claim])[0].predicate == "increases_risk"


def test_pico_weighted_agreement_and_uncertainty_priority():
    positive_1 = _claim("Metformin", "reduces", "cardiovascular events", "p1")
    positive_2 = _claim("Metformin", "improves", "cardiovascular events", "p2")
    negative = _claim("Metformin", "increases risk", "cardiovascular events", "p3")
    papers = [
        {"id": "p1", "text": "randomized controlled trial n=200", "citation_count": 100},
        {"id": "p2", "text": "randomized controlled trial n=200", "citation_count": 100},
        {"id": "p3", "text": "cohort study n=200", "citation_count": 100},
    ]
    claims = []
    for claim, paper in zip([positive_1, positive_2, negative], papers):
        claims.extend(attach_provenance([claim], paper))

    agreements = weighted_agreements(claims)

    assert agreements
    assert {agreement.verdict for agreement in agreements} == {"Agree"}
    assert agreements[0].weighted_agreement == 0.727

    # A disagreement becomes an uncertainty-priority gap with the paper's
    # citation/reliability weighted formula.
    negative.study_reliability = 1.0
    agreements = weighted_agreements(claims)
    gaps = uncertainty_priorities(claims, agreements)
    assert agreements[0].verdict == "Partial Agreement"
    assert gaps and gaps[0].gap_signals["uncertainty_impact"] > 0
