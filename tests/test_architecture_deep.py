import os
import pytest
from services.common.medical_data_layer import MedicalDataLayer
from services.common.biobert import BioBERTClaimExtractor
from services.mas_orchestrator.main import checkpoint_stage, _IN_MEMORY_CHECKPOINTS


def test_faiss_medical_data_layer_indexing():
    layer = MedicalDataLayer(dimensions=64)
    doc_id = layer.upsert_document({
        "id": "doc_1",
        "title": "Metformin and AMPK activation in diabetic models",
        "abstract": "Metformin activates AMPK and reduces gluconeogenesis."
    })
    
    assert doc_id == "doc_1"
    
    # Test FAISS search
    results = layer.search("AMPK activation gluconeogenesis", limit=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc_1"
    assert "similarity" in results[0]


def test_biobert_extraction_and_entities():
    extractor = BioBERTClaimExtractor()
    text = "In patients with diabetes, empagliflozin significantly reduces cardiovascular mortality."
    claims = extractor.extract(text, paper_id="empa_01")
    
    assert len(claims) >= 1
    assert claims[0]["predicate"] in ["reduces", "significantly reduces"]
    assert "empagliflozin" in claims[0]["subject"].lower()
    assert "mortality" in claims[0]["object"].lower()


def test_orchestrator_checkpointing():
    checkpoint_stage("test_run_123", stage="agent_1_claim_extraction", status="RUNNING", data={"sample": 42})
    assert "test_run_123" in _IN_MEMORY_CHECKPOINTS
    assert _IN_MEMORY_CHECKPOINTS["test_run_123"]["stage"] == "agent_1_claim_extraction"
    assert _IN_MEMORY_CHECKPOINTS["test_run_123"]["status"] == "RUNNING"
