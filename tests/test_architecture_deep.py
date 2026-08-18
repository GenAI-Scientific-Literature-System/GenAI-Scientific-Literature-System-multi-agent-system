import os
import pytest
from services.common.medical_data_layer import MedicalDataLayer
from services.common.biobert import BioBERTClaimExtractor
from services.mas_orchestrator.main import (
    checkpoint_stage,
    _IN_MEMORY_CHECKPOINTS,
    start_run,
    execute_five_agent_pipeline,
    PipelineRequest
)


def test_faiss_medical_data_layer_indexing_and_updates():
    layer = MedicalDataLayer(dimensions=64, auto_connect=True)
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

    # Test FAISS document update and non-stale vector re-indexing
    layer.upsert_document({
        "id": "doc_1",
        "title": "Metformin updated title",
        "abstract": "Metformin inhibits mitochondrial complex I."
    })
    results_updated = layer.search("mitochondrial complex", limit=1)
    assert len(results_updated) == 1
    assert results_updated[0]["id"] == "doc_1"


def test_biobert_extraction_and_entities():
    extractor = BioBERTClaimExtractor(auto_load=True)
    text = "In patients with diabetes, empagliflozin significantly reduces cardiovascular mortality."
    claims = extractor.extract(text, paper_id="empa_01")
    
    assert len(claims) >= 1
    assert claims[0]["predicate"] in ["reduces", "significantly reduces"]
    assert "empagliflozin" in claims[0]["subject"].lower()
    assert "mortality" in claims[0]["object"].lower()
    assert "pico" in claims[0]


def test_orchestrator_execution_with_papers_and_query(monkeypatch):
    # Mock Retriever to return local test papers without network latency
    from pipeline.retrieval import Retriever
    mock_papers = [{
        "id": "p_mock_01",
        "title": "Metformin and Cellular Aging",
        "abstract": "Metformin reduces reactive oxygen species and improves cellular survival."
    }]
    monkeypatch.setattr(Retriever, "retrieve", lambda self, q, domains=None: mock_papers)

    # 1. Test run initialization
    init_res = start_run(PipelineRequest(query="SGLT2 inhibitors in heart failure"))
    run_id = init_res["run_id"]
    assert run_id.startswith("run-")
    assert init_res["status"] == "INITIALIZED"

    # 2. Test 5-agent execution with provided papers
    papers = [{
        "id": "p_hf_01",
        "title": "Dapagliflozin in Heart Failure",
        "abstract": "A randomized trial (n=4744) demonstrated that dapagliflozin significantly reduces cardiovascular death in heart failure."
    }]
    exec_res = execute_five_agent_pipeline(run_id, PipelineRequest(query="", papers=papers))
    assert exec_res["status"] == "SUCCESS"
    assert "result" in exec_res
    assert "claims" in exec_res["result"]
    assert len(exec_res["result"]["claims"]) >= 1
    assert "agreements" in exec_res["result"]

    # 3. Test query-only execution
    run_id_query = start_run(PipelineRequest(query="Metformin aging"))["run_id"]
    exec_query_res = execute_five_agent_pipeline(run_id_query, PipelineRequest(query="Metformin aging"))
    assert exec_query_res["status"] == "SUCCESS"
    assert "result" in exec_query_res
