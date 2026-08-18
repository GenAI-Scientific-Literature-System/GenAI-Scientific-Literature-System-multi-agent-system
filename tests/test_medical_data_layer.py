from services.common.biobert import BioBERTClaimExtractor
from services.common.medical_data_layer import MedicalDataLayer


def test_in_memory_medical_data_layer_indexes_and_searches_documents():
    layer = MedicalDataLayer()
    layer.upsert_document({"id": "pmid-1", "title": "Metformin outcomes", "abstract": "Metformin reduces cardiovascular risk."})
    layer.upsert_document({"id": "pmid-2", "title": "Oncology trial", "abstract": "A cancer trial reported survival."})

    results = layer.search("metformin cardiovascular", limit=1)

    assert results[0]["id"] == "pmid-1"
    assert results[0]["similarity"] > 0


def test_biobert_service_has_grounded_claim_fallback():
    claims = BioBERTClaimExtractor().extract(
        "Metformin reduces cardiovascular events in adults.", paper_id="pmid-1"
    )

    assert claims[0]["paper_id"] == "pmid-1"
    assert claims[0]["predicate"] == "reduces"
