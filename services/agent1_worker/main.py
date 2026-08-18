from pydantic import BaseModel
from services.common.app import service_app
from services.common.biobert import BioBERTClaimExtractor

app = service_app("Agent 1 Claim Extraction")
extractor = BioBERTClaimExtractor(auto_load=True)


class ExtractionRequest(BaseModel):
    paper_id: str
    text: str


@app.post("/extract")
def extract(request: ExtractionRequest):
    return {
        "claims": extractor.extract(request.text, request.paper_id),
        "model": extractor.model_name,
        "is_model_loaded": extractor._is_loaded,
    }
