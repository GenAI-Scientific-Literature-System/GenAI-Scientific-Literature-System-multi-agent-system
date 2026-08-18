from fastapi import HTTPException
from pydantic import BaseModel
from services.common.app import service_app
from services.common.medical_data_layer import MedicalDataLayer

app = service_app("Medical Data Layer")
layer = MedicalDataLayer()


class Document(BaseModel):
    id: str | None = None
    title: str = ""
    abstract: str = ""
    text: str = ""
    pmid: str | None = None


@app.post("/documents")
def upsert(document: Document):
    return {"id": layer.upsert_document(document.model_dump())}


@app.get("/search")
def search(query: str, limit: int = 10):
    if not query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    return {"documents": layer.search(query, limit)}


@app.get("/connections")
def connections():
    return layer.connect()
