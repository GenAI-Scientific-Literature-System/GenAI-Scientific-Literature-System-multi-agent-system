"""
FastAPI Backend — Epistemic Reasoning Engine
Endpoints for Agent 4, Agent 5, and the full pipeline.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from models import PipelineInput, PipelineOutput, HypothesisInput
from pipeline import run_pipeline
from agent4 import Agent4
from agent5 import Agent5

app = FastAPI(
    title="Epistemic Reasoning Engine",
    description="Agent 4 (Hypothesis Compatibility) + Agent 5 (Epistemic Boundary Analysis)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static assets at root-relative paths matching the HTML references
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/css", StaticFiles(directory=os.path.join(frontend_path, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(frontend_path, "js")), name="js")


@app.get("/")
def root():
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "Epistemic Engine running", "docs": "/docs"}


@app.get("/health")
def health():
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    return {
        "status": "healthy",
        "api_key_set": bool(api_key),
        "agents": ["Agent_4_Hypothesis_Compatibility", "Agent_5_Epistemic_Boundary"]
    }


@app.post("/api/pipeline", response_model=PipelineOutput)
def run_full_pipeline(input_data: PipelineInput):
    """
    Run the full Agent 4 → Agent 5 pipeline.
    Accepts a list of hypotheses and returns compatibility analysis + epistemic boundaries.
    """
    if len(input_data.hypotheses) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 hypotheses required for analysis."
        )
    if len(input_data.hypotheses) > 8:
        raise HTTPException(
            status_code=400,
            detail="Maximum 8 hypotheses per request (combinatorial limit)."
        )

    try:
        result = run_pipeline(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent4")
def run_agent4(input_data: PipelineInput):
    """Run only Agent 4 hypothesis compatibility simulation."""
    if len(input_data.hypotheses) < 2:
        raise HTTPException(status_code=400, detail="At least 2 hypotheses required.")
    try:
        agent = Agent4()
        results = agent.simulate(input_data.hypotheses, input_data.context or "")
        return {"agreements": [r.dict() for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent5")
def run_agent5(input_data: PipelineInput):
    """Run only Agent 5 epistemic boundary analysis (no Agent 4 context)."""
    try:
        agent = Agent5()
        results = agent.analyze(input_data.hypotheses, [])
        return {"uncertainties": [r.dict() for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample")
def get_sample():
    """Return a sample input for testing the pipeline."""
    return {
        "hypotheses": [
            {
                "id": "h1",
                "text": "Regular aerobic exercise significantly reduces systolic blood pressure in adults with hypertension by improving endothelial function.",
                "paper_id": "paper_001",
                "domain": "cardiovascular medicine",
                "assumptions": ["exercise duration >= 30 min", "frequency >= 3x/week", "no medication changes"],
                "variables": ["systolic BP", "exercise duration", "VO2 max", "endothelial NO production"],
                "evidence": "Meta-analysis of 54 RCTs showing mean reduction of 8 mmHg"
            },
            {
                "id": "h2",
                "text": "Resistance training is equally effective as aerobic exercise in reducing blood pressure through muscle mass-mediated insulin sensitivity improvements.",
                "paper_id": "paper_002",
                "domain": "exercise physiology",
                "assumptions": ["progressive overload applied", "protein intake adequate", "3+ months duration"],
                "variables": ["systolic BP", "muscle mass", "insulin sensitivity", "GLUT4 expression"],
                "evidence": "RCT with 120 participants over 6 months"
            },
            {
                "id": "h3",
                "text": "Blood pressure reduction from exercise is primarily mediated by neurological adaptations reducing sympathetic nervous system activity, not peripheral vascular changes.",
                "paper_id": "paper_003",
                "domain": "neurocardiology",
                "assumptions": ["central mechanisms dominate", "autonomic nervous system measurable"],
                "variables": ["sympathetic nerve activity", "heart rate variability", "baroreflex sensitivity"],
                "evidence": "Longitudinal study using microneurography in 45 subjects"
            }
        ],
        "context": "Mechanisms of exercise-induced blood pressure reduction in hypertensive adults"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
