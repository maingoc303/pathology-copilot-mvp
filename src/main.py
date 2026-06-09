from fastapi import FastAPI, HTTPException
from src.orchestrator import PathologyOrchestrator, CasePayload

app = FastAPI(title="Pathology Co-Pilot MVP", version="1.0.0")
orchestrator = PathologyOrchestrator()

@app.get("/health")
def health_check():
    # Render hits this endpoint automatically to verify your app is running
    return {
        "status": "healthy", 
        "environment": "render_deployment",
        "orchestration": "active"
    }

@app.post("/api/v1/analyze")
async def analyze_case(payload: CasePayload):
    try:
        result = await orchestrator.process_case(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
