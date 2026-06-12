from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from src.orchestrator import PathologyOrchestrator, CasePayload
import os
from pydantic import BaseModel
from typing import Optional
import tempfile

app = FastAPI(title="Pathology Co-Pilot MVP", version="1.0.0")
orchestrator = PathologyOrchestrator()

@app.get("/", response_class=HTMLResponse)
def read_root():
    # Resolve the path to the template file dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, "templates", "index.html")
    
    with open(template_path, "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)

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

# ==========================================
# NEW CO-PILOT DOWNLOAD ENDPOINT
# ==========================================
#from fastapi import APIRouter, Response, HTTPException
#from pydantic import BaseModel
#from typing import Optional
#import tempfile
#import os

# Ensure weasyprint is imported at the top of your file
try:
    from weasyprint import HTML
except ImportError:
    pass

