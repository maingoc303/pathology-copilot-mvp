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

router = APIRouter()

class ReportPayload(BaseModel):
    age: str
    sex: str
    biomarkers: str
    clinical_notes: str
    diagnostic_findings: str
    roi_image_b64: Optional[str] = None

@router.post("/api/v1/download-report")
async def generate_pdf_report(payload: ReportPayload):
    # Construct an elegant, print-ready, high-resolution pathology template
    roi_html_element = ""
    if payload.roi_image_b64:
        roi_html_element = f"""
        <div class="section-block">
            <h3>III. Targeted Microscopic Field (Region of Interest Capture)</h3>
            <div style="text-align: center; margin-top: 10px;">
                <img src="{payload.roi_image_b64}" style="max-width: 100%; max-height: 250px; border: 1px solid #cbd5e1; border-radius: 4px;" />
            </div>
        </div>
        """

    formatted_findings = payload.diagnostic_findings.replace("\n\n", "</p><p>").replace("\n", "<br>")

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4;
                margin: 20mm 15mm;
                @bottom-right {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 8pt;
                    color: #64748b;
                }}
            }}
            body {{
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                color: #1e293b;
                line-height: 1.6;
                font-size: 10.5pt;
                margin: 0; padding: 0;
            }}
            .header-bar {{
                border-bottom: 2px solid #0f172a;
                padding-bottom: 12px;
                margin-bottom: 20px;
            }}
            .hospital-title {{
                font-size: 18pt; font-weight: bold; color: #0f172a; text-transform: uppercase; letter-spacing: 0.5px;
            }}
            .doc-type {{
                font-size: 11pt; color: #475569; font-weight: 500; margin-top: 4px;
            }}
            .patient-card-table {{
                width: 100%; border-collapse: collapse; margin-bottom: 20px; background: #f8fafc; border: 1px solid #e2e8f0;
            }}
            .patient-card-table td {{
                padding: 10px 12px; border: 1px solid #e2e8f0; font-size: 10pt;
            }}
            .section-block {{
                margin-bottom: 22px;
                page-break-inside: avoid;
            }}
            h3 {{
                font-size: 12pt; color: #0f172a; border-left: 4px solid #2563eb; padding-left: 8px; margin-bottom: 10px; margin-top: 0;
            }}
            p {{ margin: 0 0 8px 0; text-align: justify; }}
            .findings-box {{
                background: #ffffff; border: 1px solid #cbd5e1; padding: 15px; border-radius: 4px; font-size: 10pt;
            }}
        </style>
    </head>
    <body>
        <div class="header-bar">
            <div class="hospital-title">Pathology Workspace Consultation Service</div>
            <div class="doc-type">Clinical Multi-Modal Synthesis Case Report</div>
        </div>

        <table class="patient-card-table">
            <tr>
                <td style="width: 25%; font-weight: bold; color: #475569;">Patient Age:</td>
                <td style="width: 25%;">{payload.age}</td>
                <td style="width: 25%; font-weight: bold; color: #475569;">Biological Sex:</td>
                <td style="width: 25%;">{payload.sex}</td>
            </tr>
            <tr>
                <td style="font-weight: bold; color: #475569;">Biomarker Profiles:</td>
                <td colspan="3">{payload.biomarkers}</td>
            </tr>
        </table>

        <div class="section-block">
            <h3>I. Clinical Presentation & Macroscopic Background</h3>
            <p>{payload.clinical_notes}</p>
        </div>

        {roi_html_element}

        <div class="section-block">
            <h3>II. Microscopic Evaluations & AI Agent Findings</h3>
            <div class="findings-box">
                <p>{formatted_findings}</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Transform into physical PDF output via temporary cache memory layers
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        HTML(string=html_template).write_pdf(tmp_path)
        
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
            
        os.remove(tmp_path)
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=pathology_case_report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weasyprint conversion pipeline failure: {str(e)}")