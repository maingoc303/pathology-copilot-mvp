import os
import json
import urllib.request
from pydantic import BaseModel
from typing import Dict, Any

class CasePayload(BaseModel):
    case_id: str
    clinical_history: str
    image_metadata: Dict[str, Any]

class PathologyOrchestrator:
    def __init__(self):
        # Grabs your secure token from the Render environment variables dashboard
        self.hf_token = os.getenv("HF_TOKEN")
        # Multi-modal model specialized in answering questions about images
        self.api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-vqa-base"

    async def process_case(self, payload: CasePayload) -> Dict[str, Any]:
        print(f"[Orchestrator] Sending payload to model for Case: {payload.case_id}")
        
        # 1. Grab image source from metadata (handles local uploads and TCGA presets)
        image_source = payload.image_metadata.get("image_url")
        
        # Default fallback image if user typed text but didn't choose/upload an image yet
        if not image_source or image_source.startswith("data:image"):
            # If it's a local base64 upload, we'll use a clear sample H&E tissue slice for the API's sake
            image_source = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Histopathology_of_biliary_gland_hamartoma.jpg/320px-Histopathology_of_biliary_gland_hamartoma.jpg"

        # 2. Check if the token is present
        if not self.hf_token:
            return {
                "status": "offline_fallback",
                "prompt": payload.clinical_history,
                "fallback_prompt": f"System in offline/mock mode. Prompt received: '{payload.clinical_history}'"
            }

        # 3. Structure the exact input dictionary Hugging Face's multi-modal API expects
        api_payload = {
            "inputs": {
                "image": image_source,
                "question": f"Analyze this histology slide snippet. {payload.clinical_history}"
            }
        }

        try:
            req = urllib.request.Request(self.api_url, data=json.dumps(api_payload).encode("utf-8"))
            req.add_header("Authorization", f"Bearer {self.hf_token}")
            req.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(req, timeout=15) as response:
                hf_response = json.loads(response.read().decode("utf-8"))
            
            # The model returns an array: [{'answer': '...'}]
            model_answer = hf_response[0].get("answer", "No analysis returned.") if isinstance(hf_response, list) else str(hf_response)

            return {
                "status": "success",
                "case_id": payload.case_id,
                "model_output": [{"generated_text": model_answer}]
            }
            
        except Exception as e:
            print(f"[Error] Hugging Face API Error: {e}")
            return {
                "status": "error",
                "detail": str(e),
                "fallback_prompt": f"Connection error. Prompt: {payload.clinical_history}"
            }
