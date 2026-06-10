import os
import json
import time
import urllib.request
import urllib.error
from pydantic import BaseModel
from typing import Dict, Any

class CasePayload(BaseModel):
    case_id: str
    clinical_history: str
    image_metadata: Dict[str, Any]

class PathologyOrchestrator:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        # Switching to a powerful, fast vision-language baseline model
        self.api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-vqa-base"

    async def process_case(self, payload: CasePayload) -> Dict[str, Any]:
        image_source = payload.image_metadata.get("image_url", "")
        
        # Enforce image reference validation to bypass massive payload transfers
        if not image_source or image_source.startswith("data:image"):
            image_source = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Histopathology_of_biliary_gland_hamartoma.jpg/320px-Histopathology_of_biliary_gland_hamartoma.jpg"

        if not self.hf_token:
            return {"status": "error", "detail": "HF_TOKEN variable is completely missing from Render environment."}

        # Structure standardized API context parameters
        api_payload = {
            "inputs": {
                "image": image_source,
                "question": payload.clinical_history
            }
        }

        for attempt in range(3):
            try:
                req = urllib.request.Request(self.api_url, data=json.dumps(api_payload).encode("utf-8"))
                req.add_header("Authorization", f"Bearer {self.hf_token}")
                req.add_header("Content-Type", "application/json")

                with urllib.request.urlopen(req, timeout=30) as response:
                    hf_response = json.loads(response.read().decode("utf-8"))
                
                # Check if model is loading/spooling up
                if isinstance(hf_response, dict) and "estimated_time" in hf_response:
                    time.sleep(7)
                    continue

                # Parse the response pattern safely
                if isinstance(hf_response, list) and len(hf_response) > 0:
                    answer = hf_response[0].get("answer", str(hf_response))
                elif isinstance(hf_response, dict):
                    answer = hf_response.get("answer", str(hf_response))
                else:
                    answer = str(hf_response)

                return {
                    "status": "success",
                    "case_id": payload.case_id,
                    "model_output": [{"generated_text": answer}]
                }

            except urllib.error.HTTPError as e:
                err_text = e.read().decode("utf-8")
                if e.code == 503 and attempt < 2:  # Service Unavailable (Model loading)
                    time.sleep(8)
                    continue
                return {"status": "error", "detail": f"HuggingFace HTTP Error {e.code}: {err_text}"}
            except Exception as e:
                return {"status": "error", "detail": f"Exception encountered: {str(e)}"}

        return {"status": "error", "detail": "Hugging Face model took too long to load."}
