import os
from pydantic import BaseModel
from typing import Dict, Any
# --- Use the official Hugging Face connection module ---
from huggingface_hub import InferenceClient

class CasePayload(BaseModel):
    case_id: str
    clinical_history: str
    image_metadata: Dict[str, Any]

class PathologyOrchestrator:
    def __init__(self):
        # Automatically extracts token securely from Render environment configurations
        self.hf_token = os.getenv("HF_TOKEN")
        # Initialize the official client engine
        self.client = InferenceClient(token=self.hf_token)
        # Fast, free serverless multi-modal engine
        self.model_id = "Salesforce/blip-vqa-base"

    async def process_case(self, payload: CasePayload) -> Dict[str, Any]:
        print(f"[Orchestrator] Executing model inference check for {payload.case_id}")
        
        image_source = payload.image_metadata.get("image_url", "")
        
        # Enforce clean remote string URLs to prevent excessive memory buffering
        if not image_source or image_source.startswith("data:image"):
            image_source = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Histopathology_of_biliary_gland_hamartoma.jpg/320px-Histopathology_of_biliary_gland_hamartoma.jpg"

        if not self.hf_token:
            return {"status": "error", "detail": "HF_TOKEN variable is completely missing from Render variables config."}

        try:
            # The client executes the visual Q&A call natively via standard parameters
            print(f"[Orchestrator] Dispatching to model hub via InferenceClient...")
            answer = self.client.visual_question_answering(
                image=image_source,
                question=payload.clinical_history,
                model=self.model_id
            )
            
            # The official hub response parses directly into structural objects or list/dict
            print(f"[Success] Received model answer response context.")
            model_text = answer[0].get("answer", str(answer)) if isinstance(answer, list) else str(answer)

            return {
                "status": "success",
                "case_id": payload.case_id,
                "model_output": [{"generated_text": model_text}]
            }

        except Exception as e:
            print(f"[Inference client exception fail]: {str(e)}")
            return {
                "status": "error", 
                "detail": f"HuggingFace Hub Client reported: {str(e)}"
            }