import os
from pydantic import BaseModel
from typing import Dict, Any, Optional
from huggingface_hub import InferenceClient

class CasePayload(BaseModel):
    case_id: str
    clinical_history: str
    image_metadata: Dict[str, Any]

class PathologyOrchestrator:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.client = InferenceClient(token=self.hf_token)
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    async def process_case(self, payload: CasePayload) -> Dict[str, Any]:
        if not self.hf_token:
            return {"status": "error", "detail": "HF_TOKEN environment variable is completely missing on Render."}

        image_source = payload.image_metadata.get("image_url", "")
        roi_snapshot_detected = "roi_snapshot" in payload.image_metadata and payload.image_metadata["roi_snapshot"] is not None
        user_query = payload.clinical_history

        system_instruction = (
            "You are PathChat, a state-of-the-art computational pathology generalist AI. Evaluate "
            "the provided case files, annotations, and spatial ROI captures to provide high-quality diagnostic guidance."
        )

        # Build structural injection string context
        prompt_payload = f"Case Identifier: {payload.case_id}\n"
        prompt_payload += f"Base Specimen Whole Slide URL: {image_source}\n"
        
        if roi_snapshot_detected:
            prompt_payload += "Spatial Zoom State: User has isolated a high-power field Region of Interest (ROI) view canvas for this question.\n"
        
        prompt_payload += f"Pathology Assessment Inquiry: {user_query}\n\nFormulate professional analysis report:"

        try:
            chat_completion = self.client.chat_completion(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt_payload}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            response_text = chat_completion.choices[0].message.content
            return {
                "status": "success",
                "case_id": payload.case_id,
                "model_output": [{"generated_text": response_text}]
            }
        except Exception as e:
            return {"status": "error", "detail": f"Hugging Face core cluster failure: {str(e)}"}