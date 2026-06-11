import os
import json
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
        
        # --- FIX: Updated target path identifier to use the active production cluster ---
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    async def process_case(self, payload: CasePayload) -> Dict[str, Any]:
        if not self.hf_token:
            return {"status": "error", "detail": "HF_TOKEN key is missing from Render configurations."}

        image_source = payload.image_metadata.get("image_url", "")
        roi_snapshot_detected = "roi_snapshot" in payload.image_metadata and payload.image_metadata["roi_snapshot"] is not None
        
        user_prompt = payload.clinical_history
        ehr_context = ""

        try:
            parsed_history = json.loads(payload.clinical_history)
            user_prompt = parsed_history.get("user_prompt", "")
            profile = parsed_history.get("ehr_profile", {})
            
            ehr_context = (
                f"--- ELECTRONIC HEALTH RECORD (EHR) REGISTERED PROFILE ---\n"
                f"• Patient Demographics: Age {profile.get('age', 'N/A')} | Sex: {profile.get('sex', 'N/A')}\n"
                f"• Molecular/Biomarker Assays: {profile.get('biomarkers', 'None listed')}\n"
                f"• Background History Logs: {profile.get('summary_notes', 'None recorded')}\n"
                if profile.get('attached_document_raw') : f"• Attached Diagnostic File Contents:\n{profile.get('attached_document_raw')}\n" else ""
                f"----------------------------------------------------------\n"
            )
        except Exception:
            pass

        system_instruction = (
            "You are PathChat, a world-class multimodal computational pathology AI generalist assistant. "
            "Cross-reference the explicit EHR profile parameters with the specified tissue specimen metrics "
            "and requested spatial query to generate professional, accurate clinical insights."
        )

        prompt_payload = f"Case Study ID: {payload.case_id}\n"
        if ehr_context:
            prompt_payload += ehr_context
            
        prompt_payload += f"Base Whole Slide Image: {image_source}\n"
        if roi_snapshot_detected:
            prompt_payload += "Spatial Annotation: User has isolated a high-power field focus block coordinate region.\n"
            
        prompt_payload += f"Diagnostic User Query: {user_prompt}\n\nFormulate Pathology Assessment Report:"

        try:
            chat_completion = self.client.chat_completion(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt_payload}
                ],
                max_tokens=450,
                temperature=0.2
            )
            
            response_text = chat_completion.choices[0].message.content
            return {
                "status": "success",
                "case_id": payload.case_id,
                "model_output": [{"generated_text": response_text}]
            }
        except Exception as e:
            return {"status": "error", "detail": f"Hugging Face Inference Cluster failed: {str(e)}"}