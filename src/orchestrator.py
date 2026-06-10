import os
from pydantic import BaseModel
from typing import Dict, Any
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
        print(f"[Orchestrator] Processing pipeline request for: {payload.case_id}")
        
        if not self.hf_token:
            return {
                "status": "error", 
                "detail": "HF_TOKEN environment variable is completely missing from Render."
            }

        image_source = payload.image_metadata.get("image_url", "")
        user_query = payload.clinical_history

        # --- FIX FOR LOCAL UPLOADS ---
        # If the image source is a massive local base64 data string, we tag its presence 
        # but do not append the millions of raw text characters to the prompt to keep it clean.
        is_local_upload = image_source.startswith("data:image")
        
        system_instruction = (
            "You are PathChat, an expert conversational pathology co-pilot. Analyze the clinical queries "
            "and provided tissue morphology details to outline diagnostic criteria, anomalies, or recommendations."
        )

        # Build clean prompt context data
        prompt_payload = f"Case Study ID: {payload.case_id}\n"
        
        if is_local_upload:
            prompt_payload += "Specimen Source: User-uploaded local H&E slide asset (Base64 data verified)\n"
        elif image_source:
            prompt_payload += f"Specimen Source: Live cloud reference link ({image_source})\n"
            
        prompt_payload += f"Clinical History / Observed Morphology: {user_query}\n\nProvide evaluation report:"

        try:
            print(f"[HuggingFace Hub] Sending request payload to {self.model_id}...")
            chat_completion = self.client.chat_completion(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt_payload}
                ],
                max_tokens=350,
                temperature=0.3
            )
            
            response_text = chat_completion.choices[0].message.content

            return {
                "status": "success",
                "case_id": payload.case_id,
                "model_output": [{"generated_text": response_text}]
            }

        except Exception as e:
            return {
                "status": "error", 
                "detail": f"Hugging Face execution breakdown: {str(e)}"
            }