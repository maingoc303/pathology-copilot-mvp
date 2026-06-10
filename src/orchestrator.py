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
        # Defining distinct vision and text models for absolute stability
        self.vision_model = "Salesforce/blip-vqa-base"
        self.text_model = "meta-llama/Meta-Llama-3-8B-Instruct"

    async def process_case(self, payload: CasePayload) -> Dict[str, Any]:
        print(f"[Orchestrator] Multi-modal parsing sequence triggered for: {payload.case_id}")
        
        if not self.hf_token:
            return {"status": "error", "detail": "HF_TOKEN key is missing from Render Environment Configurations."}

        # 1. Grab image and query parameters
        image_source = payload.image_metadata.get("image_url", "")
        user_query = payload.clinical_history

        # --- FIX: Smarter Agent Routing Mechanism ---
        # If no image url is present, or if it's a huge local data chunk, route it to standard Chat Completion
        if not image_source or image_source.startswith("data:image") or "wikipedia" not in image_source:
            print("[Agent Route] No clean remote image link found. Defaulting to Text LLM Pipeline...")
            try:
                # Ask Llama-3 the question directly using standard chat completions
                messages = [{"role": "user", "content": f"You are a helpful Pathology AI Assistant. Answer the user's clinical query: {user_query}"}]
                chat_completion = self.client.chat_completion(
                    model=self.text_model,
                    messages=messages,
                    max_tokens=150
                )
                answer_text = chat_completion.choices[0].message.content
                return {
                    "status": "success",
                    "case_id": payload.case_id,
                    "model_output": [{"generated_text": answer_text}]
                }
            except Exception as text_err:
                print(f"[Text LLM Error]: {str(text_err)}")
                return {"status": "error", "detail": f"Text LLM Engine reported: {str(text_err)}"}

        # 2. Vision Path: Run standard Visual Question Answering for clean image URLs
        print(f"[Agent Route] Valid cloud image link found. Querying Multi-modal Visual pipeline...")
        try:
            answer = self.client.visual_question_answering(
                image=image_source,
                question=user_query,
                model=self.vision_model
            )
            
            # Extract standard label/answer attributes from output list
            if isinstance(answer, list) and len(answer) > 0:
                model_text = answer[0].get("answer", str(answer))
            else:
                model_text = str(answer)

            return {
                "status": "success",
                "case_id": payload.case_id,
                "model_output": [{"generated_text": f"Based on the histology slice features, the matching target indicates: {model_text}."}]
            }

        except Exception as vision_err:
            print(f"[Vision Engine Error]: {str(vision_err)}")
            return {"status": "error", "detail": f"Vision Model Hub reported: {str(vision_err)}"}