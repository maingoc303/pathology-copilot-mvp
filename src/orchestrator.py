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
        # Read the Hugging Face access token from your Render dashboard settings
        self.hf_token = os.getenv("HF_TOKEN")
        self.client = InferenceClient(token=self.hf_token)
        
        # Mapping our emulation endpoints on the Hugging Face Hub
        self.judith_text_brain = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.pathchat_vision_bridge = "Salesforce/blip-vqa-base"

    async def process_case(self, payload: CasePayload) -> Dict[str, Any]:
        print(f"[Orchestrator] Running Hugging Face PathChat/Judith Pipeline for: {payload.case_id}")
        
        if not self.hf_token:
            return {
                "status": "error", 
                "detail": "HF_TOKEN environment variable is completely missing from your Render Dashboard configurations."
            }

        image_source = payload.image_metadata.get("image_url", "")
        user_query = payload.clinical_history

        # Isolate heavy raw text strings to prevent urllib/socket drops
        has_valid_image = image_source and not image_source.startswith("data:image")

        try:
            # --- PATHCHAT VISION WORKFLOW ---
            if has_valid_image:
                print(f"[PathChat Pipeline] Directing tissue matrix payload to visual QA layer...")
                
                # Formulate a structured instruction-tuned prompt mirroring Mahmood Lab datasets
                tuned_query = f"System Persona: PathChat Pathology Generalist AI. Task: Analyze this H&E specimen. Query: {user_query}"
                
                answer = self.client.visual_question_answering(
                    image=image_source,
                    question=tuned_query,
                    model=self.pathchat_vision_bridge
                )
                
                model_text = answer[0].get("answer", str(answer)) if isinstance(answer, list) else str(answer)
                formatted_response = f"**[PathChat Evaluation]** Micro-architectural analysis suggests: {model_text}."

            # --- JUDITH AGENTIC WORKFLOW ---
            else:
                print(f"[Judith Pipeline] Directing conversational text query to clinical reasoning brain...")
                
                system_instruction = (
                    "You are Judith, an agentic pathology coordinator model. You specialize in clinical criteria evaluation, "
                    "diagnostic break-downs, and recommending algorithmic next-steps for medical datasets."
                )
                
                chat_completion = self.client.chat_completion(
                    model=self.judith_text_brain,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_query}
                    ],
                    max_tokens=250
                )
                
                formatted_response = chat_completion.choices[0].message.content

            return {
                "status": "success",
                "case_id": payload.case_id,
                "model_output": [{"generated_text": formatted_response}]
            }

        except Exception as e:
            print(f"[HuggingFace Pipeline Exception]: {str(e)}")
            return {
                "status": "error", 
                "detail": f"Hugging Face hub returned an operation error: {str(e)}"
            }