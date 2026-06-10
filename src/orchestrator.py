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
        # Using the official serverless vision model natively supported by Hugging Face
        self.model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    async def process_case(self, payload: CasePayload) -> Dict[str, Any]:
        print(f"[Orchestrator] Triggering PathChat Emulation via Hugging Face for: {payload.case_id}")
        
        if not self.hf_token:
            return {
                "status": "error", 
                "detail": "HF_TOKEN environment variable is missing from your Render Settings."
            }

        image_source = payload.image_metadata.get("image_url", "")
        user_query = payload.clinical_history

        # Fallback to an absolute public asset if user hasn't selected an image
        if not image_source or image_source.startswith("data:image"):
            image_source = "https://picsum.photos/id/1025/600/400"

        try:
            # Build standard instruction-tuned multimodal query blocks
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"You are PathChat, an expert pathology co-pilot. Analyze this specimen and answer: {user_query}"},
                        {"type": "image_url", "image_url": {"url": image_source}}
                    ]
                }
            ]

            print(f"[HuggingFace Hub] Sending request to {self.model_id}...")
            chat_completion = self.client.chat_completion(
                model=self.model_id,
                messages=messages,
                max_tokens=300
            )
            
            response_text = chat_completion.choices[0].message.content

            return {
                "status": "success",
                "case_id": payload.case_id,
                "model_output": [{"generated_text": response_text}]
            }

        except Exception as e:
            print(f"[HuggingFace Error Tracing]: {str(e)}")
            return {
                "status": "error", 
                "detail": f"Hugging Face hub cluster returned: {str(e)}"
            }