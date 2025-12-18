from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Optional
import instructor
from openai import OpenAI
import os

# Dummy key for demo
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "sk-dummy"

# 1. DEFINE THE DATA MODEL
# This is the "Contract". The AI must strictly adhere to this structure.
class CustomerInsight(BaseModel):
    sentiment_score: int = Field(..., description="Score 1-10 of customer satisfaction")
    key_issues: List[str] = Field(..., description="Bullet points of complaints")
    action_required: bool = Field(..., description="True if a human must intervene")

    # 2. VALIDATION LOGIC (Self-Correction)
    @field_validator('sentiment_score')
    def check_score_range(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Score must be between 1 and 10')
        return v

def run_extraction_demo():
    print("--- STARTING ANALYST AGENT ---")

    # Patching OpenAI client with Instructor to force Pydantic outputs
    client = instructor.from_openai(OpenAI())

    sample_ticket = """
    Subject: I am furious!
    Body: result was terrible. I tried to refund but the button is broken.
    I waited 4 hours on the phone. This is unacceptable!
    """

    print(f"Analyzing Ticket:\n{sample_ticket}\n")

    try:
        # 3. TYPE-SAFE GENERATION
        # response_model argument forces the LLM to output our Pydantic class
        insight = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=CustomerInsight,
            messages=[
                {"role": "user", "content": f"Extract insights from this: {sample_ticket}"},
            ],
            max_retries=3,
        )

        print("--- EXTRACTED JSON ---")
        print(insight.model_dump_json(indent=2))

        if insight.action_required:
            print("\n🚨 ALERT: Ticket escalated to human support agent.")

    except Exception as e:
        print(f"Extraction failed (Likely due to missing API Key in demo): {e}")
        # Mock output for portfolio display
        mock_output = CustomerInsight(
            sentiment_score=1,
            key_issues=["Refund button broken", "Long wait times"],
            action_required=True
        )
        print("\n(Mock Result for Portfolio Display):")
        print(mock_output.model_dump_json(indent=2))

if __name__ == "__main__":
    run_extraction_demo()
