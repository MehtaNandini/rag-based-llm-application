from pydantic import BaseModel, Field
from typing import Optional


class AskRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(..., description="The question to ask the AI.")
    prompt_type: str = Field("strict", description="The type of prompt to use: 'basic', 'strict', or 'summarize'.")
