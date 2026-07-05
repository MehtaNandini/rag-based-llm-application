from pydantic import BaseModel
from typing import List, Optional


class SourceChunk(BaseModel):
    """A chunk of text used as source material for an answer."""
    document_id: str
    document_name: str
    chunk_id: str
    text: str
    score: float


class AskResponse(BaseModel):
    """Response model for asking a question."""
    answer: str
    sources: List[SourceChunk]


class DocumentResponse(BaseModel):
    """Response model for document metadata."""
    document_id: str
    document_name: str
    chunk_count: int


class ListDocumentsResponse(BaseModel):
    """Response model for listing indexed documents."""
    documents: List[DocumentResponse]
