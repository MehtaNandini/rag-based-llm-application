import os
from io import BytesIO
from typing import List, Dict, Any
from pypdf import PdfReader
import markdown

from app.document_processing.cleaner import clean_text
from app.document_processing.chunker import chunk_text
from app.core.config import settings

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(BytesIO(file_content))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_markdown(file_content: bytes) -> str:
    """Extract text from a Markdown file, stripping HTML if necessary."""
    # Simple markdown to text conversion (or just use raw text)
    # For RAG, raw markdown is often fine, but we decode it.
    text = file_content.decode('utf-8')
    return text

def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from a plain text file."""
    return file_content.decode('utf-8')

def process_document(file_name: str, file_content: bytes, document_id: str) -> List[Dict[str, Any]]:
    """
    Process an uploaded document:
    1. Extract text based on file type
    2. Clean the text
    3. Chunk the text
    4. Return chunks with metadata
    """
    ext = os.path.splitext(file_name)[1].lower()
    
    if ext == '.pdf':
        raw_text = extract_text_from_pdf(file_content)
    elif ext in ['.md', '.markdown']:
        raw_text = extract_text_from_markdown(file_content)
    elif ext == '.txt':
        raw_text = extract_text_from_txt(file_content)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
        
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        processed_chunks.append({
            "document_id": document_id,
            "document_name": file_name,
            "chunk_id": f"{document_id}_{i}",
            "text": chunk
        })
        
    return processed_chunks
