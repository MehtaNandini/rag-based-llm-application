from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.schemas.requests import AskRequest
from app.schemas.responses import AskResponse, ListDocumentsResponse, DocumentResponse
from app.document_processing.loader import process_document
from app.retrieval.retriever import index_document_chunks, get_all_documents, remove_document
from app.generation.rag_chain import answer_question
import uuid

router = APIRouter()

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document, process it, chunk it, and index it in the vector store.
    """
    try:
        content = await file.read()
        document_id = str(uuid.uuid4())
        
        # 1. Process document into chunks
        chunks = process_document(file.filename, content, document_id)
        
        # 2. Index chunks
        index_document_chunks(chunks)
        
        return {
            "message": f"Successfully indexed '{file.filename}'",
            "document_id": document_id,
            "chunks_processed": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(request: AskRequest):
    """
    Ask a question over the uploaded documents.
    """
    try:
        answer, sources = answer_question(request.question, request.prompt_type)
        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents", response_model=ListDocumentsResponse)
async def list_documents():
    """
    List all documents currently in the index.
    """
    try:
        docs = get_all_documents()
        doc_responses = [
            DocumentResponse(
                document_id=doc["document_id"],
                document_name=doc["document_name"],
                chunk_count=doc["chunk_count"]
            )
            for doc in docs
        ]
        return ListDocumentsResponse(documents=doc_responses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the index.
    """
    try:
        remove_document(document_id)
        return {"message": f"Successfully deleted document {document_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
