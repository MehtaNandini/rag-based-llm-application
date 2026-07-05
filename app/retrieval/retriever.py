from typing import List, Dict, Any
from app.retrieval.embeddings import embedding_model
from app.retrieval.vector_store import vector_store
from app.core.config import settings

def index_document_chunks(chunks: List[Dict[str, Any]]):
    """Generate embeddings for chunks and store them in the vector DB."""
    if not chunks:
        return
        
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings
    embeddings = embedding_model.embed_batch(texts)
    
    # Store in ChromaDB
    vector_store.add_chunks(chunks, embeddings)

def retrieve_relevant_chunks(question: str, top_k: int = None) -> List[Dict[str, Any]]:
    """Retrieve the most relevant chunks for a given question."""
    if top_k is None:
        top_k = settings.top_k_results
        
    # Generate embedding for the question
    query_embedding = embedding_model.embed_text(question)
    
    # Query the vector store
    results = vector_store.query(query_embedding, top_k=top_k)
    
    return results

def get_all_documents() -> List[Dict[str, Any]]:
    """Get a list of all indexed documents."""
    return vector_store.list_documents()

def remove_document(document_id: str):
    """Remove a document from the index."""
    vector_store.delete_document(document_id)
