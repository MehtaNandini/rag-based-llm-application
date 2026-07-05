import chromadb
from typing import List, Dict, Any
from app.core.config import settings

class VectorStore:
    """Abstraction over ChromaDB for storing and retrieving document chunks."""
    
    def __init__(self):
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        
        # Create or get the main collection
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )
        
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add document chunks and their embeddings to the vector store."""
        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"]
            } 
            for chunk in chunks
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
    def query(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """Query the vector store for the most similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "chunk_id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "document_id": results['metadatas'][0][i]['document_id'],
                    "document_name": results['metadatas'][0][i]['document_name'],
                    "score": results['distances'][0][i] if results['distances'] else 0.0
                })
                
        return formatted_results
        
    def delete_document(self, document_id: str):
        """Delete all chunks associated with a document_id."""
        self.collection.delete(
            where={"document_id": document_id}
        )
        
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all unique documents in the store and their chunk counts."""
        # Note: ChromaDB doesn't have a direct "group by" feature, 
        # so we get all metadatas and group them manually.
        # For a production app with many docs, this might be slow,
        # but it's fine for a personal project/CV.
        results = self.collection.get(include=["metadatas"])
        
        doc_counts = {}
        doc_names = {}
        
        for meta in results['metadatas']:
            doc_id = meta['document_id']
            doc_names[doc_id] = meta['document_name']
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            
        documents = []
        for doc_id, count in doc_counts.items():
            documents.append({
                "document_id": doc_id,
                "document_name": doc_names[doc_id],
                "chunk_count": count
            })
            
        return documents

# Singleton instance
vector_store = VectorStore()
