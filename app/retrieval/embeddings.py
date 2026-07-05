from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import settings

class EmbeddingModel:
    """Wrapper around SentenceTransformers for generating dense embeddings."""
    
    def __init__(self):
        # Load the model specified in the settings
        self.model = SentenceTransformer(settings.embedding_model)
        
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single piece of text."""
        # encode returns a numpy array, we convert to list of floats for DB insertion
        return self.model.encode(text).tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

# Singleton instance to avoid reloading the model
embedding_model = EmbeddingModel()
