from app.retrieval.embeddings import embedding_model

def test_embedding_generation():
    """Test that embedding model generates correctly shaped vectors."""
    text = "This is a test sentence."
    embedding = embedding_model.embed_text(text)
    
    # all-MiniLM-L6-v2 outputs 384-dimensional vectors
    assert len(embedding) == 384
    assert isinstance(embedding[0], float)

def test_batch_embedding():
    texts = ["Sentence one.", "Sentence two."]
    embeddings = embedding_model.embed_batch(texts)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
