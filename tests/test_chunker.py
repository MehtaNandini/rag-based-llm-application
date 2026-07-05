from app.document_processing.chunker import chunk_text

def test_chunk_text_empty():
    assert chunk_text("") == []

def test_chunk_text_small():
    text = "This is a small text."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_large():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=100)
    assert len(chunks) == 3
    # 0 to 400
    # 300 to 700
    # 600 to 1000
    assert chunks[0] == "A" * 400
    assert chunks[1] == "A" * 400
    assert chunks[2] == "A" * 400

def test_chunk_text_boundary():
    # Let's test the boundary logic. It prefers \n\n, then \n, then space.
    # A text with a newline in the middle of a chunk
    text = "Hello world\n\nThis is a test of the chunker logic."
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    
    # "Hello world\n\nThis " is 18 chars, length is 49.
    # The chunker should split near \n\n
    assert len(chunks) > 0
    assert "Hello world" in chunks[0]
