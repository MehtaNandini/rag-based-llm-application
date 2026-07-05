# Application Architecture

## Document Ingestion Flow

```mermaid
graph TD
    A[User Uploads Document] --> B[FastAPI Endpoint: /upload]
    B --> C{File Type?}
    C -->|PDF| D[PyPDF Extractor]
    C -->|TXT/MD| E[Text Extractor]
    D --> F[Text Cleaner]
    E --> F
    F --> G[Text Chunker]
    G --> H[Sentence Transformers]
    H -->|Generate Embeddings| I[ChromaDB Vector Store]
```

## Retrieval and Generation (RAG) Flow

```mermaid
graph TD
    A[User Asks Question] --> B[FastAPI Endpoint: /ask]
    B --> C[Sentence Transformers]
    C -->|Embed Question| D[ChromaDB Vector Store]
    D -->|Similarity Search| E[Top-K Document Chunks]
    E --> F[Prompt Builder]
    F -->|Inject Context & Question| G{LLM Provider}
    G -->|Mock| H[Mock Response]
    G -->|OpenAI| I[OpenAI API]
    G -->|Ollama| J[Local Model]
    H --> K[Return Answer & Sources]
    I --> K
    J --> K
```

## Key Components

1. **Document Processing**: Cleans and chunks the text into overlapping segments to preserve context across boundaries.
2. **Retrieval**: Uses dense vector embeddings (Cosine Similarity) to find the top `K` most relevant chunks for a given query.
3. **Generation**: Uses a strict prompt template that forces the LLM to only use the provided context, triggering a guardrail fallback if the answer is not present.
