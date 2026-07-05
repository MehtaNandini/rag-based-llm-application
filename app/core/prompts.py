# Basic RAG Prompt
BASIC_RAG_PROMPT = """You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {question}

Answer:"""

# Strict Grounded Prompt (With Guardrail)
STRICT_GROUNDED_PROMPT = """You are a helpful assistant. Answer the user's question based strictly on the provided context.
If the answer cannot be found in the context, do not guess or use outside knowledge. Instead, say exactly:
"The information is not available in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""

# Summarization Prompt
SUMMARIZATION_PROMPT = """You are a helpful assistant. Provide a concise summary of the following document chunk.

Document:
{context}

Summary:"""
