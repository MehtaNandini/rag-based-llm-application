from typing import List, Dict, Any, Tuple
from app.generation.llm_provider import get_llm
from app.retrieval.retriever import retrieve_relevant_chunks
from app.core.prompts import BASIC_RAG_PROMPT, STRICT_GROUNDED_PROMPT
from app.schemas.responses import SourceChunk

def answer_question(question: str, prompt_type: str = "strict") -> Tuple[str, List[SourceChunk]]:
    """
    RAG Orchestrator:
    1. Retrieve relevant chunks
    2. Format prompt
    3. Call LLM
    4. Return answer and sources
    """
    # 1. Retrieve
    chunks = retrieve_relevant_chunks(question)
    
    # Format sources for response
    sources = []
    for chunk in chunks:
        sources.append(
            SourceChunk(
                document_id=chunk["document_id"],
                document_name=chunk["document_name"],
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                score=chunk["score"]
            )
        )
        
    # If no chunks found, return immediately to save LLM call
    if not chunks:
        return "The information is not available in the uploaded documents.", []
        
    # 2. Format Context
    context_text = "\n\n---\n\n".join([f"Source ({c['document_name']}):\n{c['text']}" for c in chunks])
    
    # Select prompt template
    if prompt_type == "strict":
        prompt_template = STRICT_GROUNDED_PROMPT
    else:
        prompt_template = BASIC_RAG_PROMPT
        
    prompt = prompt_template.format(context=context_text, question=question)
    
    # 3. Call LLM
    llm = get_llm()
    answer = llm.generate(prompt)
    
    # 4. Check Guardrail explicitly if strict (the LLM should output exactly this, but we can also enforce it)
    # This is handled by the prompt, but we return whatever the LLM says.
    
    return answer, sources
