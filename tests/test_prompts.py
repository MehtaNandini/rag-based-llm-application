from app.core.prompts import BASIC_RAG_PROMPT, STRICT_GROUNDED_PROMPT

def test_basic_prompt_formatting():
    context = "This is the context."
    question = "What is this?"
    
    formatted = BASIC_RAG_PROMPT.format(context=context, question=question)
    
    assert "This is the context." in formatted
    assert "What is this?" in formatted

def test_strict_prompt_formatting():
    context = "Contextual information."
    question = "Ask something."
    
    formatted = STRICT_GROUNDED_PROMPT.format(context=context, question=question)
    
    assert "Contextual information." in formatted
    assert "Ask something." in formatted
    assert "information is not available" in formatted
