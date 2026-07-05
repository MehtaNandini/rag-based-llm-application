import re

def clean_text(text: str) -> str:
    """
    Clean extracted text by removing excessive whitespace, 
    newlines, and non-printable characters.
    """
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Strip leading and trailing whitespace
    return text.strip()
