from typing import List

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks of `chunk_size` characters with `chunk_overlap`.
    Tries to split on boundaries like double newline, newline, or space to avoid breaking words.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # If this is not the last chunk, try to find a good boundary
        if end < text_length:
            # Try finding a double newline
            boundary = text.rfind('\n\n', start, end)
            
            # If no double newline, try single newline
            if boundary == -1 or boundary < start + (chunk_size // 2):
                boundary = text.rfind('\n', start, end)
                
            # If no newline, try space
            if boundary == -1 or boundary < start + (chunk_size // 2):
                boundary = text.rfind(' ', start, end)
                
            # If a suitable boundary was found, adjust the end
            if boundary != -1 and boundary > start:
                end = boundary
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        # Move start forward, accounting for overlap
        start = end - chunk_overlap
        
        # Ensure we always move forward (to prevent infinite loops)
        
        # Better safety check:
        # If end is at or past text_length, we are done
        if end >= text_length:
            break
            
        # The next start should be end - overlap.
        start = end - chunk_overlap
        
        # But if overlap is too big or end didn't advance, ensure we progress
        if start <= end - chunk_size: 
            start = end
            
    return chunks
