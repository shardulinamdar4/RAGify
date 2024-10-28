import logging
from typing import List




def paragraph_chunking(document_text: str) -> List[str]:
    """Chunk text based on paragraphs."""
    paragraphs = document_text.split("\n\n")  # Split by double newlines
    chunks = [para.strip() for para in paragraphs if para.strip()]
    return chunks

def get_paragraph_chunks(document_text: str) -> List[str]:
    """Load a document and chunk it by paragraphs."""
    chunks = paragraph_chunking(document_text)
    logging.info(f"Split document into {len(chunks)} paragraph-based chunks")
    return chunks

