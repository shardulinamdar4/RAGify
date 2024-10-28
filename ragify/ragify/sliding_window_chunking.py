import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Adjust import as needed
from typing import List

CHUNK_SIZE = 100  
CHUNK_OVERLAP = 20  

def sliding_window_chunking(text: str) -> List[str]:
    """Split input text into overlapping chunks."""
    logging.info("Processing input text for sliding window chunking")
    
    # Initialize the text splitter with chunk size and overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    # Split the text into chunks
    chunks = splitter.split_text(text)

    logging.info(f"Split into {len(chunks)} overlapping chunks of text")
    return chunks
