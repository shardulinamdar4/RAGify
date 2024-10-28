import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

CHUNK_SIZE = 100  # Define your chunk size here
CHUNK_OVERLAP = 20  # Define your chunk overlap here

def recursive_chunking(document_text: str) -> List[str]:
    """Split the input document text into chunks using RecursiveCharacterTextSplitter."""
    # Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    
    # Split the document text into chunks
    chunks = splitter.split_text(document_text)

    logging.info(f"Split into {len(chunks)} chunks of text")
    return chunks
