from langchain.text_splitter import SpacyTextSplitter
import logging

CHUNK_SIZE = 100  # Define your chunk size here
CHUNK_OVERLAP = 20  # Define your chunk overlap here

def sentence_chunking(document_text: str, language: str = "en") -> list[str]:
    """Split a document into sentence chunks based on the provided text and language."""
    logging.info("Splitting document into sentence chunks.")

    splitter = SpacyTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator = '\n',    pipeline = 'sentencizer'
    )

    # Split the document text into chunks
    chunks = splitter.split_text(text=document_text)  # Call on the instance

    logging.info(f"Split into {len(chunks)} sentence chunks of text")
    return chunks
