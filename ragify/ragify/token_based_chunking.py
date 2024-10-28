import logging
from transformers import GPT2Tokenizer  # Ensure you have transformers installed
from typing import List
from langchain.text_splitter import CharacterTextSplitter


# Initialize the tokenizer (e.g., GPT-2)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
TOKEN_LIMIT = 100  # Define your token limit



def token_based_chunking(document_text: str, token_limit: int = TOKEN_LIMIT) -> List[str]:
    """Chunk a string of text by token limit."""
    logging.info("Processing input text for token-based chunking")
    
    # Use the token-based chunking function
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n\n", 
    chunk_size=100, 
    chunk_overlap=10, 
    is_separator_regex=False,
    model_name='text-embedding-3-small',
    encoding_name='text-embedding-3-small',)

    doc_list = text_splitter.create_documents([document_text])
    logging.info(f"Split text into {len(doc_list)} token-based chunks")
    return doc_list

