import argparse
import logging
from pathlib import Path
from ragify.paragraph_chunking import paragraph_chunking
from ragify.recursive_chunking import recursive_chunking
from ragify.token_based_chunking import token_based_chunking
from ragify.sliding_window_chunking import sliding_window_chunking
from ragify.sentence_chunking import sentence_chunking
from ragify.semantic_chunking import semantic_chunking



logger = logging.getLogger(__name__)

def apply_chunking_strategy(document_text: str, strategy: str, **kwargs):
    """Apply the specified chunking strategy to a document."""
    if strategy == "paragraph":
        return paragraph_chunking(document_text)
    elif strategy == "token":
        token_limit = kwargs.get("token_limit", 512)
        return token_based_chunking(document_text, token_limit)
    elif strategy == "semantic":
        return semantic_chunking(document_text)
    elif strategy == "sliding_window":
        chunk_size = kwargs.get("chunk_size", 512)
        chunk_overlap = kwargs.get("chunk_overlap", 50)
        return sliding_window_chunking(document_text)
    elif strategy == "recursive":
        return recursive_chunking(document_text)
    elif strategy == "sentence":
        return sentence_chunking(document_text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def main(args: argparse.Namespace):
    """Load a document, apply the chunking strategy, and output the results."""
    file_path = Path(args.file_path)
    
    if not file_path.is_file():
        raise ValueError(f"The specified file does not exist: {file_path}")

    # Read the document content
    with open(file_path, "r", encoding="utf-8") as file:
        document_text = file.read()

    # Apply the chosen chunking strategy
    chunks = apply_chunking_strategy(
        document_text, 
        args.chunking_strategy,
        token_limit=args.token_limit,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Log the chunks
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Chunk {i}:\n{chunk}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="Apply a chunking strategy to a specified document.")

    # Required arguments for file path and chunking strategy
    parser.add_argument("file_path", help="Path to the document file")
    parser.add_argument(
        "--chunking-strategy",
        choices=["paragraph", "token", "semantic", "sliding_window", "recursive", "sentence"],
        required=True,
        help="Select the chunking strategy to use."
    )
    
    # Optional parameters for specific chunking strategies
    parser.add_argument("--token-limit", type=int, default=512, help="Token limit for token-based chunking")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size for sliding window chunking")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap size for sliding window chunking")

    args = parser.parse_args()
    main(args)
