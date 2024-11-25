import re

def dynamic_chunking(text, chunk_size):
    """
    Dynamically chunks text based on a provided chunk size and sentence boundaries.
    """
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += f" {sentence}"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def adaptive_overlap_chunking(text, chunk_size, overlap):
    """
    Implements adaptive overlapping chunking.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks

def hierarchical_chunking(text, chunk_size):
    """
    Implements hierarchical chunking dynamically.
    """
    initial_chunks = dynamic_chunking(text, chunk_size)

    final_chunks = []
    for chunk in initial_chunks:
        if len(chunk.split()) > chunk_size // 2:
            sub_chunks = dynamic_chunking(chunk, chunk_size // 2)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks
