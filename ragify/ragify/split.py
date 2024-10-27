import logging
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from chunking_strategies.constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_MAP,
    INGEST_THREADS,
    SOURCE_RESPOSITORY_PATH,
)


def load_single_document(file_path: str) -> tuple[str, list[Document]]:
    """Load a single document from a file path."""
    logging.info(f"Loading {file_path}")
    file_extension = os.path.splitext(file_path)[1]
    ext_metadata = DOCUMENT_MAP.get(file_extension)
    if ext_metadata:
        loader_class = ext_metadata.get("loader")
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return file_extension, loader.load()


def load_document_batch(filepaths: list[str]):
    """Load multiple documents in parallel."""
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


def load_documents(source_dir: pathlib.Path) -> list[Document]:
    """Load all documents from the source documents directory."""
    all_files = source_dir.rglob("*")
    paths = []
    for file_path in all_files:
        file_extension = os.path.splitext(file_path)[1]
        source_file_path = os.path.join(source_dir, file_path)
        if file_extension in DOCUMENT_MAP.keys():
            paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs


def get_split_documents(directory: pathlib.Path):
    """Load documents into vectorstore."""
    logging.info(f"Loading documents from {directory}")
    documents = load_documents(directory)
    texts = []
    # Split the documents into chunks
    for file_extension, docs in documents:
        ext_metadata = DOCUMENT_MAP.get(file_extension)
        # If there is no language defined, don't chunk the text
        if "language" not in ext_metadata:
            chunks = docs
        # If there is a language defined, chunk the text according to the language
        else:
            language = ext_metadata["language"]
            # If the language is None, use the basic splitter
            if language is None:
                splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            # Otherwise use the specific language
            else:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
            chunks = splitter.split_documents(docs)
        texts.extend(chunks)
    logging.info(f"Loaded {len(documents)} documents from {directory}")
    logging.info(f"Split into {len(texts)} chunks of text")
    return texts


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO)
    get_split_documents(SOURCE_RESPOSITORY_PATH)
