import json
import logging
import os
import pathlib
from datetime import datetime

import pandas as pd
from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from chunking_strategies.constants import (
    DEVICE,
    DIRECTORY_PATH,
    EMBEDDING_MODEL_NAME,
    KNOWLEDGE_REPOSITORY_PATH,
    PGVECTOR_DATABASE_NAME,
    PGVECTOR_DRIVER,
    PGVECTOR_HOST,
    PGVECTOR_PASS,
    PGVECTOR_PORT,
    PGVECTOR_USER,
)
from chunking_strategies.split import get_split_documents

logger = logging.getLogger(__name__)


def get_embedder():
    """Define embedder to convert text into vectors."""
    model_kwargs = {"device": DEVICE}
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        show_progress=True,
    )

    return embedder


def ingest(dirs: list[pathlib.Path], collection: str, delete: bool, logs_folder_id: str = None):
    """Load documents into a vectorstore."""
    # Get documents
    all_documents = []
    directory_source_urls = {}
    for directory in dirs:
        meta_lookup_path = directory / "meta_lookup.json"
        if not os.path.exists(meta_lookup_path):
            raise FileNotFoundError(f"No such file or directory: {meta_lookup_path} - Is the file missing from your folder or s3 bucket?")
        with open(meta_lookup_path) as f:
            meta_lookup = json.load(f)
        documents = get_split_documents(directory=directory)
        for doc in documents:
            source = str(pathlib.Path(doc.metadata["source"]).relative_to(directory))
            doc.metadata = meta_lookup[source]
            doc.metadata["source"] = source
            directory_source_url = (directory.relative_to(KNOWLEDGE_REPOSITORY_PATH), source, doc.metadata["url"])
            if directory_source_url not in directory_source_urls:
                directory_source_urls[directory_source_url] = 0
            directory_source_urls[directory_source_url] += 1
        all_documents.extend(documents)

    # Create embeddings
    embedder = get_embedder()

    # Build the Postgres connection string
    connection_string = PGVector.connection_string_from_db_params(
        driver=PGVECTOR_DRIVER,
        host=PGVECTOR_HOST,
        port=int(PGVECTOR_PORT),
        database=PGVECTOR_DATABASE_NAME,
        user=PGVECTOR_USER,
        password=PGVECTOR_PASS,
    )

    # Delete the collection (if requested)
    if delete:
        db = PGVector(
            embedding_function=embedder,
            collection_name=collection,
            connection_string=connection_string,
            use_jsonb=True,
        )
        db.delete_collection()
        logger.info(f"Collection {collection} deleted")

    # Load the documents
    logger.info(f"Loading {len(all_documents)} embeddings to {PGVECTOR_HOST} - {PGVECTOR_DATABASE_NAME} - {collection}")
    db = PGVector(
        embedding_function=embedder,
        collection_name=collection,
        connection_string=connection_string,
        use_jsonb=True,
    )
    db.add_documents(documents=all_documents)
    logger.info(f"Successfully loaded {len(all_documents)} embeddings")

    db.create_collection()

    if logs_folder_id is not None:
        from chunking_strategies.googledrive import GDrive

        directory_source_url_chunks = [
            list(directory_source_url) + [chunks] for directory_source_url, chunks in directory_source_urls.items()
        ]
        df = pd.DataFrame(directory_source_url_chunks, columns=["origin", "path", "url", "chunks"])
        gd = GDrive()
        outpath = DIRECTORY_PATH / "tmp" / "results.csv"
        df.to_csv(outpath, index=False)
        gd.upload_file(file_path=outpath, filename=f"{PGVECTOR_HOST} - {collection} - {datetime.now()}.csv", folder_id=logs_folder_id)
        os.remove(outpath)
