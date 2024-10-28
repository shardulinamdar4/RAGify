import os
import pathlib

import torch
from dotenv import load_dotenv
from langchain.text_splitter import Language


load_dotenv()

# PATHS
DIRECTORY_PATH = pathlib.Path(os.path.dirname(__file__)).parent

# INGEST
INGEST_THREADS = 8
TOKEN_LIMIT=500
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 250
SIMILARITY_THRESHOLD=0.7

# PGVECTOR
PGVECTOR_DRIVER = os.environ.get("PGVECTOR_DRIVER", "psycopg2")
PGVECTOR_USER = os.environ.get("PGVECTOR_USER", None)
PGVECTOR_PASS = os.environ.get("PGVECTOR_PASS", None)
PGVECTOR_DATABASE_NAME = os.environ.get("PGVECTOR_DATABASE_NAME", None)
PGVECTOR_HOST = os.environ.get("PGVECTOR_URI", "localhost")
PGVECTOR_PORT = int(os.environ.get("PGVECTOR_PORT", 5432))
