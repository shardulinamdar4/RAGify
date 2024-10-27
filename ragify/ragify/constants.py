import os
import pathlib

import torch
from dotenv import load_dotenv
from langchain.text_splitter import Language
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)

load_dotenv()

# PATHS
DIRECTORY_PATH = pathlib.Path(os.path.dirname(__file__)).parent
GOOGLEDRIVE_REPOSITORY_PATH = KNOWLEDGE_REPOSITORY_PATH / "googledrive"

# INGEST
INGEST_THREADS = 8
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 250
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DOCUMENT_MAP = {
    ".txt": {
        "loader": TextLoader,
        "language": None,
    },
    ".html": {
        "loader": TextLoader,
        "language": Language.HTML,
    },
    ".md": {
        "loader": TextLoader,
        "language": Language.MARKDOWN,
    },
    ".py": {
        "loader": TextLoader,
        "language": Language.PYTHON,
    },
    ".pdf": {
        "loader": PDFMinerLoader,
        "language": None,
    },
    ".csv": {
        "loader": CSVLoader,
    },
    ".xls": {
        "loader": UnstructuredExcelLoader,
    },
    ".xlsx": {
        "loader": UnstructuredExcelLoader,
    },
    ".docx": {
        "loader": Docx2txtLoader,
        "language": None,
    },
    ".doc": {
        "loader": Docx2txtLoader,
        "language": None,
    },
    ".pptx": {
        "loader": UnstructuredPowerPointLoader,
    },
    ".ppt": {
        "loader": UnstructuredPowerPointLoader,
    },
}


# PGVECTOR
PGVECTOR_DRIVER = os.environ.get("PGVECTOR_DRIVER", "psycopg2")
PGVECTOR_USER = os.environ.get("PGVECTOR_USER", None)
PGVECTOR_PASS = os.environ.get("PGVECTOR_PASS", None)
PGVECTOR_DATABASE_NAME = os.environ.get("PGVECTOR_DATABASE_NAME", None)
PGVECTOR_HOST = os.environ.get("PGVECTOR_URI", "localhost")
PGVECTOR_PORT = int(os.environ.get("PGVECTOR_PORT", 5432))
