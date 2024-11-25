import os
from bs4 import BeautifulSoup  # For HTML parsing
from PyPDF2 import PdfReader  # For PDF parsing

def read_txt(file_path):
    """Reads plain text files."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf(file_path):
    """Reads PDF files and extracts text."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_html(file_path):
    """Reads HTML files and extracts text content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    return soup.get_text()

def normalize_document(file_path):
    """Detects file type and extracts plain text."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.txt':
        return read_txt(file_path)
    elif ext == '.pdf':
        return read_pdf(file_path)
    elif ext == '.html':
        return read_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
