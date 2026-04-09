import os
import fitz  # PyMuPDF
import re
from typing import List, Dict

def clean_text(text: str) -> str:
    # Basic cleanup
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Splits text into chunks of approximately `chunk_size` words with `overlap`."""
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks

def parse_pdf_directory(directory_path: str) -> List[Dict[str, str]]:
    """
    Recursively scans the directory for PDF files, extracts text,
    and returns a list of chunks with metadata.
    """
    all_chunks = []
    
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return all_chunks
        
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                try:
                    doc = fitz.open(file_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text = page.get_text("text")

                        clean_page_text = clean_text(text)
                        if not clean_page_text:
                            continue
                            
                        # Chunk the page text
                        page_chunks = chunk_text(clean_page_text, chunk_size=250, overlap=50)
                        
                        for i, chunk in enumerate(page_chunks):
                            all_chunks.append({
                                "content": chunk,
                                "metadata": {
                                    "source": file,
                                    "page": page_num + 1,
                                    "chunk_index": i
                                }
                            })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    return all_chunks
