# Run this standalone script to parse textbooks and upload embeddings to Qdrant
import asyncio
from app.retrieval.pdf_parser import parse_pdf_directory
from app.retrieval.qdrant_service import QdrantService

def ingest():
    print("Starting Ingestion Process...")
    
    # Target directory specified by user
    textbook_dir = r"C:\Users\krish\Downloads\cbse_class_12_physics_tb"
    
    # 1. Parse Directory and get chunks
    print(f"Scanning directory: {textbook_dir}")
    chunks = parse_pdf_directory(textbook_dir)
    print(f"Total chunks extracted: {len(chunks)}")
    
    if chunks:
        # 2. Upload to Qdrant
        print("Connecting to Qdrant Cloud...")
        qdrant_service = QdrantService()
        qdrant_service.ingest_chunks(chunks)
    else:
        print("Nothing to ingest. Make sure the path is correct and contains PDFs.")

if __name__ == "__main__":
    ingest()
