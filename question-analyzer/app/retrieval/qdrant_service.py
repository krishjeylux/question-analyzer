import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter
from sentence_transformers import SentenceTransformer
from app.core.config import settings

class QdrantService:
    def __init__(self):
        # Connect to the cloud cluster using credentials from .env
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        # BGE or MiniLM works well for standard RAG. Using a lightweight, performant model.
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = "cbse_physics_textbook"
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                # Using 384 dimensions for all-MiniLM-L6-v2
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                print(f"Created new collection: {self.collection_name}")
        except Exception as e:
            print(f"Error checking/creating collection: {e}")

    def ingest_chunks(self, chunks: List[Dict[str, str]]):
        if not chunks:
            print("No chunks to ingest.")
            return

        print(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        points = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[i].tolist(),
                    payload={
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "page": chunk["metadata"]["page"]
                    }
                )
            )

        print("Uploading points to Qdrant Cloud...")
        # Uploading in batches to avoid overwhelming the network
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"Uploaded batch {i//batch_size + 1}/{len(points)//batch_size + 1}")
            
        print("Ingestion complete.")

    def search_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vector = self.encoder.encode([query])[0]
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k
        )
        
        results = []
        for hit in search_result.points:
            results.append({
                "content": hit.payload.get("content", ""),
                "source": hit.payload.get("source", ""),
                "page": hit.payload.get("page", ""),
                "score": hit.score
            })
            
        return results
