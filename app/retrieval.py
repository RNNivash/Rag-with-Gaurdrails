import logging
import numpy as np
from typing import List, Dict, Any
from app.database import Database
from app.ingestion import get_embedding_model

logger = logging.getLogger(__name__)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a)
    vb = np.array(b)
    dot = np.dot(va, vb)
    norm = (np.linalg.norm(va) * np.linalg.norm(vb))
    if norm == 0:
        return 0.0
    return float(dot / norm)

def retrieve_relevant_chunks(
    query: str,
    top_k: int = 4,
    index_name: str = "vector_index"
) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k most relevant chunks for a user query.
    Attempts MongoDB Atlas $vectorSearch first; falls back gracefully to
    in-memory cosine similarity if Atlas Vector Index isn't configured.
    """
    model = get_embedding_model()
    query_vector = model.encode(query, show_progress_bar=False).tolist()
    collection = Database.get_collection()

    # 1. Attempt Atlas $vectorSearch
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": max(top_k * 10, 20),
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "filename": 1,
                    "chunk_index": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        if results:
            logger.info(f"Atlas Vector Search returned {len(results)} results.")
            return results
    except Exception as e:
        logger.warning(f"Atlas $vectorSearch unavailable or failed ({e}). Falling back to exact vector comparison.")

    # 2. Fallback: Exact Cosine Similarity across documents in collection
    cursor = collection.find({}, {"_id": 0, "filename": 1, "chunk_index": 1, "text": 1, "embedding": 1})
    scored_docs = []
    for doc in cursor:
        emb = doc.get("embedding")
        if emb:
            sim = cosine_similarity(query_vector, emb)
            scored_docs.append({
                "filename": doc.get("filename"),
                "chunk_index": doc.get("chunk_index"),
                "text": doc.get("text"),
                "score": round(sim, 4)
            })

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[:top_k]
