import io
import uuid
import logging
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.database import Database

logger = logging.getLogger(__name__)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    return _embedding_model

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text content from uploaded PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            extracted_text.append(text)
    return "\n\n".join(extracted_text)

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """Splits raw text into manageable overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def ingest_pdf_file(filename: str, file_bytes: bytes) -> Dict[str, Any]:
    """Extracts text from a PDF, chunks it, generates embeddings, and saves to MongoDB."""
    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        raise ValueError("Could not extract any readable text from the uploaded PDF.")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text chunks generated from the document.")

    model = get_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    collection = Database.get_collection()
    docs_to_insert = []
    doc_group_id = str(uuid.uuid4())

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        docs_to_insert.append({
            "group_id": doc_group_id,
            "filename": filename,
            "chunk_index": idx,
            "text": chunk,
            "embedding": embedding
        })

    result = collection.insert_many(docs_to_insert)
    logger.info(f"Ingested {len(docs_to_insert)} chunks for file {filename}")
    return {
        "filename": filename,
        "chunks_indexed": len(result.inserted_ids),
        "group_id": doc_group_id
    }
