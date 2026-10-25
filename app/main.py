import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.database import Database
from app.ingestion import ingest_pdf_file
from app.retrieval import retrieve_relevant_chunks
from app.generation import generate_rag_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG with Guardrails API",
    description="Vector search with MongoDB and LLM answering via OpenRouter",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[dict]

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API service is up and running."}

@app.get("/health")
def health_check():
    db_connected = Database.ping()
    return {
        "status": "healthy" if db_connected else "degraded",
        "mongodb_connected": db_connected
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are currently supported.")

    try:
        content = await file.read()
        res = ingest_pdf_file(filename=file.filename, file_bytes=content)
        return {
            "status": "success",
            "message": f"Successfully processed and embedded {file.filename}",
            "data": res
        }
    except Exception as e:
        logger.exception("Error processing uploaded PDF")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        chunks = retrieve_relevant_chunks(query=req.query, top_k=req.top_k or 4)
        answer = generate_rag_response(query=req.query, chunks=chunks)
        return QueryResponse(
            query=req.query,
            answer=answer,
            sources=chunks
        )
    except Exception as e:
        logger.exception("Error answering query")
        raise HTTPException(status_code=500, detail=str(e))
