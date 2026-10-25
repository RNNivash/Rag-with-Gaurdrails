# Advanced RAG with Guardrails

A complete Retrieval-Augmented Generation (RAG) system with document ingestion, MongoDB Vector Search, and OpenRouter LLM generation, featuring an interactive Streamlit UI and a robust FastAPI backend.

---

## 🏛️ Architecture Overview

```
 ┌─────────────────┐       HTTP Upload / Query      ┌─────────────────────────┐
 │                 │ ─────────────────────────────> │                         │
 │  Streamlit UI   │                                │    FastAPI Backend      │
 │ (streamlit_app) │ <───────────────────────────── │       (app/main)        │
 └─────────────────┘           JSON Response        └────────────┬────────────┘
                                                                 │
                                ┌────────────────────────────────┴────────────────────────────────┐
                                ▼                                                                 ▼
                   ┌─────────────────────────┐                                      ┌───────────────────────────┐
                   │   Document Ingestion    │                                      │   Retrieval & Generation  │
                   │    (app/ingestion)      │                                      │ (app/retrieval,generation)│
                   └────────────┬────────────┘                                      └─────────────┬─────────────┘
                                │                                                                 │
               SentenceTransformers (MiniLM-L6-v2)                                                │
                                │                                                                 │
                                ▼                                                                 │
                     ┌─────────────────────┐                                                      │
                     │  MongoDB Atlas DB   │ <────────────────────────────────────────────────────┘
                     │   (Vector Search)   │
                     └─────────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ OpenRouter API (LLM)│
                     │ (Llama 3.3 / Mistral│
                     └─────────────────────┘
```

---

## 📁 Project Structure

```
├── app/
│   ├── __init__.py
│   ├── config.py         # Environment settings and constants
│   ├── database.py       # MongoDB Atlas connection manager
│   ├── ingestion.py      # PDF text extraction, chunking, and embedding
│   ├── retrieval.py      # MongoDB Atlas vector search (with fallback)
│   ├── generation.py     # OpenRouter LLM completions with prompt guardrails
│   └── main.py           # FastAPI server endpoints (Health, Upload, Query)
├── streamlit_app.py      # Frontend interactive chat & PDF uploader
├── requirements.txt      # Python dependencies
├── .env                  # Private secrets (ignored by git)
├── .env.example          # Environment variable template
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (or 3.11 / 3.12)
- MongoDB Atlas cluster account (or local MongoDB)
- OpenRouter API Key ([https://openrouter.ai](https://openrouter.ai))

### 2. Configure Secrets
Copy `.env.example` into `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=rag_db
MONGODB_COLLECTION_NAME=documents
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
API_HOST=http://127.0.0.1:8000
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or with uv: `uv pip install -r requirements.txt`)*

---

## 🏃 Running the Application

### 1. Launch FastAPI Backend
```bash
uvicorn app.main:app --reload --port 8000
```
- API Documentation (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Launch Streamlit Chat Interface
In a second terminal:
```bash
streamlit run streamlit_app.py
```
Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🛡️ Guardrails & Safety
1. **Context-Grounding Guardrails**: System instructions enforce strict grounding to retrieved context chunks, preventing out-of-domain hallucinations.
2. **Graceful Fallbacks**: If MongoDB Atlas `$vectorSearch` index is not yet built, the retrieval service seamlessly uses in-collection exact cosine similarity.
3. **Environment Isolation**: `.gitignore` strictly protects `.env`, virtualenvs, and uploaded files from being committed.
