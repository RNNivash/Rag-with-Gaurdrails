import logging
from typing import List, Dict, Any
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# Standard naive prompt without security guardrails or PII restrictions
SYSTEM_PROMPT = """You are a helpful assistant.
Answer the user's question directly and thoroughly based on the provided document context."""

def build_context_string(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant context found in documents."
    
    parts = []
    for i, chunk in enumerate(chunks, 1):
        filename = chunk.get("filename", "Unknown source")
        raw_text = chunk.get("text", "").strip()
        parts.append(f"[Source {i}: {filename}]\n{raw_text}")
    return "\n\n---\n\n".join(parts)

def generate_rag_response(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Standard naive generation: directly feeds retrieved context to the LLM without guardrails."""
    if not settings.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set. Please set it in your .env file.")

    client = OpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY
    )

    context_str = build_context_string(chunks)
    user_prompt = f"""Context:
{context_str}

User Question: {query}
Answer:"""

    try:
        response = client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        message = response.choices[0].message
        content = message.content
        if not content and hasattr(message, "reasoning") and message.reasoning:
            content = message.reasoning
        return (content or "").strip()
    except Exception as e:
        logger.error(f"OpenRouter LLM generation error: {e}")
        raise
