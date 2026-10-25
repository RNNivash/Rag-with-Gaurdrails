import streamlit as st
import requests
import os

API_HOST = os.getenv("API_HOST", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Base RAG Assistant",
    page_icon="📖",
    layout="wide"
)

# App Title & Header
st.title("📖 Base RAG Assistant (No Guardrails)")
st.caption("Ask questions powered by MongoDB Vector Search & OpenRouter LLM — Test for confidential data exposure")

# Sidebar for Setup & Document Upload
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Index PDF", use_container_width=True):
            with st.spinner("Extracting text, chunking, and embedding..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_HOST}/upload", files=files, timeout=60)
                    if response.status_code == 200:
                        data = response.json().get("data", {})
                        st.success(f"Indexed **{data.get('chunks_indexed', 0)}** chunks from `{uploaded_file.name}`!")
                    else:
                        st.error(f"Error {response.status_code}: {response.json().get('detail', response.text)}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

    st.divider()
    st.header("⚙️ System Status")
    try:
        health_res = requests.get(f"{API_HOST}/health", timeout=3)
        if health_res.status_code == 200:
            status_data = health_res.json()
            if status_data.get("mongodb_connected"):
                st.success("MongoDB: Connected ✅")
            else:
                st.warning("MongoDB: Not Connected ⚠️")
        else:
            st.warning("Backend API reachable but returned error.")
    except Exception:
        st.error("Backend API is currently offline ❌")

    st.caption("Configure `.env` with your `MONGODB_URI` and `OPENROUTER_API_KEY`.")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Retrieved Context Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {idx}** (`{src.get('filename')}` - Score: `{src.get('score', 'N/A')}`)")
                    st.text(src.get("text", "")[:400] + ("..." if len(src.get("text", "")) > 400 else ""))

# User query input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            try:
                res = requests.post(
                    f"{API_HOST}/query",
                    json={"query": prompt, "top_k": 4},
                    timeout=60
                )
                if res.status_code == 200:
                    result = res.json()
                    answer = result.get("answer", "No response provided.")
                    sources = result.get("sources", [])
                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 Retrieved Context Sources"):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(f"**Source {idx}** (`{src.get('filename')}` - Score: `{src.get('score', 'N/A')}`)")
                                st.text(src.get("text", "")[:400] + ("..." if len(src.get("text", "")) > 400 else ""))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    err_msg = f"API Error ({res.status_code}): {res.json().get('detail', res.text)}"
                    st.error(err_msg)
            except Exception as e:
                st.error(f"Request failed: {e}")
