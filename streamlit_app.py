"""
Streamlit Frontend Application for Agentic AI Knowledge Assistant.
Provides a modern chat interface, chat history, expander for retrieved context,
confidence scores, and backend fallback support (Google Gemini powered).
"""

from typing import Any
import requests
import streamlit as st
from src.config import settings

# Page Configuration
st.set_page_config(
    page_title="Agentic AI Knowledge Assistant (Gemini)",
    page_icon="🤖",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 Agentic AI Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Ask questions strictly grounded in the Agentic AI eBook (Powered by Google Gemini 3.5 Flash Lite). Out-of-scope questions will be refused.</div>',
    unsafe_allow_html=True
)

# Helper to clean response content
def clean_answer_text(raw_answer: Any) -> str:
    """Extract clean readable text from Gemini response structure or raw string."""
    if isinstance(raw_answer, str):
        # In case raw string representation of a list of dicts was passed
        if raw_answer.startswith("[") and raw_answer.endswith("]"):
            try:
                import ast
                parsed = ast.literal_eval(raw_answer)
                if isinstance(parsed, list):
                    return clean_answer_text(parsed)
            except Exception:
                pass
        return raw_answer.strip()
    
    if isinstance(raw_answer, list):
        text_parts = []
        for block in raw_answer:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts).strip()
        
    return str(raw_answer).strip()

# Backend URL
API_URL = f"http://{settings.FASTAPI_HOST}:{settings.FASTAPI_PORT}/chat"

# Sidebar Info
with st.sidebar:
    st.header("⚙️ System Status & Settings")
    st.markdown("**LLM Engine:** `gemini-3.5-flash-lite` (Google)")
    st.markdown("**Embedding Model:** `gemini-embedding-001` (3072-dim)")
    st.markdown("**Vector Store:** Pinecone (`agentic-ai-index`)")
    
    st.divider()
    st.markdown("### 🔍 Sample Test Queries")
    sample_queries = [
        "What is Agentic AI according to the eBook?",
        "How do AI agents differ from traditional automation systems?",
        "What are the core components of an Agentic Architecture?",
        "What role does memory play in Agentic AI workflows?",
        "Who won the 2022 FIFA World Cup?",
        "What is the population of Tokyo?"
    ]
    
    for q in sample_queries:
        if st.button(q, key=f"btn_{q}"):
            st.session_state.preset_query = q

    st.divider()
    st.caption("Powered by Google Gemini, LangChain, LangGraph & FastAPI")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Agentic AI Knowledge Assistant powered by Google Gemini. Ask me anything from the Agentic AI eBook!",
            "context": [],
            "score": None
        }
    ]

# Display Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        if msg.get("score") is not None:
            st.caption(f"📊 **Retrieval Relevance Score:** `{msg['score']:.4f}`")
            
        if msg.get("context"):
            with st.expander("📚 View Retrieved Context Chunks (Top 3)"):
                for idx, chunk in enumerate(msg["context"], 1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.info(chunk)

# Handle Preset Query Selection from Sidebar
user_input = None
if "preset_query" in st.session_state:
    user_input = st.session_state.pop("preset_query")

# Chat Input Box
chat_input = st.chat_input("Ask a question about Agentic AI...")
if chat_input:
    user_input = chat_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching document & generating response via Gemini..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": user_input},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = clean_answer_text(data["final_answer"])
                    context = data.get("retrieved_context", [])
                    score = data.get("confidence_score", 0.0)
                    
                    st.write(answer)
                    st.caption(f"📊 **Retrieval Relevance Score:** `{score:.4f}`")
                    
                    if context:
                        with st.expander("📚 View Retrieved Context Chunks (Top 3)"):
                            for idx, chunk in enumerate(context, 1):
                                st.markdown(f"**Chunk {idx}:**")
                                st.info(chunk)
                                
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "context": context,
                        "score": score
                    })
                else:
                    err_msg = f"❌ Server returned error code {response.status_code}: {response.text}"
                    st.error(err_msg)
                    
            except requests.exceptions.ConnectionError:
                st.warning("⚠️ FastAPI server is unreachable at " + API_URL + ". Attempting direct graph execution...")
                try:
                    from src.graph import run_rag_pipeline
                    result = run_rag_pipeline(user_input)
                    answer = clean_answer_text(result["final_answer"])
                    context = result["retrieved_context"]
                    score = result["confidence_score"]
                    
                    st.write(answer)
                    st.caption(f"📊 **Retrieval Relevance Score:** `{score:.4f}`")
                    if context:
                        with st.expander("📚 View Retrieved Context Chunks (Top 3)"):
                            for idx, chunk in enumerate(context, 1):
                                st.markdown(f"**Chunk {idx}:**")
                                st.info(chunk)
                                
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "context": context,
                        "score": score
                    })
                except Exception as ex:
                    st.error(f"❌ Could not process request: {ex}")
