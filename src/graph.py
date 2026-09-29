"""
LangGraph Workflow Module for Agentic AI Knowledge Assistant.
Implements StateGraph with retrieval node, relevance scoring, and grounded generation node
powered by Google Gemini API.
"""

from typing import List, Dict, Any, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END, START

from src.config import settings


class AgentState(TypedDict):
    """
    Graph State defining data passed between nodes.
    - question: The user's input query.
    - context: List of retrieved context chunk strings.
    - answer: The final output answer from LLM or refusal message.
    - score: Confidence/relevance score (0.0 to 1.0).
    - retrieved_sources: List of metadata sources for retrieved chunks.
    """
    question: str
    context: List[str]
    answer: str
    score: float
    retrieved_sources: List[Dict[str, Any]]


def get_vectorstore():
    """Get Pinecone vector store connection using Google Gemini Embeddings (3072-dim)."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY
    )
    return PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings
    )


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieval Node:
    Fetches top 3 relevant chunks from Pinecone vector store along with similarity scores.
    """
    question = state["question"]
    
    try:
        vectorstore = get_vectorstore()
        results_with_scores = vectorstore.similarity_search_with_score(
            query=question,
            k=settings.TOP_K_RESULTS
        )
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return {
            "context": [],
            "score": 0.0,
            "retrieved_sources": []
        }
    
    context_chunks = []
    sources = []
    similarity_scores = []
    
    for doc, score in results_with_scores:
        context_chunks.append(doc.page_content)
        similarity_scores.append(float(score))
        sources.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": round(float(score), 4)
        })
    
    avg_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    normalized_score = round(max(0.0, min(1.0, avg_score)), 4)
    
    return {
        "context": context_chunks,
        "score": normalized_score,
        "retrieved_sources": sources
    }


def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generation Node:
    Generates answer using ChatGoogleGenerativeAI (gemini-2.5-flash) strictly based on retrieved context.
    Refuses if context is missing or irrelevant.
    """
    question = state["question"]
    context_chunks = state.get("context", [])
    relevance_score = state.get("score", 0.0)
    
    REFUSAL_RESPONSE = "I cannot answer based on the provided document."
    
    if not context_chunks or relevance_score < 0.35:
        return {
            "answer": REFUSAL_RESPONSE,
            "score": relevance_score
        }
    
    joined_context = "\n\n---\n\n".join(context_chunks)
    
    system_prompt = (
        "You are an expert AI assistant answering questions about the Agentic AI eBook.\n\n"
        "GUIDELINES FOR ANSWERING:\n"
        "1. Answer using ONLY information directly supported by the provided context below.\n"
        "2. If the context supports a partial answer (for example, listing some components or details mentioned in the text), provide that partial answer based on the context.\n"
        "3. If the retrieved context is incomplete for a full answer, explicitly state that the document context provided is partial or incomplete.\n"
        "4. NEVER invent missing components, extrapolate, or use outside/general knowledge.\n"
        "5. Refuse ONLY if the provided context contains NO useful or relevant information for answering the question. In that case, respond EXACTLY with:\n"
        "\"I cannot answer based on the provided document.\"\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=0.0,
        google_api_key=settings.GEMINI_API_KEY
    )
    
    chain = prompt | llm
    
    response = chain.invoke({
        "context": joined_context,
        "question": question
    })
    
    content = response.content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        answer_text = "".join(text_parts).strip()
    else:
        answer_text = str(content).strip()
    
    return {
        "answer": answer_text
    }


def build_rag_graph():
    """Build and compile the LangGraph StateGraph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


rag_app = build_rag_graph()


def run_rag_pipeline(question: str) -> Dict[str, Any]:
    """Execute the RAG pipeline for a given query."""
    initial_state = {
        "question": question,
        "context": [],
        "answer": "",
        "score": 0.0,
        "retrieved_sources": []
    }
    
    final_state = rag_app.invoke(initial_state)
    
    return {
        "final_answer": final_state.get("answer", "I cannot answer based on the provided document."),
        "retrieved_context": final_state.get("context", []),
        "confidence_score": final_state.get("score", 0.0),
        "retrieved_sources": final_state.get("retrieved_sources", [])
    }
