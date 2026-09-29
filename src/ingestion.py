"""
Document Ingestion Module for Agentic AI Knowledge Assistant.
Extracts text from PDF, splits text into chunks, generates Google Gemini embeddings with rate-limit
handling (batching & exponential backoff), and stores vectors with deterministic IDs in Pinecone.
"""

import os
import sys
import time
import hashlib
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import settings


def load_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found at path: {pdf_path}. "
            f"Please place your PDF file in the 'data/' directory."
        )
    
    print(f"[PDF] Loading PDF document from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[OK] Loaded {len(documents)} page(s) from PDF.")
    return documents


def split_documents(documents: List):
    print(
        f"[SPLIT] Splitting documents with chunk_size={settings.CHUNK_SIZE} "
        f"and chunk_overlap={settings.CHUNK_OVERLAP}..."
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"[OK] Document split into {len(chunks)} chunk(s).")
    return chunks


def get_embeddings():
    """
    Initialize Google Generative AI Embeddings model (native 3072 dimensions).
    """
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY
    )


def generate_deterministic_id(chunk_index: int, page_content: str) -> str:
    """
    Generate a stable, unique deterministic ID for each document chunk using MD5 hash.
    This prevents duplicate records when ingestion is rerun or resumed.
    """
    hash_digest = hashlib.md5(page_content.encode('utf-8')).hexdigest()[:12]
    return f"doc_chunk_{chunk_index:04d}_{hash_digest}"


def initialize_pinecone_index():
    """
    Initialize Pinecone index with settings.EMBEDDING_DIMENSION (3072).
    If an existing index has a mismatched dimension, delete and recreate it safely.
    """
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    existing_indexes = pc.list_indexes()
    existing_index_names = [index_info["name"] for index_info in existing_indexes]
    
    if settings.PINECONE_INDEX_NAME in existing_index_names:
        index_description = pc.describe_index(settings.PINECONE_INDEX_NAME)
        current_dim = index_description.dimension
        
        if current_dim != settings.EMBEDDING_DIMENSION:
            print(
                f"[WARN] Pinecone index '{settings.PINECONE_INDEX_NAME}' dimension ({current_dim}) "
                f"does not match required Gemini embedding dimension ({settings.EMBEDDING_DIMENSION})."
            )
            print(f"[RECREATE] Recreating index '{settings.PINECONE_INDEX_NAME}' with {settings.EMBEDDING_DIMENSION} dimensions...")
            pc.delete_index(settings.PINECONE_INDEX_NAME)
            time.sleep(5)
            existing_index_names.remove(settings.PINECONE_INDEX_NAME)
            
    if settings.PINECONE_INDEX_NAME not in existing_index_names:
        print(f"[INDEX] Creating Pinecone index: '{settings.PINECONE_INDEX_NAME}' (dimension={settings.EMBEDDING_DIMENSION})...")
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"[OK] Index '{settings.PINECONE_INDEX_NAME}' created successfully.")
        time.sleep(5)  # Wait for index to become ready
    else:
        print(f"[INFO] Pinecone index '{settings.PINECONE_INDEX_NAME}' ready (dimension={settings.EMBEDDING_DIMENSION}).")


def add_batch_with_retry(
    vector_store: PineconeVectorStore,
    batch_chunks: List,
    batch_ids: List[str],
    batch_num: int,
    total_batches: int,
    max_retries: int = 5,
    initial_delay: float = 5.0
):
    """
    Add a batch of chunks to Pinecone with exponential backoff for HTTP 429 rate limit errors.
    """
    delay = initial_delay
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[UPLOAD] Uploading Batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks) [Attempt {attempt}/{max_retries}]...")
            vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
            print(f"[OK] Batch {batch_num}/{total_batches} successfully ingested!")
            return True
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "Quota exceeded" in error_msg or "rate limit" in error_msg.lower():
                print(f"[WARN] Rate limit encountered on Batch {batch_num}/{total_batches} (Attempt {attempt}). Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
                delay *= 2.0  # Exponential backoff
            else:
                print(f"[ERROR] Batch {batch_num}/{total_batches} failed with error: {e}")
                if attempt == max_retries:
                    raise e
                time.sleep(delay)
                delay *= 1.5

    raise RuntimeError(f"Batch {batch_num}/{total_batches} failed after {max_retries} retries due to persistent rate limit errors.")


def ingest_pdf(pdf_path: Optional[str] = None, batch_size: int = 15, delay_between_batches: float = 3.0):
    """
    Execute full ingestion pipeline with batching and exponential backoff retry.
    """
    settings.validate_keys()
    path = pdf_path or settings.PDF_PATH
    
    # 1. Load PDF
    documents = load_pdf(path)
    
    # 2. Split text
    chunks = split_documents(documents)
    
    # 3. Setup Pinecone Index
    initialize_pinecone_index()
    
    # 4. Generate Deterministic Vector IDs
    chunk_ids = [
        generate_deterministic_id(idx, chunk.page_content)
        for idx, chunk in enumerate(chunks)
    ]
    
    # 5. Initialize Vector Store Connection
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    
    # 6. Batched Ingestion with Exponential Backoff Retries
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size
    
    print(f"[START] Batched ingestion of {total_chunks} chunks in {total_batches} batches (batch_size={batch_size})...")
    
    successful_chunks = 0
    
    for i in range(0, total_chunks, batch_size):
        batch_num = (i // batch_size) + 1
        batch_chunks = chunks[i : i + batch_size]
        batch_ids = chunk_ids[i : i + batch_size]
        
        add_batch_with_retry(
            vector_store=vector_store,
            batch_chunks=batch_chunks,
            batch_ids=batch_ids,
            batch_num=batch_num,
            total_batches=total_batches
        )
        
        successful_chunks += len(batch_chunks)
        
        if i + batch_size < total_chunks:
            print(f"[WAIT] Sleeping {delay_between_batches}s to prevent rate limit...")
            time.sleep(delay_between_batches)
            
    print(f"\n[DONE] Ingestion complete! Successfully processed {successful_chunks}/{total_chunks} chunks into Pinecone index '{settings.PINECONE_INDEX_NAME}'.")
    return vector_store


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as e:
        print(f"[FAIL] Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)

