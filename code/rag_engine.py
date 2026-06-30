import os
import glob
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# CONFIGURATION

KNOWLEDGE_BASE_DIR = "knowledge_base"
CHROMA_DB_DIR = "knowledge_base/chroma_db"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
COLLECTION_NAME = "clinical_knowledge"
N_RESULTS = 3


# TEXT CHUNKING

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits text into overlapping word-based chunks.
    Overlap ensures context is not lost at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# LOAD AND INDEX KNOWLEDGE BASE

def build_knowledge_base():
    """
    Loads all .txt files from knowledge_base folder,
    chunks them, embeds them, and stores in a persistent
    ChromaDB database on disk.

    On subsequent runs: loads existing database instantly
    without re-embedding — no redundant processing.

    Returns the ChromaDB collection.
    """
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Persistent client saves to disk
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Check if collection already exists on disk
    existing_names = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing_names:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )
        print(f"Knowledge base loaded from disk ({collection.count()} chunks).")
        return collection

    # First run — build and embed from scratch
    print("Building knowledge base for the first time...")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    txt_files = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.txt"))

    if not txt_files:
        print(f"Warning: No .txt files found in {KNOWLEDGE_BASE_DIR}/")
        return collection

    all_chunks = []
    all_ids = []
    all_metadata = []

    for filepath in txt_files:
        filename = Path(filepath).stem
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({
                "source": filename,
                "chunk_index": i
            })

    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadata
    )

    print(f"Knowledge base built: {len(all_chunks)} chunks from {len(txt_files)} files.")
    return collection


# RETRIEVAL

def retrieve_context(collection, query, n_results=N_RESULTS):
    """
    Takes a query string, retrieves the most semantically
    similar chunks from the knowledge base.
    Returns a formatted string ready to inject into the LLM prompt.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    if not results or not results["documents"][0]:
        return ""

    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    context_parts = []
    for chunk, source in zip(chunks, sources):
        context_parts.append(f"[Source: {source}]\n{chunk}")

    return "\n\n---\n\n".join(context_parts)


# BUILD RETRIEVAL QUERY FROM ML RESULTS

def build_retrieval_query(model_results, clinical_context, llm_recommendation):
    """
    Constructs a meaningful query from ML results to retrieve
    the most relevant knowledge base chunks.
    """
    task_type = llm_recommendation.get("task_type", "")
    model_name = llm_recommendation.get("recommended_model", "")
    rationale = llm_recommendation.get("clinical_rationale", "")

    top_features = []
    if model_results.get("feature_importances"):
        top_features = list(model_results["feature_importances"].keys())[:3]

    n_clusters = model_results.get("n_clusters", "")

    query_parts = [
        clinical_context,
        f"Task: {task_type} using {model_name}",
        rationale,
    ]

    if top_features:
        query_parts.append(f"Top predictive features: {', '.join(top_features)}")

    if n_clusters:
        query_parts.append(f"Patient segmentation into {n_clusters} clusters")

    return " ".join(filter(None, query_parts))