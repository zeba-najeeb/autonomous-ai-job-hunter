import os
import re
import time
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from google import genai

from src.profile.parser import PROJECT_ROOT, chunk_cv_text, extract_text_from_pdf

# Load environment variables from C:\krish_naik\coursera\job_hunt\.env
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Persistent directory where ChromaDB stores vector embeddings on disk
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "candidate_cvs"


def get_gemini_client() -> genai.Client:
    """Initializes and returns the Google GenAI client."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise ValueError(
            "Valid GEMINI_API_KEY is missing in .env! Please check your .env file."
        )
    return genai.Client(api_key=api_key)


def normalize_candidate_id(candidate_name: str) -> str:
    """
    Converts a candidate's name or session ID into a clean, safe identifier
    (e.g., 'Zeba Najeeb' -> 'zeba_najeeb').
    """
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", candidate_name.strip().lower()).strip("_")
    return clean or "default_candidate"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generates vector embeddings for a list of text strings using Google's
    gemini-embedding-001 model.
    """
    client = get_gemini_client()
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
    )
    return [embedding.values for embedding in response.embeddings]


def get_cv_collection() -> chromadb.Collection:
    """
    Opens (or creates) the persistent ChromaDB collection configured with
    Cosine Similarity ('hnsw:space': 'cosine').
    """
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def store_cv_in_vector_db(
    cv_path: str | Path,
    candidate_name: str = "Zeba Najeeb",
) -> dict:
    """
    Extracts text from a PDF CV, splits it into overlapping chunks, embeds them
    with Gemini, and upserts them into ChromaDB with a sliding `last_active_at` timestamp.
    """
    candidate_id = normalize_candidate_id(candidate_name)
    current_timestamp = int(time.time())

    # 1. Extract and chunk the CV using Module 1's parser
    cv_text = extract_text_from_pdf(cv_path)
    chunks = chunk_cv_text(cv_text, chunk_size=800, overlap=150)

    # 2. Generate vector embeddings for all chunks in one batch API call
    embeddings = embed_texts(chunks)

    # 3. Prepare unique IDs and metadata (including last_active_at for 7-day sliding TTL)
    chunk_ids = [f"{candidate_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "chunk_index": i,
            "source_file": Path(cv_path).name,
            "last_active_at": current_timestamp,
        }
        for i in range(len(chunks))
    ]

    # 4. Upsert into ChromaDB (updates if already exists, inserts if new)
    collection = get_cv_collection()
    collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {
        "candidate_id": candidate_id,
        "chunks_stored": len(chunks),
        "embedding_dimension": len(embeddings[0]) if embeddings else 0,
    }


def refresh_candidate_activity(candidate_name: str) -> None:
    """
    Sliding Window Reset: Updates `last_active_at` to the current time whenever
    a returning candidate searches jobs or moves to the next portal link.
    """
    candidate_id = normalize_candidate_id(candidate_name)
    collection = get_cv_collection()
    existing = collection.get(
        where={"candidate_id": candidate_id},
        include=["metadatas"],
    )
    ids = existing.get("ids", [])
    metadatas = existing.get("metadatas", [])

    if ids and metadatas:
        now = int(time.time())
        for meta in metadatas:
            meta["last_active_at"] = now
        collection.update(ids=ids, metadatas=metadatas)


def retrieve_relevant_cv_chunks(
    job_description_query: str,
    candidate_name: str = "Zeba Najeeb",
    top_k: int = 3,
) -> list[dict]:
    """
    Embeds a job description (or requirement query), resets the candidate's
    7-day sliding activity timer, and retrieves the top_k most relevant CV chunks.
    """
    candidate_id = normalize_candidate_id(candidate_name)
    collection = get_cv_collection()

    # Reset the 7-day sliding inactivity window since the candidate is actively searching
    refresh_candidate_activity(candidate_name)

    # 1. Embed the job description query using the exact same Gemini model
    query_embedding = embed_texts([job_description_query])[0]

    # 2. Query ChromaDB filtered strictly by candidate_id (Multi-Tenant Isolation)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"candidate_id": candidate_id},
        include=["documents", "metadatas", "distances"],
    )

    # 3. Format results with Cosine Similarity score (1 - cosine_distance)
    matched_chunks: list[dict] = []
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for doc, dist, meta in zip(documents, distances, metadatas):
        similarity_score = round((1.0 - float(dist)) * 100, 2)
        matched_chunks.append(
            {
                "chunk_index": meta.get("chunk_index"),
                "similarity_percent": similarity_score,
                "text": doc,
            }
        )

    return matched_chunks


def delete_candidate_cv(candidate_name: str) -> str:
    """
    Explicit Cleanup: Deletes all CV vector chunks belonging to a candidate
    when they finish all links or click 'Clear My Data'.
    """
    candidate_id = normalize_candidate_id(candidate_name)
    collection = get_cv_collection()
    collection.delete(where={"candidate_id": candidate_id})
    return candidate_id


def cleanup_inactive_sessions(max_inactive_days: int = 7) -> int:
    """
    Sliding 7-Day TTL Auto-Cleanup: Deletes candidate CV chunks only if the
    candidate has been completely inactive for `max_inactive_days` (default: 7 days).
    """
    cutoff_timestamp = int(time.time()) - (max_inactive_days * 86400)
    collection = get_cv_collection()

    expired = collection.get(
        where={"last_active_at": {"$lt": cutoff_timestamp}},
        include=[],
    )
    expired_ids = expired.get("ids", [])

    if expired_ids:
        collection.delete(ids=expired_ids)

    return len(expired_ids)


if __name__ == "__main__":
    sample_cv = PROJECT_ROOT / "data" / "ZebaNajeeb_CV.pdf"
    candidate = "Zeba Najeeb"

    print("=== Module 2: ChromaDB Vector Knowledge Base (7-Day Sliding TTL) ===")
    try:
        # 1. Run 7-day inactivity cleanup check
        removed_count = cleanup_inactive_sessions(max_inactive_days=7)
        print(f"1. Auto-Cleanup Check  : Removed {removed_count} inactive chunks (>7 days idle)")

        # 2. Store candidate's CV in ChromaDB
        print(f"2. Embedding and storing '{sample_cv.name}' in ChromaDB...")
        info = store_cv_in_vector_db(sample_cv, candidate_name=candidate)
        print(f"   -> Candidate ID        : {info['candidate_id']}")
        print(f"   -> Total Chunks Stored : {info['chunks_stored']}")
        print(f"   -> Vector Dimensions   : {info['embedding_dimension']} dims\n")

        # 3. Test semantic retrieval with a realistic AI Engineer job requirement
        sample_job_query = (
            "Looking for a Generative AI Engineer in Dubai with hands-on experience "
            "building RAG pipelines, LangChain, ChromaDB, FastAPI backend APIs, and LLMs."
        )
        print("3. Testing RAG Semantic Retrieval against Sample Job Query:")
        print(f"   Query: \"{sample_job_query}\"\n")

        top_matches = retrieve_relevant_cv_chunks(
            job_description_query=sample_job_query,
            candidate_name=candidate,
            top_k=3,
        )

        for rank, match in enumerate(top_matches, start=1):
            print(
                f"=== MATCH #{rank} (Chunk #{match['chunk_index']} | "
                f"Semantic Similarity: {match['similarity_percent']}%) ==="
            )
            print(match["text"][:300] + "...\n")

    except Exception as e:
        print(f"\nError: {e}")
