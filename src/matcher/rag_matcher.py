import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.portals.queue_manager import DB_PATH, get_db_connection
from src.profile.parser import PROJECT_ROOT
from src.profile.vector_store import (
    get_cv_collection,
    normalize_candidate_id,
    retrieve_relevant_cv_chunks,
    store_cv_in_vector_db,
)
from src.scrapers.job_fetcher import ActiveJobPosting, get_saved_active_jobs

# Load environment variables from C:\krish_naik\coursera\job_hunt\.env
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Threshold for unlocking Phase 4 On-Demand Career Co-Pilot
HIGH_MATCH_THRESHOLD: int = 70

# Production Model Fallback Chain (prevents 503 High Demand crashes)
MODEL_FALLBACK_CHAIN: list[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-3.8-flash"
]



# =====================================================================
# 1. Pydantic Schemas for RAG Semantic Match Scoring
# =====================================================================

class SingleJobEvaluation(BaseModel):
    """Structured RAG evaluation comparing one job against retrieved CV chunks."""
    fingerprint: str = Field(
        description="Exact job fingerprint passed in the prompt for this job."
    )
    match_score: int = Field(
        ge=0,
        le=100,
        description="Overall semantic & technical fit score from 0 to 100 based strictly on retrieved CV chunks.",
    )
    matching_skills: list[str] = Field(
        description="Specific technical skills, tools, and domain strengths from the job requirements that are verified in the candidate's CV chunks."
    )
    missing_skills: list[str] = Field(
        description="Specific skills, tools, or domain requirements mentioned in the job that are missing or not evidenced in the candidate's CV chunks."
    )
    match_reasoning: str = Field(
        description="Concise 2-sentence architectural explanation of why this score was awarded."
    )


class BatchJobEvaluationList(BaseModel):
    """Batch response schema for evaluating multiple jobs against retrieved CV chunks."""
    evaluations: list[SingleJobEvaluation]


class RankedJobMatch(BaseModel):
    """Complete ranked job record combining job metadata and RAG evaluation results."""
    fingerprint: str
    portal_name: str
    job_title: str
    company_name: str
    location: str
    posted_date_raw: str
    posted_date_iso: str
    days_ago: int
    job_url: str
    description_summary: str
    match_score: int
    vector_similarity_avg: float
    matching_skills: list[str]
    missing_skills: list[str]
    match_reasoning: str
    copilot_unlocked: bool


# =====================================================================
# 2. SQLite Table Initialization for RAG Job Evaluations
# =====================================================================

def init_evaluations_table(db_path: Path = DB_PATH) -> None:
    """
    Creates the `job_evaluations` table in SQLite to persist RAG match scores,
    skill gaps, and Co-Pilot eligibility (Score >= 70%).
    """
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                portal_name TEXT NOT NULL,
                job_title TEXT NOT NULL,
                company_name TEXT NOT NULL,
                location TEXT NOT NULL,
                posted_date_raw TEXT NOT NULL,
                posted_date_iso TEXT NOT NULL,
                days_ago INTEGER NOT NULL,
                job_url TEXT NOT NULL,
                description_summary TEXT NOT NULL,
                match_score INTEGER NOT NULL,
                vector_similarity_avg REAL NOT NULL,
                matching_skills_json TEXT NOT NULL,
                missing_skills_json TEXT NOT NULL,
                match_reasoning TEXT NOT NULL,
                copilot_unlocked INTEGER NOT NULL,
                evaluated_at TEXT NOT NULL,
                UNIQUE(candidate_id, fingerprint)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_job_eval_candidate_score
            ON job_evaluations (candidate_id, match_score DESC)
            """
        )
        conn.commit()


# =====================================================================
# 3. Ensure Candidate CV Vectors Exist in ChromaDB
# =====================================================================

def ensure_cv_indexed(
    candidate_name: str = "Zeba Najeeb",
    default_cv_path: Path = PROJECT_ROOT / "data" / "ZebaNajeeb_CV.pdf",
) -> str:
    """
    Verifies that the candidate has active CV chunks in ChromaDB.
    If not (e.g., first run or expired after 7 days), automatically embeds `default_cv_path`.
    """
    candidate_id = normalize_candidate_id(candidate_name)
    collection = get_cv_collection()
    existing = collection.get(where={"candidate_id": candidate_id}, include=[])
    if not existing.get("ids"):
        if not default_cv_path.exists():
            raise FileNotFoundError(
                f"No CV vectors found in ChromaDB for '{candidate_id}' and default CV '{default_cv_path}' does not exist."
            )
        store_cv_in_vector_db(default_cv_path, candidate_name=candidate_name)
    return candidate_id


# =====================================================================
# 4. RAG Semantic Matching & Scoring Engine (ChromaDB + Gemini)
# =====================================================================

def score_and_rank_active_jobs(
    jobs: list[ActiveJobPosting],
    candidate_name: str = "Zeba Najeeb",
    top_k_chunks: int = 4,
    db_path: Path = DB_PATH,
) -> list[RankedJobMatch]:
    """
    1. Ensures candidate's CV is indexed in ChromaDB.
    2. For each active job, retrieves the `top_k_chunks` most relevant CV chunks via Cosine Similarity.
    3. Evaluates all jobs against their retrieved CV evidence using Gemini 2.5 Flash.
    4. Unlocks Phase 4 Career Co-Pilot strictly for jobs with `match_score >= 70`.
    5. Saves results to SQLite (`job_evaluations`) and returns a sorted leaderboard (highest score first).
    """
    if not jobs:
        return []

    init_evaluations_table(db_path)
    candidate_id = ensure_cv_indexed(candidate_name)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("GEMINI_API_KEY is missing or invalid in .env")

    client = genai.Client(api_key=api_key)

    # Step 1: Retrieve top_k relevant CV chunks from ChromaDB for each job
    job_rag_contexts: list[str] = []
    vector_sim_by_fp: dict[str, float] = {}

    for job in jobs:
        query_text = (
            f"Job Title: {job.job_title} at {job.company_name}. "
            f"Summary: {job.description_summary} "
            f"Required Skills: {', '.join(job.key_requirements)}"
        )
        matched_chunks = retrieve_relevant_cv_chunks(
            job_description_query=query_text,
            candidate_name=candidate_name,
            top_k=top_k_chunks,
        )

        if matched_chunks:
            avg_sim = round(
                sum(c["similarity_percent"] for c in matched_chunks) / len(matched_chunks),
                2,
            )
        else:
            avg_sim = 0.0

        vector_sim_by_fp[job.fingerprint] = avg_sim

        chunks_formatted = "\n".join(
            f"    [CV Chunk #{c['chunk_index']} | Vector Sim: {c['similarity_percent']}%]: {c['text']}"
            for c in matched_chunks
        )

        job_rag_contexts.append(
            f"=== JOB TO EVALUATE ===\n"
            f"Fingerprint: {job.fingerprint}\n"
            f"Title: {job.job_title} @ {job.company_name} ({job.location})\n"
            f"Description: {job.description_summary}\n"
            f"Required Skills: {', '.join(job.key_requirements)}\n"
            f"Retrieved Candidate CV Chunks (from ChromaDB):\n{chunks_formatted}\n"
        )

    # Step 2: Grounded Gemini Rubric Evaluation across all jobs in the portal batch
    joined_contexts = "\n\n".join(job_rag_contexts)
    prompt = f"""
    You are a Principal AI Engineering Recruiter and RAG Scoring Engine.

    Below is a batch of active job postings along with the candidate's relevant CV chunks
    retrieved from ChromaDB.

    Strict Scoring Rubric (0 to 100):
    - 80 to 95: Strong direct match (candidate's CV chunks clearly demonstrate core required skills such as Python, LLMs/GenAI, RAG, LangChain/LangGraph, FastAPI, Automation, APIs).
    - 70 to 79: Good match (candidate matches the core AI/Backend foundation, with 2-3 learnable domain or cloud/DevOps skill gaps).
    - 50 to 69: Partial/Stretch match (e.g., executive CTO/Lead roles requiring heavy Kubernetes/Kafka/TypeScript or specialized domain experience not in CV).
    - Below 50: Low match (most core requirements are missing from the retrieved CV chunks).

    Important Rules:
    1. Ground your evaluation STRICTLY in the retrieved CV chunks provided for each job. Do NOT hallucinate skills the candidate does not have.
    2. Return the exact `fingerprint` string for each job.
    3. Clearly list `matching_skills` (present in CV) and `missing_skills` (required by job but absent in CV).

    {joined_contexts}
    """

    response = None
    last_error = None
    for model_name in MODEL_FALLBACK_CHAIN:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchJobEvaluationList,
                    temperature=0.1,
                ),
            )
            break
        except Exception as err:
            print(f"   [Fallback Notice] Model '{model_name}' busy ({type(err).__name__}), switching to backup model...")
            last_error = err

    if response is None:
        raise RuntimeError(f"All models in fallback chain failed. Last error: {last_error}")

    eval_batch = BatchJobEvaluationList.model_validate_json(response.text)
    eval_by_fp = {e.fingerprint: e for e in eval_batch.evaluations}

    # Step 3: Combine & Persist into SQLite `job_evaluations`
    ranked_results: list[RankedJobMatch] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db_connection(db_path) as conn:
        for job in jobs:
            ev = eval_by_fp.get(job.fingerprint)
            if not ev:
                continue

            copilot_ok = ev.match_score >= HIGH_MATCH_THRESHOLD
            vec_sim = vector_sim_by_fp.get(job.fingerprint, 0.0)

            ranked_item = RankedJobMatch(
                fingerprint=job.fingerprint,
                portal_name=job.portal_name,
                job_title=job.job_title,
                company_name=job.company_name,
                location=job.location,
                posted_date_raw=job.posted_date_raw,
                posted_date_iso=job.posted_date_iso,
                days_ago=job.days_ago,
                job_url=job.job_url,
                description_summary=job.description_summary,
                match_score=ev.match_score,
                vector_similarity_avg=vec_sim,
                matching_skills=ev.matching_skills,
                missing_skills=ev.missing_skills,
                match_reasoning=ev.match_reasoning,
                copilot_unlocked=copilot_ok,
            )
            ranked_results.append(ranked_item)

            conn.execute(
                """
                INSERT INTO job_evaluations (
                    candidate_id, fingerprint, portal_name, job_title, company_name,
                    location, posted_date_raw, posted_date_iso, days_ago, job_url,
                    description_summary, match_score, vector_similarity_avg,
                    matching_skills_json, missing_skills_json, match_reasoning,
                    copilot_unlocked, evaluated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id, fingerprint) DO UPDATE SET
                    match_score = excluded.match_score,
                    vector_similarity_avg = excluded.vector_similarity_avg,
                    matching_skills_json = excluded.matching_skills_json,
                    missing_skills_json = excluded.missing_skills_json,
                    match_reasoning = excluded.match_reasoning,
                    copilot_unlocked = excluded.copilot_unlocked,
                    evaluated_at = excluded.evaluated_at
                """,
                (
                    candidate_id,
                    ranked_item.fingerprint,
                    ranked_item.portal_name,
                    ranked_item.job_title,
                    ranked_item.company_name,
                    ranked_item.location,
                    ranked_item.posted_date_raw,
                    ranked_item.posted_date_iso,
                    ranked_item.days_ago,
                    ranked_item.job_url,
                    ranked_item.description_summary,
                    ranked_item.match_score,
                    ranked_item.vector_similarity_avg,
                    json.dumps(ranked_item.matching_skills),
                    json.dumps(ranked_item.missing_skills),
                    ranked_item.match_reasoning,
                    1 if ranked_item.copilot_unlocked else 0,
                    now_iso,
                ),
            )
        conn.commit()

    # Sort highest match_score first
    ranked_results.sort(key=lambda x: x.match_score, reverse=True)
    return ranked_results


def get_ranked_jobs_from_db(
    candidate_id: str,
    min_score: int = 0,
    db_path: Path = DB_PATH,
) -> list[RankedJobMatch]:
    """Retrieves saved ranked job evaluations for `candidate_id` sorted by `match_score` DESC."""
    init_evaluations_table(db_path)
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM job_evaluations
            WHERE candidate_id = ? AND match_score >= ?
            ORDER BY match_score DESC, days_ago ASC
            """,
            (candidate_id, min_score),
        ).fetchall()

    results: list[RankedJobMatch] = []
    for r in rows:
        results.append(
            RankedJobMatch(
                fingerprint=r["fingerprint"],
                portal_name=r["portal_name"],
                job_title=r["job_title"],
                company_name=r["company_name"],
                location=r["location"],
                posted_date_raw=r["posted_date_raw"],
                posted_date_iso=r["posted_date_iso"],
                days_ago=r["days_ago"],
                job_url=r["job_url"],
                description_summary=r["description_summary"],
                match_score=r["match_score"],
                vector_similarity_avg=r["vector_similarity_avg"],
                matching_skills=json.loads(r["matching_skills_json"]),
                missing_skills=json.loads(r["missing_skills_json"]),
                match_reasoning=r["match_reasoning"],
                copilot_unlocked=bool(r["copilot_unlocked"]),
            )
        )
    return results


# =====================================================================
# 5. End-to-End Verification Test (Module 5)
# =====================================================================

if __name__ == "__main__":
    candidate_name = "Zeba Najeeb"
    candidate_id = normalize_candidate_id(candidate_name)

    print("=== Module 5: RAG Matcher & Scoring Engine (ChromaDB + Gemini) ===")
    print(f"Candidate : {candidate_name} (ID: {candidate_id})")

    # 1. Load active jobs discovered in Module 4 from SQLite
    active_jobs = get_saved_active_jobs(candidate_id=candidate_id)
    if not active_jobs:
        print("\nNo active jobs found in SQLite! Please run `python -m src.scrapers.job_fetcher` first.")
    else:
        print(f"1. Found {len(active_jobs)} verified active jobs in SQLite (`discovered_jobs`).")
        print("2. Retrieving top CV chunks from ChromaDB & scoring fit with Gemini 2.5 Flash...\n")

        ranked_jobs = score_and_rank_active_jobs(
            jobs=active_jobs,
            candidate_name=candidate_name,
            top_k_chunks=4,
        )

        print("=" * 105)
        print("RANKED JOB MATCH LEADERBOARD (High-Match Co-Pilot Unlocks at Score >= 70%)")
        print("=" * 105)

        for rank, item in enumerate(ranked_jobs, start=1):
            badge = (
                "UNLOCKED (Score >= 70%)"
                if item.copilot_unlocked
                else "LOCKED   (Score <  70%)"
            )
            print(
                f"#{rank} | Score: {item.match_score:>3}% (Vector Sim: {item.vector_similarity_avg}%) | "
                f"Co-Pilot: {badge}"
            )
            print(f"   Role & Company  : {item.job_title} @ {item.company_name} ({item.location})")
            print(f"   Portal & Age    : {item.portal_name} | Posted {item.posted_date_raw} ({item.posted_date_iso})")
            print(f"   Matching Skills : {', '.join(item.matching_skills)}")
            print(f"   Skill Gaps      : {', '.join(item.missing_skills) if item.missing_skills else 'None'}")
            print(f"   RAG Assessment  : {item.match_reasoning}")
            print("-" * 105)
