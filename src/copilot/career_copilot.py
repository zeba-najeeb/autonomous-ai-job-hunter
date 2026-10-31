import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.matcher.rag_matcher import (
    HIGH_MATCH_THRESHOLD,
    MODEL_FALLBACK_CHAIN,
    RankedJobMatch,
    get_ranked_jobs_from_db,
)
from src.portals.queue_manager import DB_PATH, get_db_connection
from src.profile.parser import PROJECT_ROOT
from src.profile.vector_store import (
    normalize_candidate_id,
    retrieve_relevant_cv_chunks,
)

# Load environment variables from C:\krish_naik\coursera\job_hunt\.env
load_dotenv(PROJECT_ROOT / ".env", override=True)


# =====================================================================
# 1. Pydantic Schemas for Phase 4 On-Demand Career Co-Pilot
# =====================================================================

class UpskillRecommendation(BaseModel):
    """Concrete course and mini-project recommendation to close a specific skill gap."""
    missing_skill: str = Field(description="The missing skill or tool from the job requirements.")
    recommended_course: str = Field(
        description="Specific high-signal course, documentation track, or certification to learn this skill quickly."
    )
    hands_on_mini_project: str = Field(
        description="A weekend portfolio mini-project the candidate can build to prove this skill in an interview."
    )


class InterviewQAItem(BaseModel):
    """Role-specific technical/architectural interview question & CV-grounded answer."""
    question: str = Field(
        description="A realistic technical, architectural, or behavioral interview question tailored to this job."
    )
    why_they_ask: str = Field(
        description="1-sentence explanation of what the interviewer is testing with this question."
    )
    cv_grounded_answer: str = Field(
        description=(
            "A strong, senior-level sample answer using the candidate's ACTUAL projects, metrics, "
            "and experience from their retrieved CV chunks."
        )
    )


class CareerCoPilotKit(BaseModel):
    """Complete On-Demand Career Co-Pilot Kit generated strictly for High-Match Jobs (>= 70%)."""
    tailored_cv_bullets: list[str] = Field(
        description=(
            "4 to 5 ATS-optimized resume bullet points rewritten from the candidate's REAL CV experience "
            "to align with this job's exact keywords (zero fabricated experience)."
        )
    )
    tailored_cover_letter: str = Field(
        description="A concise, high-conversion cover letter / hiring manager outreach message tailored to the company and role."
    )
    skill_gap_roadmap: list[UpskillRecommendation] = Field(
        description="Actionable courses and mini-projects to close the specific missing_skills for this role."
    )
    interview_prep_qa: list[InterviewQAItem] = Field(
        description="4 role-specific technical and architectural interview questions with answers grounded in the candidate's CV."
    )


# =====================================================================
# 2. SQLite Table Initialization for Cached Co-Pilot Kits
# =====================================================================

def init_copilot_table(db_path: Path = DB_PATH) -> None:
    """
    Creates the `copilot_kits` table in SQLite so generated Career Co-Pilot kits
    are saved and never need to be re-generated twice for the same job.
    """
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS copilot_kits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                job_title TEXT NOT NULL,
                company_name TEXT NOT NULL,
                match_score INTEGER NOT NULL,
                kit_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(candidate_id, fingerprint)
            )
            """
        )
        conn.commit()


# =====================================================================
# 3. Strict Eligibility Guard (Score >= 70% Only)
# =====================================================================

def verify_copilot_eligibility(job: RankedJobMatch) -> None:
    """
    Enforces the Phase 4 Efficiency & Focus Rule:
    On-Demand Career Co-Pilot features are strictly unlocked ONLY when `match_score >= 70%`.
    """
    if not job.copilot_unlocked or job.match_score < HIGH_MATCH_THRESHOLD:
        raise PermissionError(
            f"[Co-Pilot Locked] '{job.job_title}' at '{job.company_name}' has a Match Score of {job.match_score}% "
            f"(below the {HIGH_MATCH_THRESHOLD}% threshold). On-Demand Career Co-Pilot is strictly unlocked "
            f"for high-matching roles (>={HIGH_MATCH_THRESHOLD}%) to focus your job hunt and save LLM tokens."
        )


# =====================================================================
# 4. On-Demand Career Co-Pilot Generator (with Model Fallback & Cache)
# =====================================================================

def generate_copilot_kit_for_job(
    job: RankedJobMatch,
    candidate_name: str = "Zeba Najeeb",
    force_regenerate: bool = False,
    db_path: Path = DB_PATH,
) -> CareerCoPilotKit:
    """
    1. Verifies the job is eligible (`match_score >= 70%`).
    2. Checks SQLite (`copilot_kits`) if a kit was already generated for this job.
    3. Retrieves the candidate's top CV chunks from ChromaDB.
    4. Uses Gemini (with MODEL_FALLBACK_CHAIN) to generate:
       - 4-5 ATS-Optimized CV Bullets (strictly grounded in real CV facts)
       - Tailored Cover Letter / Outreach Note
       - Skill-Gap Courses & Mini-Projects for `job.missing_skills`
       - 4 Custom Interview Questions & CV-Grounded Answers
    5. Saves and returns the `CareerCoPilotKit`.
    """
    # 1. Enforce >= 70% Match Score Gate
    verify_copilot_eligibility(job)

    init_copilot_table(db_path)
    candidate_id = normalize_candidate_id(candidate_name)

    # 2. Return cached kit from SQLite if already generated
    if not force_regenerate:
        with get_db_connection(db_path) as conn:
            row = conn.execute(
                "SELECT kit_json FROM copilot_kits WHERE candidate_id = ? AND fingerprint = ?",
                (candidate_id, job.fingerprint),
            ).fetchone()
            if row:
                return CareerCoPilotKit.model_validate_json(row["kit_json"])

    # 3. Retrieve relevant CV chunks from ChromaDB to ground the generation
    query_text = f"{job.job_title} at {job.company_name}. {job.description_summary} {' '.join(job.matching_skills)}"
    cv_chunks = retrieve_relevant_cv_chunks(
        job_description_query=query_text,
        candidate_name=candidate_name,
        top_k=5,
    )
    cv_evidence = "\n\n".join(
        f"[CV Chunk #{c['chunk_index']}]:\n{c['text']}" for c in cv_chunks
    )

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("GEMINI_API_KEY is missing or invalid in .env")

    client = genai.Client(api_key=api_key)

    missing_str = ", ".join(job.missing_skills) if job.missing_skills else "Advanced Cloud Scaling / Domain Specialization"

    prompt = f"""
    You are an Executive AI Career Co-Pilot and Principal Technical Hiring Manager.

    The candidate ({candidate_name}) has selected a HIGH-MATCH job ({job.match_score}% Match)
    and explicitly requested On-Demand Career Co-Pilot assistance:

    === TARGET JOB ===
    Role            : {job.job_title}
    Company         : {job.company_name} ({job.location})
    Portal          : {job.portal_name}
    Summary         : {job.description_summary}
    Matching Skills : {', '.join(job.matching_skills)}
    Missing Skills  : {missing_str}

    === CANDIDATE'S REAL CV EVIDENCE (FROM CHROMADB) ===
    {cv_evidence}

    === STRICT GENERATION RULES ===
    1. `tailored_cv_bullets`: Rewrite 4 to 5 high-impact CV bullet points using the candidate's ACTUAL roles, projects, and metrics from the CV chunks above, naturally incorporating ATS keywords for "{job.job_title}" at "{job.company_name}". NEVER fabricate companies or fake experience.
    2. `tailored_cover_letter`: Write a compelling, concise cover letter / hiring manager outreach note (150-200 words) pitching {candidate_name} for "{job.job_title}" at "{job.company_name}", referencing real achievements from the CV chunks.
    3. `skill_gap_roadmap`: For each skill in Missing Skills ({missing_str}), recommend 1 specific course/learning resource and 1 concrete weekend mini-project that bridges the gap.
    4. `interview_prep_qa`: Create 4 realistic technical/system-design/behavioral interview questions for "{job.job_title}" at "{job.company_name}", and write sample answers grounded directly in {candidate_name}'s real CV projects.
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
                    response_schema=CareerCoPilotKit,
                    temperature=0.25,
                ),
            )
            break
        except Exception as err:
            print(f"   [Fallback Notice] Model '{model_name}' busy ({type(err).__name__}), switching to backup model...")
            last_error = err

    if response is None:
        raise RuntimeError(f"All models in fallback chain failed. Last error: {last_error}")

    kit = CareerCoPilotKit.model_validate_json(response.text)

    # 5. Cache in SQLite `copilot_kits`
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO copilot_kits (
                candidate_id, fingerprint, job_title, company_name,
                match_score, kit_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id, fingerprint) DO UPDATE SET
                kit_json = excluded.kit_json,
                created_at = excluded.created_at
            """,
            (
                candidate_id,
                job.fingerprint,
                job.job_title,
                job.company_name,
                job.match_score,
                kit.model_dump_json(),
                now_iso,
            ),
        )
        conn.commit()

    return kit


def print_copilot_kit(job: RankedJobMatch, kit: CareerCoPilotKit) -> None:
    """Neatly formats and prints the generated Career Co-Pilot Kit in the console."""
    print("\n" + "=" * 105)
    print(f"CAREER CO-PILOT KIT | {job.job_title} @ {job.company_name} (Match Score: {job.match_score}%)")
    print("=" * 105)

    print("\n1. ATS-OPTIMIZED CV BULLET POINTS (Grounded in Your Real CV):")
    for idx, bullet in enumerate(kit.tailored_cv_bullets, start=1):
        print(f"   [{idx}] {bullet}")

    print("\n2. TAILORED COVER LETTER / HIRING MANAGER OUTREACH:")
    print("-" * 95)
    for line in kit.tailored_cover_letter.splitlines():
        print(f"   {line}")
    print("-" * 95)

    print("\n3. SKILL-GAP COURSES & PORTFOLIO MINI-PROJECTS:")
    for idx, item in enumerate(kit.skill_gap_roadmap, start=1):
        print(f"   [{idx}] Missing Skill : {item.missing_skill}")
        print(f"       Course        : {item.recommended_course}")
        print(f"       Mini-Project  : {item.hands_on_mini_project}")

    print("\n4. CUSTOM INTERVIEW Q&A (Grounded in Your Actual CV Projects):")
    for idx, qa in enumerate(kit.interview_prep_qa, start=1):
        print(f"   Q{idx}: {qa.question}")
        print(f"       Why They Ask       : {qa.why_they_ask}")
        print(f"       CV-Grounded Answer : {qa.cv_grounded_answer}\n")
    print("=" * 105)


# =====================================================================
# 5. End-to-End Verification Test (Module 6A)
# =====================================================================

if __name__ == "__main__":
    candidate_name = "Zeba Najeeb"
    candidate_id = normalize_candidate_id(candidate_name)

    print("=== Module 6A: On-Demand Career Co-Pilot (Strictly for Score >= 70%) ===")
    ranked_jobs = get_ranked_jobs_from_db(candidate_id=candidate_id)

    if not ranked_jobs:
        print("No ranked jobs found in SQLite! Run `python -m src.matcher.rag_matcher` first.")
    else:
        # 1. Test the < 70% Guardrail on any low-match job first
        low_match_jobs = [j for j in ranked_jobs if not j.copilot_unlocked]
        if low_match_jobs:
            low_job = low_match_jobs[0]
            print(f"\n1. Testing Guardrail on Low-Match Job ({low_job.match_score}% - {low_job.job_title} @ {low_job.company_name}):")
            try:
                generate_copilot_kit_for_job(low_job, candidate_name=candidate_name)
            except PermissionError as err:
                print(f"   -> Guardrail Verified! Blocked with message:\n      {err}")

        # 2. Generate the full On-Demand Career Co-Pilot Kit for the #1 Highest-Matching Job (>= 70%)
        high_match_jobs = [j for j in ranked_jobs if j.copilot_unlocked]
        if high_match_jobs:
            top_job = high_match_jobs[0]
            print(
                f"\n2. Generating On-Demand Career Co-Pilot Kit for #1 High-Match Job:\n"
                f"   -> {top_job.job_title} @ {top_job.company_name} (Score: {top_job.match_score}%)..."
            )
            copilot_kit = generate_copilot_kit_for_job(top_job, candidate_name=candidate_name)
            print_copilot_kit(top_job, copilot_kit)
