import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.profile.parser import PROJECT_ROOT

# Load environment variables from C:\krish_naik\coursera\job_hunt\.env
load_dotenv(PROJECT_ROOT / ".env", override=True)

# SQLite Database Path
DB_PATH: Path = PROJECT_ROOT / "data" / "job_hunter.db"

# Portal Queue Statuses & Fetch Strategies
QueueStatus = Literal["PENDING", "IN_PROGRESS", "COMPLETED"]
FetchStrategy = Literal["OPEN_API", "ATS_BOARD", "SEARCH_INDEX"]


# =====================================================================
# 1. Pydantic Schemas for Gemini Portal Discovery & Queue Tracking
# =====================================================================

class PortalItem(BaseModel):
    """Schema for a single job portal or career link recommended for a location."""
    portal_name: str = Field(
        description="Name of the job portal or career platform (e.g., 'LinkedIn UAE', 'Bayt', 'GulfTalent', 'RemoteOK')."
    )
    portal_url: str = Field(
        description="Direct base URL or search domain of the job portal (e.g., 'https://www.bayt.com', 'https://www.gulftalent.com')."
    )
    fetch_strategy: FetchStrategy = Field(
        description=(
            "Best strategy for our Smart Discovery Adapter in Module 4: "
            "'OPEN_API' for open remote job feeds (RemoteOK, Jobicy, Himalayas), "
            "'ATS_BOARD' for public company ATS boards (Greenhouse, Lever, Workable), or "
            "'SEARCH_INDEX' for major regional portals with anti-bot walls (LinkedIn, Bayt, GulfTalent, NaukriGulf, Indeed)."
        )
    )
    reason: str = Field(
        description="Brief 1-sentence explanation of why this portal is valuable for the candidate's target role and location."
    )


class PortalRecommendationList(BaseModel):
    """Structured output returned by Gemini for a candidate's target location."""
    target_location: str = Field(description="The location entered by the candidate.")
    portals: list[PortalItem] = Field(
        description="Ordered list of 5 to 6 high-signal job portals best suited for the target location and role."
    )


class QueuedPortal(BaseModel):
    """Represents a saved portal row inside the SQLite stateful queue."""
    id: int
    candidate_id: str
    link_order: int
    portal_name: str
    portal_url: str
    fetch_strategy: FetchStrategy
    reason: str
    status: QueueStatus
    updated_at: str


# =====================================================================
# 2. SQLite Database Initialization & Connection Helper
# =====================================================================

def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Creates a SQLite connection with row_factory set to sqlite3.Row."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """
    Initializes the SQLite database table for storing each candidate's
    location-based job portal queue (PENDING -> IN_PROGRESS -> COMPLETED).
    """
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portal_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                target_job_title TEXT NOT NULL,
                target_location TEXT NOT NULL,
                link_order INTEGER NOT NULL,
                portal_name TEXT NOT NULL,
                portal_url TEXT NOT NULL,
                fetch_strategy TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_portal_queue_candidate
            ON portal_queue (candidate_id, link_order)
            """
        )
        conn.commit()


# =====================================================================
# 3. AI Regional Portal Recommender (Gemini 2.5 Flash)
# =====================================================================

def recommend_portals_for_location(
    target_job_title: str,
    target_location: str,
) -> PortalRecommendationList:
    """
    Uses Gemini to suggest an ordered list of 5-6 regional or remote job portals
    for the candidate's target location and role, classifying each portal's fetch strategy.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise ValueError(
            "GEMINI_API_KEY is missing or invalid in C:\\krish_naik\\coursera\\job_hunt\\.env"
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an expert Regional Tech Recruiter and Job Market Strategist.

    A candidate is searching for the following role and location:
    - Target Job Title: "{target_job_title}"
    - Target Location: "{target_location}"

    Recommend an ordered list of 5 to 6 of the best job portals, regional job boards, or ATS ecosystems
    where active jobs for "{target_job_title}" in "{target_location}" are hosted.

    Rules for `fetch_strategy`:
    - Use "SEARCH_INDEX" for major regional portals that have login/Cloudflare walls (e.g., LinkedIn, Bayt, GulfTalent, NaukriGulf, Dubizzle, Indeed, Glassdoor).
    - Use "ATS_BOARD" for direct company Applicant Tracking System ecosystems (e.g., Greenhouse, Lever, Workable).
    - Use "OPEN_API" for open remote tech job boards (e.g., RemoteOK, Jobicy, Himalayas).
      Always include at least 1 OPEN_API or ATS_BOARD source alongside major regional portals so our hybrid scraper has diverse coverage.
    """

    response = None
    last_error = None
    for model_name in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PortalRecommendationList,
                    temperature=0.2,
                ),
            )
            break
        except Exception as err:
            print(f"   [Fallback Notice] Model '{model_name}' busy ({type(err).__name__}), switching to backup model...")
            last_error = err

    if response is None:
        raise RuntimeError(f"All models in fallback chain failed. Last error: {last_error}")

    return PortalRecommendationList.model_validate_json(response.text)


# =====================================================================
# 4. Stateful Queue Operations (PENDING -> IN_PROGRESS -> COMPLETED)
# =====================================================================

def save_portal_queue(
    candidate_id: str,
    target_job_title: str,
    target_location: str,
    portals: list[PortalItem],
    db_path: Path = DB_PATH,
) -> list[QueuedPortal]:
    """
    Replaces any previous queue for `candidate_id` with a fresh ordered queue
    of portals all initialized to status = 'PENDING'.
    """
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db_connection(db_path) as conn:
        # Clear any old queue for this candidate session before inserting the new queue
        conn.execute("DELETE FROM portal_queue WHERE candidate_id = ?", (candidate_id,))

        for idx, portal in enumerate(portals, start=1):
            conn.execute(
                """
                INSERT INTO portal_queue (
                    candidate_id, target_job_title, target_location,
                    link_order, portal_name, portal_url,
                    fetch_strategy, reason, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
                """,
                (
                    candidate_id,
                    target_job_title,
                    target_location,
                    idx,
                    portal.portal_name,
                    portal.portal_url,
                    portal.fetch_strategy,
                    portal.reason,
                    now_iso,
                    now_iso,
                ),
            )
        conn.commit()

    return get_candidate_queue(candidate_id, db_path=db_path)


def get_candidate_queue(
    candidate_id: str,
    db_path: Path = DB_PATH,
) -> list[QueuedPortal]:
    """Returns the full ordered list of portals and their current status for `candidate_id`."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, candidate_id, link_order, portal_name, portal_url,
                   fetch_strategy, reason, status, updated_at
            FROM portal_queue
            WHERE candidate_id = ?
            ORDER BY link_order ASC
            """,
            (candidate_id,),
        ).fetchall()

    return [QueuedPortal(**dict(row)) for row in rows]


def select_next_portal(
    candidate_id: str,
    db_path: Path = DB_PATH,
) -> QueuedPortal | None:
    """
    1. If the candidate already has a portal marked 'IN_PROGRESS', returns it (resuming where they left off).
    2. Otherwise, picks the first 'PENDING' portal in link_order, updates its status to 'IN_PROGRESS', and returns it.
    3. If all portals are 'COMPLETED', returns None.
    """
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db_connection(db_path) as conn:
        # 1. Check if there is already an active IN_PROGRESS link
        in_progress_row = conn.execute(
            """
            SELECT id, candidate_id, link_order, portal_name, portal_url,
                   fetch_strategy, reason, status, updated_at
            FROM portal_queue
            WHERE candidate_id = ? AND status = 'IN_PROGRESS'
            ORDER BY link_order ASC
            LIMIT 1
            """,
            (candidate_id,),
        ).fetchone()

        if in_progress_row:
            return QueuedPortal(**dict(in_progress_row))

        # 2. Otherwise find the next PENDING link
        next_pending_row = conn.execute(
            """
            SELECT id
            FROM portal_queue
            WHERE candidate_id = ? AND status = 'PENDING'
            ORDER BY link_order ASC
            LIMIT 1
            """,
            (candidate_id,),
        ).fetchone()

        if not next_pending_row:
            return None

        portal_id = next_pending_row["id"]
        conn.execute(
            """
            UPDATE portal_queue
            SET status = 'IN_PROGRESS', updated_at = ?
            WHERE id = ? AND candidate_id = ?
            """,
            (now_iso, portal_id, candidate_id),
        )
        conn.commit()

        updated_row = conn.execute(
            """
            SELECT id, candidate_id, link_order, portal_name, portal_url,
                   fetch_strategy, reason, status, updated_at
            FROM portal_queue
            WHERE id = ?
            """,
            (portal_id,),
        ).fetchone()

        return QueuedPortal(**dict(updated_row))


def mark_portal_completed_and_advance(
    candidate_id: str,
    portal_id: int,
    db_path: Path = DB_PATH,
) -> QueuedPortal | None:
    """
    Marks the specified `portal_id` as 'COMPLETED' and automatically selects
    and promotes the next 'PENDING' portal in the queue to 'IN_PROGRESS' (Link #1 -> Link #2).
    """
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE portal_queue
            SET status = 'COMPLETED', updated_at = ?
            WHERE id = ? AND candidate_id = ?
            """,
            (now_iso, portal_id, candidate_id),
        )
        conn.commit()

    # Automatically advance to the next PENDING link
    return select_next_portal(candidate_id, db_path=db_path)


def print_queue_table(queue: list[QueuedPortal]) -> None:
    """Helper to neatly display the current SQLite portal queue in the console."""
    print("\n" + "=" * 95)
    print(f"{'#':<4} | {'Status':<12} | {'Strategy':<13} | {'Portal Name':<22} | {'URL'}")
    print("-" * 95)
    for item in queue:
        print(
            f"#{item.link_order:<3} | {item.status:<12} | {item.fetch_strategy:<13} | "
            f"{item.portal_name[:22]:<22} | {item.portal_url}"
        )
    print("=" * 95)


# =====================================================================
# 5. End-to-End Verification Test (Module 3)
# =====================================================================

if __name__ == "__main__":
    test_candidate_id = "zeba_najeeb"
    test_role = "AI Engineer"
    test_location = "Dubai, UAE"

    print("=== Module 3: Location Portal Recommender & SQLite Queue Manager ===")
    print(f"Candidate ID    : {test_candidate_id}")
    print(f"Target Role     : {test_role}")
    print(f"Target Location : {test_location}")

    try:
        print("\n1. Asking Gemini to recommend regional job portals for Dubai, UAE...")
        rec_list = recommend_portals_for_location(
            target_job_title=test_role,
            target_location=test_location,
        )

        print(f"2. Saving {len(rec_list.portals)} portals into SQLite ({DB_PATH.name}) as PENDING...")
        initial_queue = save_portal_queue(
            candidate_id=test_candidate_id,
            target_job_title=test_role,
            target_location=test_location,
            portals=rec_list.portals,
        )
        print_queue_table(initial_queue)

        print("\n3. Selecting Link #1 to begin scanning (PENDING -> IN_PROGRESS)...")
        active_link_1 = select_next_portal(test_candidate_id)
        if active_link_1:
            print(f"   -> Active Portal: Link #{active_link_1.link_order} - {active_link_1.portal_name} [{active_link_1.status}]")
        print_queue_table(get_candidate_queue(test_candidate_id))

        if active_link_1:
            print(f"\n4. Simulating finishing all jobs on Link #{active_link_1.link_order} ({active_link_1.portal_name})...")
            print("   Marking Link #1 COMPLETED and auto-advancing to Link #2...")
            active_link_2 = mark_portal_completed_and_advance(test_candidate_id, active_link_1.id)
            if active_link_2:
                print(f"   -> Next Active Portal: Link #{active_link_2.link_order} - {active_link_2.portal_name} [{active_link_2.status}]")
            print_queue_table(get_candidate_queue(test_candidate_id))

    except Exception as e:
        print(f"\nError during Module 3 verification: {e}")
