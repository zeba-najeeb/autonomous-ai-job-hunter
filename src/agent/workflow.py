from pathlib import Path
from typing import TypedDict

from src.copilot.career_copilot import (
    generate_copilot_kit_for_job,
    print_copilot_kit,
)
from src.matcher.rag_matcher import (
    RankedJobMatch,
    get_ranked_jobs_from_db,
    score_and_rank_active_jobs,
)
from src.portals.queue_manager import (
    QueuedPortal,
    get_candidate_queue,
    mark_portal_completed_and_advance,
    print_queue_table,
    recommend_portals_for_location,
    save_portal_queue,
    select_next_portal,
)
from src.profile.parser import PROJECT_ROOT, extract_text_from_pdf
from src.profile.role_expander import CandidateSearchProfile, expand_candidate_role
from src.profile.vector_store import (
    cleanup_inactive_sessions,
    delete_candidate_cv,
    normalize_candidate_id,
    store_cv_in_vector_db,
)
from src.scrapers.job_fetcher import fetch_and_filter_jobs_for_portal


# =====================================================================
# 1. Agent State Definition (LangGraph / State Machine Compatible)
# =====================================================================

class JobHunterAgentState(TypedDict, total=False):
    """Typed state dictionary passed across the 5-Phase Agentic Workflow nodes."""
    cv_path: str
    target_job_title: str
    target_location: str
    candidate_id: str
    candidate_name: str
    expanded_job_titles: list[str]
    top_cv_skills: list[str]
    active_portal: QueuedPortal | None
    ranked_jobs: list[RankedJobMatch]


# =====================================================================
# 2. Agentic Workflow Nodes (Phases 1 -> 5)
# =====================================================================

def node_1_onboard_candidate(state: JobHunterAgentState) -> JobHunterAgentState:
    """
    Phase 1 Node:
    - Runs 7-day sliding TTL auto-cleanup on ChromaDB.
    - Extracts text from the candidate's PDF CV (`parser.py`).
    - Expands the target job title into 3-4 closely related titles (`role_expander.py`).
    - Embeds CV chunks into ChromaDB with multi-tenant `candidate_id` isolation (`vector_store.py`).
    """
    cleanup_inactive_sessions(max_inactive_days=7)

    cv_path = Path(state["cv_path"])
    target_job_title = state["target_job_title"]
    target_location = state["target_location"]

    print(f"\n[Phase 1] Parsing CV ('{cv_path.name}') & expanding role '{target_job_title}' in '{target_location}'...")
    cv_text = extract_text_from_pdf(cv_path)

    profile: CandidateSearchProfile = expand_candidate_role(
        cv_text=cv_text,
        target_job_title=target_job_title,
        target_location=target_location,
    )

    candidate_id = normalize_candidate_id(profile.candidate_name)
    vec_info = store_cv_in_vector_db(cv_path, candidate_name=profile.candidate_name)

    print(f"   -> Candidate Identified : {profile.candidate_name} (ID: {candidate_id})")
    print(f"   -> ChromaDB Indexed     : {vec_info['chunks_stored']} chunks ({vec_info['embedding_dimension']} dims, 7-day sliding TTL)")
    print(f"   -> Expanded Job Titles  : {', '.join(profile.expanded_job_titles)}")
    print(f"   -> Extracted CV Skills  : {', '.join(profile.top_cv_skills)}")

    state["candidate_id"] = candidate_id
    state["candidate_name"] = profile.candidate_name
    state["expanded_job_titles"] = profile.expanded_job_titles
    state["top_cv_skills"] = profile.top_cv_skills
    return state


def node_2_prepare_portal_queue(
    state: JobHunterAgentState,
    force_new_queue: bool = False,
) -> JobHunterAgentState:
    """
    Phase 2 Node:
    - Checks if the candidate already has a saved portal queue in SQLite (`data/job_hunter.db`).
    - If not (or if `force_new_queue=True`), asks Gemini to recommend regional portals and saves them as `PENDING`.
    - Selects the current `IN_PROGRESS` portal (or promotes the first `PENDING` portal to `IN_PROGRESS`).
    """
    candidate_id = state["candidate_id"]
    existing_queue = get_candidate_queue(candidate_id)

    if not existing_queue or force_new_queue:
        print(f"\n[Phase 2] Discovering regional job portals for '{state['target_location']}'...")
        rec_list = recommend_portals_for_location(
            target_job_title=state["target_job_title"],
            target_location=state["target_location"],
        )
        existing_queue = save_portal_queue(
            candidate_id=candidate_id,
            target_job_title=state["target_job_title"],
            target_location=state["target_location"],
            portals=rec_list.portals,
        )
    else:
        print(f"\n[Phase 2] Resuming saved SQLite portal queue for '{state['candidate_name']}'...")

    active_portal = select_next_portal(candidate_id)
    print_queue_table(get_candidate_queue(candidate_id))

    state["active_portal"] = active_portal
    return state


def node_3_scan_and_match_portal(state: JobHunterAgentState) -> JobHunterAgentState:
    """
    Phase 3 Node:
    - Fetches active jobs from `state['active_portal']` using the Smart Discovery Adapter.
    - Filters out jobs > 30 days old (Gate #1) and cross-portal duplicates (Gate #2).
    - Runs the RAG Semantic Matcher against ChromaDB CV chunks to compute 0-100% Match Scores.
    """
    portal = state.get("active_portal")
    if not portal:
        print("\n[Phase 3] All portals in the queue are COMPLETED!")
        state["ranked_jobs"] = []
        return state

    print(
        f"\n[Phase 3] Scanning Active Portal: Link #{portal.link_order} — "
        f"{portal.portal_name} ({portal.fetch_strategy})..."
    )
    active_jobs, stats = fetch_and_filter_jobs_for_portal(
        candidate_id=state["candidate_id"],
        portal=portal,
        primary_job_title=state["target_job_title"],
        expanded_job_titles=state["expanded_job_titles"],
        target_location=state["target_location"],
        max_results=5,
    )
    print(f"   -> Gate #1 (<=30d) & Gate #2 (Deduplication) Stats: {stats}")

    if not active_jobs:
        # If jobs for this portal were already fetched in this session, load ranked jobs from DB
        ranked = get_ranked_jobs_from_db(state["candidate_id"])
        portal_ranked = [j for j in ranked if j.portal_name == portal.portal_name]
        state["ranked_jobs"] = portal_ranked
        return state

    print("   -> Scoring & ranking active jobs against ChromaDB CV chunks...")
    ranked_jobs = score_and_rank_active_jobs(
        jobs=active_jobs,
        candidate_name=state["candidate_name"],
        top_k_chunks=4,
    )
    state["ranked_jobs"] = ranked_jobs
    return state


def display_ranked_jobs_table(portal: QueuedPortal, ranked_jobs: list[RankedJobMatch]) -> None:
    """Displays the ranked job list for the current portal with Match Scores and Co-Pilot status."""
    print("\n" + "=" * 105)
    print(f"RANKED ACTIVE JOBS ON LINK #{portal.link_order}: {portal.portal_name.upper()} (<= 30 Days)")
    print("=" * 105)
    for idx, job in enumerate(ranked_jobs, start=1):
        status_tag = (
            "CO-PILOT UNLOCKED (>=70%)"
            if job.copilot_unlocked
            else "LOCKED (<70% Match)      "
        )
        print(
            f"[{idx}] {job.match_score:>3}% Match | {status_tag} | "
            f"{job.job_title} @ {job.company_name} ({job.posted_date_raw})"
        )
        print(f"    Matching Skills : {', '.join(job.matching_skills[:6])}")
        print(f"    Missing Skills  : {', '.join(job.missing_skills[:5]) if job.missing_skills else 'None'}")
        print("-" * 105)


# =====================================================================
# 3. Interactive Human-in-the-Loop Orchestrator (Phases 1 -> 5)
# =====================================================================

def run_interactive_career_copilot() -> None:
    """
    End-to-End Interactive Agent CLI:
    Connects Phases 1, 2, 3, 4, and 5 into a seamless Human-in-the-Loop experience.
    """
    default_cv = PROJECT_ROOT / "data" / "ZebaNajeeb_CV.pdf"

    print("=" * 95)
    print("   AUTONOMOUS AI JOB HUNTER & RAG CAREER CO-PILOT AGENT (END-TO-END ORCHESTRATOR)")
    print("=" * 95)

    cv_input = input(f"1. Enter path to PDF CV [Press Enter for '{default_cv.name}']: ").strip()
    cv_path = Path(cv_input) if cv_input else default_cv

    role_input = input("2. Enter Target Job Title [Press Enter for 'AI Engineer']: ").strip() or "AI Engineer"
    loc_input = input("3. Enter Target Location  [Press Enter for 'Dubai, UAE']: ").strip() or "Dubai, UAE"

    state: JobHunterAgentState = {
        "cv_path": str(cv_path),
        "target_job_title": role_input,
        "target_location": loc_input,
    }

    # Phase 1: Candidate Onboarding & ChromaDB Indexing
    state = node_1_onboard_candidate(state)

    # Phase 2: Location Portal Queue Setup / Resume
    state = node_2_prepare_portal_queue(state)

    # Phase 3, 4 & 5 Loop: Scan Active Link -> On-Demand Co-Pilot -> Advance to Next Link
    while state.get("active_portal") is not None:
        active_portal = state["active_portal"]
        assert active_portal is not None

        state = node_3_scan_and_match_portal(state)
        ranked_jobs = state.get("ranked_jobs", [])

        if ranked_jobs:
            display_ranked_jobs_table(active_portal, ranked_jobs)
        else:
            print(f"\nNo new unique jobs found on {active_portal.portal_name}.")

        # Phase 4 & 5 Human-in-the-Loop Menu
        while True:
            print(
                f"\n[Human-in-the-Loop Menu — Current Portal: Link #{active_portal.link_order} ({active_portal.portal_name})]"
            )
            print("  • Enter Job Number (1 - 5) -> Launch On-Demand Career Co-Pilot (Requires Score >= 70%)")
            print("  • Type 'next'              -> Mark current portal COMPLETED & advance to next portal link")
            print("  • Type 'queue'             -> View full SQLite portal queue status")
            print("  • Type 'clear'             -> Purge my CV vectors from ChromaDB now & exit")
            print("  • Type 'quit'              -> Save progress & exit (7-day sliding session stays active)")

            choice = input("\nYour action: ").strip().lower()

            if choice in {"q", "quit", "exit"}:
                print(
                    "\nProgress saved in SQLite! Your ChromaDB vectors remain active for 7 days of inactivity."
                )
                return

            if choice == "clear":
                removed_id = delete_candidate_cv(state["candidate_name"])
                print(f"\nExplicitly purged all CV vectors for '{removed_id}' from ChromaDB. Goodbye!")
                return

            if choice == "queue":
                print_queue_table(get_candidate_queue(state["candidate_id"]))
                continue

            if choice == "next":
                print(
                    f"\n[Phase 5] Marking Link #{active_portal.link_order} ({active_portal.portal_name}) COMPLETED..."
                )
                next_portal = mark_portal_completed_and_advance(
                    candidate_id=state["candidate_id"],
                    portal_id=active_portal.id,
                )
                print_queue_table(get_candidate_queue(state["candidate_id"]))
                state["active_portal"] = next_portal
                break

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(ranked_jobs):
                    selected_job = ranked_jobs[idx]
                    try:
                        print(
                            f"\n[Phase 4] Generating On-Demand Career Co-Pilot Kit for "
                            f"'{selected_job.job_title}' @ '{selected_job.company_name}' ({selected_job.match_score}%)..."
                        )
                        kit = generate_copilot_kit_for_job(
                            job=selected_job,
                            candidate_name=state["candidate_name"],
                        )
                        print_copilot_kit(selected_job, kit)
                    except PermissionError as err:
                        print(f"\n{err}")
                else:
                    print(f"Please enter a valid job number between 1 and {len(ranked_jobs)}.")
            else:
                print("Unrecognized command. Enter a job number (1-5), 'next', 'queue', 'clear', or 'quit'.")

    print("\nCongratulations! You have completed all job portals in your regional queue.")


if __name__ == "__main__":
    run_interactive_career_copilot()
