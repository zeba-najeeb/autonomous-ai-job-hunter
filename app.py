import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.copilot.career_copilot import (
    CareerCoPilotKit,
    generate_copilot_kit_for_job,
)
from src.matcher.rag_matcher import (
    HIGH_MATCH_THRESHOLD,
    RankedJobMatch,
    get_ranked_jobs_from_db,
    score_and_rank_active_jobs,
)
from src.portals.queue_manager import (
    QueuedPortal,
    get_candidate_queue,
    mark_portal_completed_and_advance,
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
from src.scrapers.job_fetcher import (
    fetch_and_filter_jobs_for_portal,
    get_saved_active_jobs,
)

# Load local .env automatically for Personal Mode
load_dotenv(PROJECT_ROOT / ".env", override=True)

# =====================================================================
# Page Configuration & Custom Styling
# =====================================================================
st.set_page_config(
    page_title="Autonomous AI Job Hunter & RAG Career Co-Pilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .job-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .skill-badge-match {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        margin: 2px 4px 2px 0;
        font-weight: 500;
    }
    .skill-badge-gap {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        margin: 2px 4px 2px 0;
        font-weight: 500;
    }
    .role-badge {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        margin: 3px 6px 3px 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# Session State Initialization
# =====================================================================
def init_session_state() -> None:
    defaults = {
        "candidate_name": "Zeba Najeeb",
        "candidate_id": "zeba_najeeb",
        "target_job_title": "AI Engineer",
        "target_location": "Dubai, UAE",
        "expanded_job_titles": [
            "AI Engineer",
            "Generative AI Engineer",
            "LLM Application Developer",
            "Applied AI Backend Engineer",
        ],
        "top_cv_skills": [],
        "chunks_stored": 0,
        "last_scan_stats": None,
        "copilot_kits_cache": {},  # fingerprint -> CareerCoPilotKit
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


def format_copilot_markdown(job: RankedJobMatch, kit: CareerCoPilotKit) -> str:
    """Formats a generated Career Co-Pilot Kit as a clean Markdown file for 1-click download."""
    lines = [
        f"# Career Co-Pilot Kit: {job.job_title} @ {job.company_name}",
        f"**Location:** {job.location} | **Portal:** {job.portal_name} | **RAG Match Score:** {job.match_score}%",
        f"**Job URL:** {job.job_url}",
        "",
        "---",
        "",
        "## 1. ATS-Optimized CV Bullet Points (Grounded in Your CV)",
    ]
    for b in kit.tailored_cv_bullets:
        lines.append(f"- {b}")

    lines.extend(
        [
            "",
            "## 2. Tailored Cover Letter / Hiring Manager Outreach",
            "",
            kit.tailored_cover_letter,
            "",
            "## 3. Skill-Gap Courses & Weekend Mini-Projects",
            "",
        ]
    )
    for idx, item in enumerate(kit.skill_gap_roadmap, start=1):
        lines.append(f"### {idx}. Missing Skill: {item.missing_skill}")
        lines.append(f"- **Recommended Course:** {item.recommended_course}")
        lines.append(f"- **Hands-On Mini-Project:** {item.hands_on_mini_project}")
        lines.append("")

    lines.append("## 4. Role-Specific Interview Prep Q&A (Grounded in Your CV)")
    lines.append("")
    for idx, qa in enumerate(kit.interview_prep_qa, start=1):
        lines.append(f"### Q{idx}: {qa.question}")
        lines.append(f"- **Why They Ask:** {qa.why_they_ask}")
        lines.append(f"- **CV-Grounded Sample Answer:** {qa.cv_grounded_answer}")
        lines.append("")

    return "\n".join(lines)


# =====================================================================
# Sidebar: Personal Mode Status, API Key Override & Privacy Controls
# =====================================================================
with st.sidebar:
    st.title("⚙️ Control Center")
    st.caption("Autonomous AI Job Hunter & RAG Co-Pilot")

    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    has_local_key = bool(env_key and env_key != "PASTE_YOUR_GEMINI_API_KEY_HERE")

    if has_local_key:
        st.success("✅ **Personal Mode Active**\nLocal `.env` API Key loaded (Unlimited Access)")
    else:
        st.warning("⚠️ No `.env` key detected. Paste a Gemini API Key below.")

    with st.expander("🔑 API Key Override (Optional)", expanded=not has_local_key):
        custom_api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Leave blank on your laptop to use your local .env key. Visitors on a hosted link can paste their own key here.",
        )
        if custom_api_key.strip():
            os.environ["GEMINI_API_KEY"] = custom_api_key.strip()
            st.info("Using custom session Gemini API key.")

    st.divider()

    # Ephemeral Vector Storage & Privacy Management
    st.subheader("🛡️ Ephemeral Vector DB")
    st.caption(
        f"**Active Profile:** `{st.session_state.candidate_name}`\n\n"
        f"**Tenant ID:** `{st.session_state.candidate_id}`\n\n"
        "**Retention Policy:** 7-Day Sliding Inactivity TTL"
    )

    if st.button("🗑️ Purge My CV Vectors Now", use_container_width=True):
        removed_id = delete_candidate_cv(st.session_state.candidate_name)
        st.session_state.chunks_stored = 0
        st.toast(f"Purged ChromaDB vectors for '{removed_id}'!", icon="🗑️")
        st.success(f"Deleted all CV embeddings for `{removed_id}` from ChromaDB.")

    st.divider()
    st.caption(
        "Built by **Zeba Najeeb** • Python, FastAPI, ChromaDB, Google Gemini 2.5, PyMuPDF & SQLite"
    )


# =====================================================================
# Main Header
# =====================================================================
st.title("🎯 Autonomous AI Job Hunter & RAG Career Co-Pilot")
st.markdown(
    "Upload any PDF CV, set your **Target Role** & **Location**, work through a stateful regional portal queue, "
    "rank active jobs (`<= 30 days`) with **0–100% RAG Semantic Scoring**, and unlock on-demand interview & application coaching for high-matching roles (`>= 70%`)."
)

# =====================================================================
# ZONE 1: Candidate Onboarding & CV Intelligence (Phase 1)
# =====================================================================
with st.container(border=True):
    st.subheader("1️⃣ Phase 1: Candidate Onboarding & CV Knowledge Base")

    default_cv_path = PROJECT_ROOT / "data" / "ZebaNajeeb_CV.pdf"
    col_cv, col_inputs = st.columns([1.2, 1.8])

    with col_cv:
        uploaded_pdf = st.file_uploader(
            "Upload Candidate CV (PDF)",
            type=["pdf"],
            help="Upload any PDF resume, or leave empty to use your default CV in data/ZebaNajeeb_CV.pdf.",
        )
        if uploaded_pdf is not None:
            temp_cv_dir = PROJECT_ROOT / "data" / "uploaded_cvs"
            temp_cv_dir.mkdir(parents=True, exist_ok=True)
            active_cv_path = temp_cv_dir / uploaded_pdf.name
            active_cv_path.write_bytes(uploaded_pdf.getvalue())
            st.caption(f"📄 Using uploaded CV: **`{uploaded_pdf.name}`**")
        elif default_cv_path.exists():
            active_cv_path = default_cv_path
            st.caption(f"📄 Default CV detected: **`{default_cv_path.name}`** (Ready)")
        else:
            active_cv_path = None
            st.warning("Please upload a PDF CV to get started.")

    with col_inputs:
        c1, c2 = st.columns(2)
        with c1:
            target_role = st.text_input(
                "Target Job Title",
                value=st.session_state.target_job_title,
                placeholder="e.g., AI Engineer, Full Stack Developer",
            )
        with c2:
            target_loc = st.text_input(
                "Target Location / Work Mode",
                value=st.session_state.target_location,
                placeholder="e.g., Dubai, UAE or Remote",
            )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            onboard_clicked = st.button(
                "🚀 Parse CV, Index ChromaDB & Build Queue",
                type="primary",
                use_container_width=True,
                disabled=(active_cv_path is None),
            )
        with btn_col2:
            refresh_queue_clicked = st.button(
                "🔄 Discover Fresh Portals for Location",
                use_container_width=True,
                help="Generates a fresh 5–6 portal queue in SQLite for the entered Target Job Title and Location.",
            )

    if onboard_clicked and active_cv_path is not None:
        with st.status("Running Phase 1 & Phase 2 Onboarding Pipeline...", expanded=True) as status:
            st.write("🧹 Running 7-day sliding TTL cleanup on ChromaDB...")
            cleanup_inactive_sessions(max_inactive_days=7)

            st.write(f"📄 Extracting text from `{active_cv_path.name}` via PyMuPDF...")
            cv_text = extract_text_from_pdf(active_cv_path)

            st.write("🧠 Expanding role titles & extracting core skills with Gemini...")
            profile: CandidateSearchProfile = expand_candidate_role(
                cv_text=cv_text,
                target_job_title=target_role,
                target_location=target_loc,
            )

            st.write("🔢 Embedding CV chunks (`gemini-embedding-001`) into ChromaDB...")
            vec_info = store_cv_in_vector_db(
                cv_path=active_cv_path,
                candidate_name=profile.candidate_name,
            )

            st.session_state.candidate_name = profile.candidate_name
            st.session_state.candidate_id = normalize_candidate_id(profile.candidate_name)
            st.session_state.target_job_title = target_role
            st.session_state.target_location = target_loc
            st.session_state.expanded_job_titles = profile.expanded_job_titles
            st.session_state.top_cv_skills = profile.top_cv_skills
            st.session_state.chunks_stored = vec_info["chunks_stored"]

            existing_q = get_candidate_queue(st.session_state.candidate_id)
            if not existing_q:
                st.write(f"🌍 Discovering regional job portals for `{target_loc}`...")
                rec_list = recommend_portals_for_location(
                    target_job_title=target_role,
                    target_location=target_loc,
                )
                save_portal_queue(
                    candidate_id=st.session_state.candidate_id,
                    target_job_title=target_role,
                    target_location=target_loc,
                    portals=rec_list.portals,
                )
            select_next_portal(st.session_state.candidate_id)
            status.update(
                label=f"✅ Onboarded {profile.candidate_name} ({vec_info['chunks_stored']} CV chunks indexed in ChromaDB)!",
                state="complete",
                expanded=False,
            )

    if refresh_queue_clicked:
        with st.spinner(f"Recommending regional job portals for '{target_role}' in '{target_loc}'..."):
            st.session_state.target_job_title = target_role
            st.session_state.target_location = target_loc
            rec_list = recommend_portals_for_location(
                target_job_title=target_role,
                target_location=target_loc,
            )
            save_portal_queue(
                candidate_id=st.session_state.candidate_id,
                target_job_title=target_role,
                target_location=target_loc,
                portals=rec_list.portals,
            )
            select_next_portal(st.session_state.candidate_id)
            st.toast(f"Created fresh portal queue for {target_loc}!", icon="🌍")

    # Display Expanded Roles & Skills if available
    if st.session_state.expanded_job_titles or st.session_state.top_cv_skills:
        st.divider()
        p_col1, p_col2 = st.columns([1, 1])
        with p_col1:
            st.markdown("**🎯 AI-Expanded Search Titles:**")
            roles_html = "".join(
                f'<span class="role-badge">{title}</span>'
                for title in st.session_state.expanded_job_titles
            )
            st.markdown(roles_html, unsafe_allow_html=True)
        with p_col2:
            if st.session_state.top_cv_skills:
                st.markdown("**🛠️ Extracted Core CV Skills:**")
                skills_html = "".join(
                    f'<span class="skill-badge-match">{skill}</span>'
                    for skill in st.session_state.top_cv_skills[:12]
                )
                st.markdown(skills_html, unsafe_allow_html=True)


# =====================================================================
# ZONE 2: Location Portal Queue & Active Portal Scanner (Phases 2 & 5)
# =====================================================================
queue: list[QueuedPortal] = get_candidate_queue(st.session_state.candidate_id)
active_portal: QueuedPortal | None = select_next_portal(st.session_state.candidate_id) if queue else None

with st.container(border=True):
    st.subheader("2️⃣ Phase 2 & 5: Stateful Regional Job Portal Queue (SQLite)")

    if not queue:
        st.info(
            "No portal queue found for this candidate yet. Click **'🚀 Parse CV, Index ChromaDB & Build Queue'** "
            "or **'🔄 Discover Fresh Portals for Location'** above to initialize your regional links."
        )
    else:
        completed_count = sum(1 for p in queue if p.status == "COMPLETED")
        total_count = len(queue)
        st.progress(
            completed_count / total_count if total_count else 0.0,
            text=f"Queue Progress: {completed_count} / {total_count} Regional Portals Completed",
        )

        # Render horizontal cards for each portal in the queue
        p_cols = st.columns(len(queue))
        for col, item in zip(p_cols, queue):
            with col:
                if item.status == "COMPLETED":
                    badge = "✅ COMPLETED"
                elif item.status == "IN_PROGRESS":
                    badge = "🟢 IN PROGRESS"
                else:
                    badge = "⏳ PENDING"
                st.markdown(f"**Link #{item.link_order}: {item.portal_name}**")
                st.caption(f"{badge} • `{item.fetch_strategy}`")
                st.markdown(f"[Open Portal ↗]({item.portal_url})")

        st.divider()

        if active_portal is not None:
            act_col1, act_col2 = st.columns([1.8, 1.2])
            with act_col1:
                st.markdown(
                    f"### 🎯 Active Portal: Link #{active_portal.link_order} — **{active_portal.portal_name}** (`{active_portal.fetch_strategy}`)"
                )
                st.caption(f"**Why Recommended:** {active_portal.reason} | **URL:** {active_portal.portal_url}")

            with act_col2:
                scan_btn = st.button(
                    f"🔍 Scan Active Jobs (<=30d) on {active_portal.portal_name}",
                    type="primary",
                    use_container_width=True,
                )
                next_btn = st.button(
                    f"⏭️ Mark Link #{active_portal.link_order} Completed & Advance to Next Portal",
                    use_container_width=True,
                )

            if scan_btn:
                with st.status(
                    f"Scanning {active_portal.portal_name} & scoring against ChromaDB CV chunks...",
                    expanded=True,
                ) as scan_status:
                    # Ensure CV is indexed in ChromaDB before RAG scoring
                    if active_cv_path is not None and st.session_state.chunks_stored == 0:
                        st.write("🔢 Verifying CV embeddings in ChromaDB...")
                        vec_info = store_cv_in_vector_db(
                            cv_path=active_cv_path,
                            candidate_name=st.session_state.candidate_name,
                        )
                        st.session_state.chunks_stored = vec_info["chunks_stored"]

                    st.write(
                        f"🌐 Fetching jobs from **{active_portal.portal_name}** via `{active_portal.fetch_strategy}` adapter..."
                    )
                    active_jobs, stats = fetch_and_filter_jobs_for_portal(
                        candidate_id=st.session_state.candidate_id,
                        portal=active_portal,
                        primary_job_title=st.session_state.target_job_title,
                        expanded_job_titles=st.session_state.expanded_job_titles,
                        target_location=st.session_state.target_location,
                        max_results=5,
                    )
                    st.session_state.last_scan_stats = stats
                    st.write(
                        f"🛡️ **Gate #1 (<=30 Days)** & **Gate #2 (Deduplication):** "
                        f"`{stats['raw_fetched']}` raw -> `{stats['verified_active_saved']}` new active jobs saved "
                        f"(`{stats['dropped_expired_over_30d']}` expired dropped, `{stats['dropped_duplicate_fingerprint']}` duplicates skipped)."
                    )

                    # Include any previously discovered jobs on this portal that haven't been scored yet
                    already_scored_fps = {
                        j.fingerprint
                        for j in get_ranked_jobs_from_db(st.session_state.candidate_id)
                    }
                    saved_portal_jobs = get_saved_active_jobs(
                        candidate_id=st.session_state.candidate_id,
                        portal_id=active_portal.id,
                    )
                    jobs_to_score = [
                        j for j in saved_portal_jobs if j.fingerprint not in already_scored_fps
                    ] or active_jobs

                    if jobs_to_score:
                        st.write("🧠 Running RAG Semantic Matcher (`ChromaDB` + `Gemini Rubric`)...")
                        score_and_rank_active_jobs(
                            jobs=jobs_to_score,
                            candidate_name=st.session_state.candidate_name,
                            top_k_chunks=4,
                        )

                    scan_status.update(
                        label=f"✅ Completed scan & RAG evaluation for {active_portal.portal_name}!",
                        state="complete",
                        expanded=False,
                    )
                st.rerun()

            if next_btn:
                next_p = mark_portal_completed_and_advance(
                    candidate_id=st.session_state.candidate_id,
                    portal_id=active_portal.id,
                )
                if next_p:
                    st.toast(
                        f"Link #{active_portal.link_order} marked COMPLETED! Advanced to Link #{next_p.link_order}: {next_p.portal_name}",
                        icon="⏭️",
                    )
                else:
                    st.toast("🎉 All portals in your queue are now COMPLETED!", icon="🎉")
                st.rerun()
        else:
            st.success(
                "🎉 **All regional portals in your queue are COMPLETED!** "
                "Click **'🔄 Discover Fresh Portals for Location'** above whenever you want to start a new scan cycle."
            )


# =====================================================================
# ZONE 3: Ranked Job Leaderboard & On-Demand Career Co-Pilot (Phases 3 & 4)
# =====================================================================
with st.container(border=True):
    st.subheader("3️⃣ Phase 3 & 4: Ranked Active Jobs (`<= 30 Days`) & On-Demand Career Co-Pilot")

    all_ranked_jobs: list[RankedJobMatch] = get_ranked_jobs_from_db(st.session_state.candidate_id)

    if not all_ranked_jobs:
        st.info(
            "No jobs have been scanned and scored yet. Click **'🔍 Scan Active Jobs (<=30d)'** in Zone 2 above to populate your ranked leaderboard!"
        )
    else:
        filter_col1, filter_col2, filter_col3 = st.columns([1.4, 1.2, 1.4])
        with filter_col1:
            portal_names = ["All Scanned Portals"] + sorted({j.portal_name for j in all_ranked_jobs})
            default_idx = 0
            if active_portal and active_portal.portal_name in portal_names:
                default_idx = portal_names.index(active_portal.portal_name)
            selected_portal_filter = st.selectbox(
                "Filter by Portal",
                options=portal_names,
                index=default_idx,
            )
        with filter_col2:
            only_unlocked = st.toggle(
                f"Show Only High-Match Jobs (>= {HIGH_MATCH_THRESHOLD}%)",
                value=False,
            )
        with filter_col3:
            high_match_count = sum(1 for j in all_ranked_jobs if j.copilot_unlocked)
            st.metric(
                label="Total Active Jobs Scored",
                value=f"{len(all_ranked_jobs)} Jobs",
                delta=f"{high_match_count} Co-Pilot Eligible (>= {HIGH_MATCH_THRESHOLD}%)",
            )

        filtered_jobs = all_ranked_jobs
        if selected_portal_filter != "All Scanned Portals":
            filtered_jobs = [j for j in filtered_jobs if j.portal_name == selected_portal_filter]
        if only_unlocked:
            filtered_jobs = [j for j in filtered_jobs if j.copilot_unlocked]

        st.divider()

        if not filtered_jobs:
            st.warning("No jobs match the current filter criteria. Try switching to 'All Scanned Portals'.")
        else:
            for idx, job in enumerate(filtered_jobs, start=1):
                score_icon = "🟢" if job.copilot_unlocked else "🟡"
                status_text = (
                    f"✨ Co-Pilot Unlocked (>={HIGH_MATCH_THRESHOLD}%)"
                    if job.copilot_unlocked
                    else f"🔒 Co-Pilot Locked (<{HIGH_MATCH_THRESHOLD}%)"
                )

                with st.container(border=True):
                    top_c1, top_c2 = st.columns([3.2, 1.2])
                    with top_c1:
                        st.markdown(
                            f"#### #{idx}. {job.job_title} — **{job.company_name}**"
                        )
                        st.caption(
                            f"📍 **{job.location}** • 🌐 Portal: `{job.portal_name}` • "
                            f"🕒 Posted: **{job.posted_date_raw}** (`{job.days_ago}d ago` / `{job.posted_date_iso}`) • "
                            f"🔗 [View Original Job Posting ↗]({job.job_url})"
                        )
                    with top_c2:
                        st.markdown(
                            f"<div style='text-align:right; font-size:1.35rem; font-weight:700;'>"
                            f"{score_icon} {job.match_score}% Match</div>"
                            f"<div style='text-align:right; font-size:0.82rem; opacity:0.8;'>"
                            f"{status_text}</div>",
                            unsafe_allow_html=True,
                        )

                    st.write(f"**Role Summary:** {job.description_summary}")
                    st.info(f"**🧠 RAG Match Analysis:** {job.match_reasoning}")

                    sk_col1, sk_col2 = st.columns(2)
                    with sk_col1:
                        st.markdown("**✅ Matching CV Skills:**")
                        if job.matching_skills:
                            m_html = "".join(
                                f'<span class="skill-badge-match">{s}</span>'
                                for s in job.matching_skills
                            )
                            st.markdown(m_html, unsafe_allow_html=True)
                        else:
                            st.caption("None identified")

                    with sk_col2:
                        st.markdown("**⚠️ Missing Skills / Gaps:**")
                        if job.missing_skills:
                            g_html = "".join(
                                f'<span class="skill-badge-gap">{s}</span>'
                                for s in job.missing_skills
                            )
                            st.markdown(g_html, unsafe_allow_html=True)
                        else:
                            st.caption("No critical skill gaps!")

                    # Phase 4: On-Demand Career Co-Pilot Gate (Strictly >= 70%)
                    if job.copilot_unlocked:
                        st.divider()
                        btn_key = f"copilot_btn_{job.fingerprint}"
                        regen_key = f"copilot_regen_{job.fingerprint}"
                        is_cached = job.fingerprint in st.session_state.copilot_kits_cache

                        cp_c1, cp_c2 = st.columns([2, 1])
                        with cp_c1:
                            launch_copilot = st.button(
                                "✨ Launch On-Demand Career Co-Pilot (CV Bullets, Cover Letter, Courses & Interview Q&A)"
                                if not is_cached
                                else "📂 View Generated Career Co-Pilot Kit",
                                key=btn_key,
                                type="primary" if not is_cached else "secondary",
                            )
                        with cp_c2:
                            force_regen = st.button(
                                "🔄 Regenerate Co-Pilot Kit",
                                key=regen_key,
                            )

                        if launch_copilot or force_regen:
                            with st.spinner(
                                f"Generating grounded Career Co-Pilot Kit for {job.job_title} @ {job.company_name}..."
                            ):
                                kit = generate_copilot_kit_for_job(
                                    job=job,
                                    candidate_name=st.session_state.candidate_name,
                                    force_regenerate=force_regen,
                                )
                                st.session_state.copilot_kits_cache[job.fingerprint] = kit

                        if job.fingerprint in st.session_state.copilot_kits_cache:
                            kit: CareerCoPilotKit = st.session_state.copilot_kits_cache[job.fingerprint]
                            tab1, tab2, tab3, tab4 = st.tabs(
                                [
                                    "📄 1. ATS CV Bullets",
                                    "✉️ 2. Tailored Cover Letter",
                                    "🛠️ 3. Skill-Gap Courses & Projects",
                                    "🎯 4. Custom Interview Q&A",
                                ]
                            )

                            with tab1:
                                st.markdown(
                                    "**ATS-Optimized Resume Bullet Points (100% Grounded in Your Actual CV):**"
                                )
                                for b in kit.tailored_cv_bullets:
                                    st.markdown(f"- {b}")

                            with tab2:
                                st.markdown("**Tailored Cover Letter / Hiring Manager Outreach Message:**")
                                st.text_area(
                                    label="Copy Cover Letter",
                                    value=kit.tailored_cover_letter,
                                    height=220,
                                    key=f"cl_text_{job.fingerprint}",
                                )

                            with tab3:
                                st.markdown("**Actionable Courses & Weekend Mini-Projects for Missing Skills:**")
                                for item in kit.skill_gap_roadmap:
                                    st.markdown(
                                        f"- **Missing Skill: `{item.missing_skill}`**\n"
                                        f"  - 📚 **Course / Track:** {item.recommended_course}\n"
                                        f"  - 🛠️ **Weekend Mini-Project:** {item.hands_on_mini_project}"
                                    )

                            with tab4:
                                st.markdown(
                                    "**Role-Specific Technical & Behavioral Interview Q&A (Grounded in Your CV):**"
                                )
                                for q_idx, qa in enumerate(kit.interview_prep_qa, start=1):
                                    with st.expander(f"Q{q_idx}: {qa.question}", expanded=(q_idx == 1)):
                                        st.caption(f"🎯 **What They Are Testing:** {qa.why_they_ask}")
                                        st.markdown(f"**✅ Sample Answer (From Your CV):**\n\n{qa.cv_grounded_answer}")

                            md_content = format_copilot_markdown(job, kit)
                            st.download_button(
                                label="📥 Download Complete Co-Pilot Kit (.md)",
                                data=md_content,
                                file_name=f"CoPilot_{job.company_name}_{job.job_title}.md".replace(" ", "_"),
                                mime="text/markdown",
                                key=f"dl_{job.fingerprint}",
                            )
                    else:
                        st.caption(
                            f"🔒 *On-Demand Career Co-Pilot is locked because this role scored {job.match_score}% "
                            f"(below the {HIGH_MATCH_THRESHOLD}% high-match threshold).*"
                        )
