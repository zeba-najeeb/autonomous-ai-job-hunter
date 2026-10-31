import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.profile.parser import PROJECT_ROOT, extract_text_from_pdf

# Load environment variables from C:\krish_naik\coursera\job_hunt\.env
load_dotenv(PROJECT_ROOT / ".env", override=True)


class CandidateSearchProfile(BaseModel):
    """Structured output schema for candidate onboarding and smart role expansion."""
    candidate_name: str = Field(
        description="Full name of the candidate extracted from the CV."
    )
    primary_job_title: str = Field(
        description="The primary target job title entered by the candidate."
    )
    target_location: str = Field(
        description="The target city, country, or remote work preference entered by the candidate."
    )
    expanded_job_titles: list[str] = Field(
        description="3 to 4 closely related job titles that match both the primary_job_title and the candidate's actual CV skills."
    )
    top_cv_skills: list[str] = Field(
        description="Top 8 to 12 core technical skills and tools extracted directly from the candidate's CV."
    )


def expand_candidate_role(
    cv_text: str,
    target_job_title: str,
    target_location: str,
) -> CandidateSearchProfile:
    """
    Combines the candidate's CV text with their desired Job Title and Location
    to extract core skills and generate 3-4 realistic, closely related job titles.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise ValueError(
            "GEMINI_API_KEY is missing or still set to 'PASTE_YOUR_GEMINI_API_KEY_HERE'! "
            "Please paste your real key in C:\\krish_naik\\coursera\\job_hunt\\.env and press Ctrl+S to save."
        )

    if not target_job_title.strip() or not target_location.strip():
        raise ValueError("Both target_job_title and target_location must be provided.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an expert Technical Recruiter and AI Career Strategist.
    
    Analyze the candidate's CV below alongside their target role and location:
    - Target Job Title entered by candidate: "{target_job_title}"
    - Target Location entered by candidate: "{target_location}"
    
    Tasks:
    1. Extract the candidate's full name and top 8-12 technical skills strictly from the CV.
    2. Suggest 3 to 4 closely related job titles that companies use in job postings for someone with these exact skills and target role.
       (Do NOT suggest unrelated roles or seniority levels that contradict the CV).

    CANDIDATE CV TEXT:
    {cv_text}
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
                    response_schema=CandidateSearchProfile,
                    temperature=0.2,
                ),
            )
            break
        except Exception as err:
            print(f"   [Fallback Notice] Model '{model_name}' busy ({type(err).__name__}), switching to backup model...")
            last_error = err

    if response is None:
        raise RuntimeError(f"All models in fallback chain failed. Last error: {last_error}")

    return CandidateSearchProfile.model_validate_json(response.text)


if __name__ == "__main__":
    # Interactive test: works for ANY candidate's CV, job title, and location!
    sample_cv = PROJECT_ROOT / "data" / "ZebaNajeeb_CV.pdf"

    print("=== Candidate Onboarding & Role Expander ===")
    job_title_input = input("Enter Target Job Title (e.g., AI Engineer): ").strip() or "AI Engineer"
    location_input = input("Enter Target Location (e.g., Dubai, UAE): ").strip() or "Dubai, UAE"

    try:
        print(f"\n1. Extracting text from {sample_cv.name}...")
        cv_text = extract_text_from_pdf(sample_cv)

        print("2. Analyzing CV & expanding target job titles with Gemini...")
        profile = expand_candidate_role(
            cv_text=cv_text,
            target_job_title=job_title_input,
            target_location=location_input,
        )

        print("\n=== GENERATED CANDIDATE SEARCH PROFILE ===")
        print(f"Candidate Name     : {profile.candidate_name}")
        print(f"Primary Job Title  : {profile.primary_job_title}")
        print(f"Target Location    : {profile.target_location}")
        print(f"Similar Job Titles : {', '.join(profile.expanded_job_titles)}")
        print(f"Top CV Skills      : {', '.join(profile.top_cv_skills)}")
    except Exception as e:
        print(f"\nError: {e}")
