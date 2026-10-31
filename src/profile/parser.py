from pathlib import Path
import pymupdf


# Automatically locate the project root folder (C:\krish_naik\coursera\job_hunt)
# no matter which directory you run the script from!
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extracts and cleans full text from any candidate's PDF CV.
    """
    # 1. Validate the file path
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"CV file not found at: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, but got: {path.suffix}")

    # 2. Open PDF and extract text from every page
    pages_text: list[str] = []
    with pymupdf.open(path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages_text.append(text)

    # 3. Combine all pages into one clean string
    full_text = "\n\n".join(pages_text).strip()

    # 4. Guard against scanned/image-only PDFs with no selectable text
    if not full_text:
        raise ValueError(
            f"No readable text found in '{path.name}'. "
            "Please upload a text-based PDF (not a scanned image)."
        )

    return full_text


def chunk_cv_text(
    cv_text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[str]:
    """
    Splits full CV text into line-aware overlapping chunks for ChromaDB RAG storage.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    lines = [line.strip() for line in cv_text.splitlines() if line.strip()]
    chunks: list[str] = []
    current_lines: list[str] = []
    current_length = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline separator

        # If adding this line exceeds chunk_size and we already have content, save the chunk
        if current_length + line_len > chunk_size and current_lines:
            chunks.append("\n".join(current_lines))

            # Build overlap for the next chunk from the end of current_lines
            overlap_lines: list[str] = []
            overlap_len = 0
            for prev_line in reversed(current_lines):
                if overlap_len + len(prev_line) + 1 <= overlap:
                    overlap_lines.insert(0, prev_line)
                    overlap_len += len(prev_line) + 1
                else:
                    break

            current_lines = overlap_lines
            current_length = overlap_len

        current_lines.append(line)
        current_length += line_len

    # Append any remaining lines as the final chunk
    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


if __name__ == "__main__":
    # Resolve sample CV path relative to PROJECT_ROOT so it works from any folder
    sample_cv = PROJECT_ROOT / "data" / "ZebaNajeeb_CV.pdf"
    try:
        extracted_cv = extract_text_from_pdf(sample_cv)
        chunks = chunk_cv_text(extracted_cv, chunk_size=800, overlap=150)

        print(f"Successfully parsed CV: {sample_cv.name}")
        print(f"Total characters extracted: {len(extracted_cv)}")
        print(f"Total chunks created: {len(chunks)}\n")

        print(f"=== CHUNK 1 ({len(chunks[0])} chars) ===")
        print(chunks[0])
        print(f"\n=== CHUNK 2 ({len(chunks[1])} chars) ===")
        print(chunks[1])
    except Exception as e:
        print(f"Error parsing CV: {e}")
