from pathlib import Path

from pdf_rag_agent.pdf_loader import extract_pdf_pages
from pdf_rag_agent.text_splitter import split_text


def process_pdf(pdf_path: str | Path, chunk_size: int = 500, chunk_overlap: int = 50,) -> list[dict[str, str | int]]:
    """Extract PDF text and split it into chunks with page metadata."""

    pages = extract_pdf_pages(pdf_path)
    documents = []

    for page in pages:
        chunks = split_text(
            page["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk in chunks:
            documents.append(
                {
                    "page": page["page"],
                    "text": chunk,
                }
            )

    return documents