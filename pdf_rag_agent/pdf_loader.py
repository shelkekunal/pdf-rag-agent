
from pathlib import Path

from pypdf import PdfReader


def extract_pdf_pages(pdf_path: str | Path) -> list[dict[str, str | int]]:
    """Extract usable text from a PDF, preserving original page numbers."""

    if not isinstance(pdf_path, (str, Path)):
        raise TypeError("pdf_path must be a string or pathlib.Path.")

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise ValueError(f"The provided path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The input file must have a .pdf extension.")

    try:
        reader = PdfReader(path)

        if reader.is_encrypted:
            raise ValueError(f"Encrypted PDFs are not supported: {path}")

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                pages.append(
                    {
                        "page": page_number,
                        "text": text.strip(),
                    }
                )

        return pages

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to extract text from PDF: {path}") from exc
