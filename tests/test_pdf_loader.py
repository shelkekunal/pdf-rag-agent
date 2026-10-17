from reportlab.pdfgen import canvas
import pytest

from pdf_rag_agent.pdf_loader import extract_pdf_pages

def test_missing_pdf_raises_error(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(missing_pdf)


def test_non_pdf_extension_raises_error(tmp_path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="\\.pdf"):
        extract_pdf_pages(text_file)


def test_invalid_input_type_raises_error():
    with pytest.raises(TypeError):
        extract_pdf_pages(123)


def test_directory_path_raises_error(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        extract_pdf_pages(tmp_path)


def test_malformed_pdf_raises_error(tmp_path):
    malformed_pdf = tmp_path / "broken.pdf"
    malformed_pdf.write_bytes(b"This is not a valid PDF")

    with pytest.raises(RuntimeError, match="Failed to extract"):
        extract_pdf_pages(malformed_pdf)

def test_extract_text_from_valid_pdf(tmp_path):
    # Arrange: create a temporary PDF with known text.
    pdf_path = tmp_path / "sample.pdf"

    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(100, 750, "Hello PDF RAG")
    pdf.save()

    # Act: extract the PDF text.
    pages = extract_pdf_pages(pdf_path)

    # Assert: verify the extracted result.
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert "Hello PDF RAG" in pages[0]["text"]

def test_extract_text_from_multiple_pages(tmp_path):
    pdf_path = tmp_path / "multi_page.pdf"

    pdf = canvas.Canvas(str(pdf_path))

    pdf.drawString(100, 750, "Page one")
    pdf.showPage()

    # Leave page 2 blank.
    pdf.showPage()

    pdf.drawString(100, 750, "Page three")
    pdf.save()

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert "Page one" in pages[0]["text"]
    assert pages[1]["page"] == 3
    assert "Page three" in pages[1]["text"]