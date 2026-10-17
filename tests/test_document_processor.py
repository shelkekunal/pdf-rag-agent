from reportlab.pdfgen import canvas

from pdf_rag_agent.document_processor import process_pdf


def test_process_pdf_preserves_page_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(50, 750, "A" * 40)
    pdf.showPage()
    pdf.drawString(50, 750, "B" * 40)
    pdf.save()

    documents = process_pdf(
        pdf_path,
        chunk_size=20,
        chunk_overlap=5,
    )

    assert len(documents) > 2
    assert all(document["page"] == 1 for document in documents if document["text"].startswith("A"))
    assert all(document["page"] == 2 for document in documents if document["text"].startswith("B"))
    assert all(len(document["text"]) <= 20 for document in documents)