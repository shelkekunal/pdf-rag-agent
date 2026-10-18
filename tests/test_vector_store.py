
import numpy as np
import pytest

from pdf_rag_agent.vector_store import VectorStore


@pytest.fixture
def sample_data():
    documents = [
        {"page": 1, "text": "Introduction to RAG"},
        {"page": 2, "text": "Vector search with FAISS"},
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    return documents, embeddings


def test_add_and_search_documents(sample_data):
    documents, embeddings = sample_data
    store = VectorStore(dimension=3)

    store.add_documents(documents, embeddings)
    results = store.search(np.array([1.0, 0.0, 0.0]), top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "Introduction to RAG"
    assert results[0]["page"] == 1
    assert results[0]["score"] == pytest.approx(1.0)


def test_empty_store_returns_no_results():
    store = VectorStore(dimension=3)

    assert store.search(np.array([1.0, 0.0, 0.0])) == []


def test_mismatched_document_and_embedding_counts():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add_documents(
            [{"page": 1, "text": "RAG"}],
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )


def test_invalid_embedding_dimension():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add_documents(
            [{"page": 1, "text": "RAG"}],
            np.array([[1.0, 0.0]]),
        )


def test_invalid_top_k():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.search(np.array([1.0, 0.0, 0.0]), top_k=0)
