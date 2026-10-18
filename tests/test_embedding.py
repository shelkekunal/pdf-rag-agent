import numpy as np
import pytest

from pdf_rag_agent.embedding import EmbeddingModel


class FakeSentenceTransformer:
    """Fake model for testing without downloading a real model."""

    def __init__(self, model_id):
        self.model_id = model_id

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array([1.0, 0.0, 0.0])

        return np.array([
            [1.0, 0.0, 0.0]
            for _ in texts
        ])


@pytest.fixture
def embedding_model(monkeypatch):
    monkeypatch.setattr(
        "pdf_rag_agent.embedding.SentenceTransformer",
        FakeSentenceTransformer,
    )
    return EmbeddingModel()


def test_embed_texts_returns_vectors(embedding_model):
    vectors = embedding_model.embed_texts(["PDF retrieval", "RAG chatbot"])

    assert vectors.shape == (2, 3)
    assert isinstance(vectors, np.ndarray)


def test_embed_empty_list_returns_empty_list(embedding_model):
    assert embedding_model.embed_texts([]) == []


def test_embed_texts_rejects_empty_strings(embedding_model):
    with pytest.raises(ValueError):
        embedding_model.embed_texts(["valid text", " "])


def test_embed_query_returns_vector(embedding_model):
    vector = embedding_model.embed_query("What is this PDF about?")

    assert vector.shape == (3,)


def test_embed_query_rejects_empty_query(embedding_model):
    with pytest.raises(ValueError):
        embedding_model.embed_query(" ")