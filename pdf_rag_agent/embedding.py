from sentence_transformers import SentenceTransformer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Generate local embeddings for text using Sentence Transformers."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model = SentenceTransformer(model_id)

    def embed_texts(self, texts: list[str]):
        """Convert a list of text strings into embedding vectors."""
        if not texts:
            return []

        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("texts must contain only non-empty strings.")

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str):
        """Convert one question into an embedding vector."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        return self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )