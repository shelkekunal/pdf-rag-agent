
import faiss
import numpy as np


class VectorStore:
    """Store document embeddings and retrieve similar documents."""

    def __init__(self, dimension: int = 384) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer.")

        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[dict[str, str | int]] = []
        self.dimension = dimension

    def add_documents(
        self,
        documents: list[dict[str, str | int]],
        embeddings: np.ndarray,
    ) -> None:
        """Store document vectors and their associated metadata."""

        if not documents:
            return

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings."
            )

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings must have shape (n, {self.dimension})."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain only finite values.")

        if any(not isinstance(doc, dict) for doc in documents):
            raise ValueError("Each document must be a dictionary.")

        if any(
            not isinstance(doc.get("text"), str) or not doc["text"].strip()
            for doc in documents
        ):
            raise ValueError("Each document must contain non-empty text.")

        if any(
            not isinstance(doc.get("page"), int)
            or isinstance(doc.get("page"), bool)
            or doc["page"] < 1
            for doc in documents
        ):
            raise ValueError("Each document must have a positive page number.")

        self.index.add(vectors)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict[str, str | int | float]]:
        """Return the most similar documents for a query vector."""

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.shape == (1, self.dimension):
            query = query.reshape(-1)

        if query.shape != (self.dimension,):
            raise ValueError(
                f"Query embedding must have shape ({self.dimension},)."
            )

        if not np.isfinite(query).all():
            raise ValueError("Query embedding must contain only finite values.")

        scores, indices = self.index.search(
            query.reshape(1, -1).astype(np.float32),
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            results.append({
                **self.documents[index],
                "score": float(score),
            })

        return results
