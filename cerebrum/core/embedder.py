from dataclasses import dataclass
import numpy as np
from numpy.typing  import NDArray
from typing import Protocol

# Type alias representing a numeric embedding 2D matrix.
# Each embedding is a NumPy NDArray with a floating-point dtype
# (e.g., float32, float64) and shape (n, dim), where `n` is the
# number of items being embedded and `dim` is the model’s output
# dimensionality.
Embedding = NDArray[np.floating]


@dataclass
class EmbeddingRecord:
    """
    Represents an embedding vector and its source model.

    Attributes:
        vector (Embedding): The numeric embedding matrix 
            produced by the model.
        model_name (str): The name or identifier of the model used to
            generate the embedding.
    """
    vector: Embedding
    model_name: str


class Embedder(Protocol):
    """
    Interface for embedding backends. All concrete embedders must 
    implement the core methods for single and batched text 
    embeddings, and expose the output dimensionality.
    """

    def embed(self, text: str) -> EmbeddingRecord:
        """
        Convert a single text string into a numeric embedding.

        Args:
            text (str): Text to embed.

        Returns:
            EmbeddingRecord: A 2D array of shape (1, dim) embedding and its metadata.
        """
        ...

    def embed_batch(self, texts: list[str]) -> EmbeddingRecord:
        """
        Convert multiple text strings into numeric embeddings.

        Args:
            texts (list[str]): list of text strings to embed.

        Returns:
            EmbeddingRecord: A 2D array of shape (n, dim) embedding and its metadata. 
                             For an empty list, shape (0, dim).
        """
        ...

    def get_dimensions(self) -> int:
        """
        Return the dimensionality of vectors produced by the embedder.

        Returns:
            int: Embedding dimensionality (dim).
        """
        ...
