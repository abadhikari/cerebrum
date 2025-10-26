import numpy as np
from numpy.typing  import NDArray
from typing import Protocol

# Type alias representing a numeric embedding vector or matrix.
# Each embedding is a NumPy NDArray with a floating-point dtype
# (e.g., float32, float64) and shape (n, dim), where `n` is the
# number of items being embedded and `dim` is the model’s output
# dimensionality.
Embedding = NDArray[np.floating]


class Embedder(Protocol):
    """
    Interface for embedding backends. All concrete embedders must 
    implement the core methods for single and batched text 
    embeddings, and expose the output dimensionality.
    """

    def embed(self, text: str) -> Embedding:
        """
        Convert a single text string into a numeric embedding.

        Args:
            text (str): Text to embed.

        Returns:
            Embedding: A 2D array of shape (1, dim).
        """
        ...

    def embed_batch(self, texts: list[str]) -> Embedding:
        """
        Convert multiple text strings into numeric embeddings.

        Args:
            texts (list[str]): List of text strings to embed.

        Returns:
            Embedding: A 2D array of shape (n, dim).
        """
        ...

    def get_dimensions(self) -> int:
        """
        Return the dimensionality of vectors produced by the embedder.

        Returns:
            int: Embedding dimensionality.
        """
        ...
