import numpy as np
from numpy.typing import NDArray
from typing import Protocol

from ..embedder import Embedding

# 1D array of float values representing similarity or distance scores.
Distances = NDArray[np.floating]

# 1D array of integer identifiers corresponding to embeddings in the store.
Ids = NDArray[np.integer]


class SemanticStore(Protocol):
  def write(self, embedding: Embedding, ids: Ids) -> None:
    """
    Add one or more embeddings to the semantic store with their associated IDs.

    Args:
        embedding (Embedding): 
            A 2D NumPy array of shape (n, d), where each row represents a vector embedding 
            to be stored. Must be contiguous and of dtype float32.
        ids (Ids): 
            A 1D NumPy array of shape (n,) containing integer IDs corresponding to each 
            embedding. Each ID should be unique within the store.
    """
    ...

  def query(self, embedding: Embedding, k: int) -> tuple[Distances, Ids]:
    """
    Search the semantic store for the top-k most similar vectors to the given embedding.

    Args:
        embedding (Embedding): 
            A single embedding of shape (1, d) to query against the store.
        k (int): 
            The number of nearest neighbors to retrieve.

    Returns:
        tuple[Distances, Ids]: 
            A tuple containing:
              - Distances (NDArray[np.floating]): 1D array of similarity scores or distances.
              - Ids (NDArray[np.integer]): 1D array of corresponding IDs for the top-k matches.
    """
    ...
  