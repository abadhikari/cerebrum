import numpy as np
import os

import faiss

from .semantic_store import Distances, Ids
from ..embedder import Embedding


class FaissClient:
  """
  A thin wrapper around FAISS providing write, query, and persistence operations 
  for a semantic embedding index.

  This implementation uses `IndexIDMap2` around `IndexFlatIP` to support custom 
  integer IDs and cosine-similarity-style searches.
  """

  def __init__(self, index_filepath: str, dimensions: int):
    """
    Initialize the FAISS client and load (or create) the index.

    Args:
        index_filepath (str): Path to the FAISS index file on disk.
        dimensions (int): Dimensionality of the embeddings stored in the index.

    Raises:
        ValueError: If an existing index is found but its dimensions differ 
                    from the provided value.
    """
    self._index_filepath = index_filepath
    self._dimensions = dimensions
    self._index = self._load_index()

  def _load_index(self) -> faiss.IndexIDMap2:
    """
    Load the FAISS index from disk or create a new one if not found.

    Returns:
        faiss.IndexIDMap2: The loaded or newly created FAISS index.

    Raises:
        ValueError: If an existing index has a dimension mismatch.
    """
    if not os.path.exists(self._index_filepath):
      return self._create_index_id_map_index()
    
    try:
      index = faiss.read_index(self._index_filepath)
      if index.d != self._dimensions:
        raise ValueError(f"Dimension mismatch: index has {index.d}, but expected {self._dimensions}")
      return index

    except OSError:
      return self._create_index_id_map_index()
  
  def _create_index_id_map_index(self) -> faiss.IndexIDMap2:
    """
    Create a new empty FAISS index wrapped with an ID mapping layer.

    Returns:
        faiss.IndexIDMap2: A new FAISS index ready for use.
    """
    return faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimensions))
  
  def write_index(self) -> None:
    """
    Persist the current in-memory FAISS index to disk.

    Raises:
        OSError: If the index cannot be written to the specified filepath.
    """
    faiss.write_index(self._index, self._index_filepath)
  
  def write(self, embedding: Embedding, ids: Ids) -> None:
    """
    Add one or more embeddings and their associated IDs to the FAISS index.

    Args:
        embedding (Embedding): 
            A 2D array of shape (n, d) containing the embeddings to insert.
        ids (Ids): 
            A 1D array of shape (n,) containing integer IDs matching each embedding.

    Raises:
        TypeError: If the dtype of embeddings or IDs is invalid.
        ValueError: If embedding or ID shapes are inconsistent.
    """
    normalized_embedding = self._normalize_embedding(embedding)
    normalized_ids = self._normalize_ids(ids)
    self._index.add_with_ids(normalized_embedding, normalized_ids)

  def _normalize_embedding(self, embedding: Embedding) -> Embedding:
    """
    Validate and normalize an embedding array for FAISS.
    Ensures dtype is float32, layout is contiguous, and L2-normalization is applied.

    Args:
        embedding (Embedding): Input embedding array.

    Returns:
        Embedding: Normalized embedding array.

    Raises:
        TypeError: If dtype is not float32.
        ValueError: If shape does not match expected dimensions.
    """
    if embedding.dtype != np.float32:
      raise TypeError(f'dtype must be np.float32 type for FAISS. Got {embedding.dtype}')
    if embedding.ndim != 2 or embedding.shape[1] != self._dimensions:
      raise ValueError(f"Expected shape (*, {self._dimensions}), got {embedding.shape}")
    if not embedding.flags.c_contiguous:
      embedding = np.ascontiguousarray(embedding)

    faiss.normalize_L2(embedding)
    return embedding

  def _normalize_ids(self, ids: Ids) -> Ids:
    """
    Validate and normalize a list of IDs for FAISS.
    Ensures IDs are 1D and of dtype int64.

    Args:
        ids (Ids): Array of integer IDs.

    Returns:
        Ids: Normalized int64 ID array.

    Raises:
        ValueError: If IDs are not 1D.
    """
    if ids.ndim != 1:
      raise ValueError(f"IDs must be 1D, got shape {ids.shape}")
    return ids.astype(np.int64, copy=False)
    
  def query(self, embedding: Embedding, k: int) -> tuple[Distances, Ids]:
    """
    Search the FAISS index for the top-k most similar embeddings.

    Uses an inner-product index (`IndexFlatIP`) with L2-normalized embeddings, 
    making the returned scores equivalent to cosine similarity values 
    (higher = more similar).

    Args:
        embedding (Embedding): 
            A single embedding of shape (1, d) to query.
        k (int): 
            The number of nearest neighbors to retrieve.

    Returns:
        tuple[Distances, Ids]: 
            - Distances: 1D array of cosine similarity scores for the top-k matches.
            - Ids: 1D array of integer IDs corresponding to the nearest embeddings.

    Raises:
        TypeError: If the query embedding has an invalid dtype or shape.
    """
    normalized_embedding = self._normalize_embedding(embedding)
    D, I = self._index.search(normalized_embedding, k)
    return D[0], I[0]
