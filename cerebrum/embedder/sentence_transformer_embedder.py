import numpy as np

from sentence_transformers import SentenceTransformer

from cerebrum.embedder.embedder import EmbeddingRecord


class SentenceTransformerEmbedder:
  """
  Embedding class using a SentenceTransformer model.

  Provides single and batched text embedding with enforced dtype normalization.
  Ensures outputs are always 2D, float arrays (shape: (n, dim)) and memory-contiguous,
  making them directly usable with FAISS or other vector search backends.
  """

  def __init__(self, model_name: str, dtype=np.float32):
    """
    Initialize the SentenceTransformer embedder.

    Args:
        model_name (str): Name of the SentenceTransformer model to load.
        dtype (np.dtype): Floating-point dtype to cast embeddings to. 
                          Must be a subtype of np.floating (default: np.float32).

    Raises:
        TypeError: If dtype is not a floating-point type.
    """
    if not np.issubdtype(dtype, np.floating):
      raise TypeError(f'dtype must be a floating-point dtype. Got {dtype}')

    self._model = SentenceTransformer(model_name)
    self._model_name = model_name
    self._dtype = dtype
    self._dimensions = self._model.get_sentence_embedding_dimension()

  def embed(self, text: str) -> EmbeddingRecord:
    """
    Embed a single text string.

    Args:
        text (str): Text to embed.

    Returns:
        EmbeddingRecord: A 2D array of shape (1, dim) embedding and its metadata.
    """
    return self.embed_batch([text])

  def embed_batch(self, texts: list[str]) -> EmbeddingRecord:
    """
    Embed a batch of texts.

    Args:
        texts (list[str]): list of text strings to embed.

    Returns:
        EmbeddingRecord: A 2D array of shape (n, dim) embedding and its metadata. with dtype matching `self._dtype`.
                    Returns an empty array of shape (0, dim) if `texts` is empty.
    """
    if len(texts) == 0:
      return EmbeddingRecord(
          vector=np.empty((0, self._dimensions), dtype=self._dtype),
          model_name=self._model_name
      )

    vector = self._model.encode(texts, convert_to_numpy=True).astype(self._dtype)
    if vector.ndim == 1:
      vector = vector.reshape(1, self._dimensions)
    return EmbeddingRecord(
      vector=np.ascontiguousarray(vector, dtype=self._dtype),
      model_name=self._model_name
    )

  def get_dimensions(self) -> int:
    """
    Get the embedding dimensionality of the underlying model.

    Returns:
        int: The number of dimensions in each embedding vector.
    """
    return self._dimensions
