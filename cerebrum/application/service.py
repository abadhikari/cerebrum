from dataclasses import dataclass
from cerebrum.core.thought import Thought
from cerebrum.infra.embedder import Embedder
from cerebrum.infra.semantic_store import Distances, Ids, SemanticStore
from cerebrum.infra.repository import Index, ThoughtRecord, ThoughtRepository, ThoughtStatus

@dataclass(frozen=True)
class SearchHit:
    """
    A ranked semantic search result.

    Attributes:
        record (ThoughtRecord): The retrieved thought metadata/content.
        score (float): Cosine-similarity score (higher = more similar).
        rank (int): Zero-based rank in the search results.
    """
    record: ThoughtRecord
    score: float
    rank: int


class Service:
    """
    High-level application service coordinating embedding, persistence,
    and semantic search.

    This layer hides infrastructure details and exposes simple operations 
    for adding thoughts and querying them.
    """

    def __init__(self, thought_repository: ThoughtRepository, embedder: Embedder, semantic_store: SemanticStore):
        """
        Initialize the service with its dependencies.

        Args:
            thought_repository (ThoughtRepository):
                Persistent storage for thoughts and index metadata.
            embedder (Embedder):
                Backend capable of converting text into embedding vectors.
            semantic_store (SemanticStore):
                Semantic store for nearest-neighbor search.
        """
        self._thought_repository = thought_repository
        self._embedder = embedder
        self._semantic_store = semantic_store
    
    def add_thought(self, thought: Thought, index_id: str) -> int:
        """
        Insert a new thought into the system.

        Args:
            thought (Thought): Domain object containing the thought body and metadata.
            index_id (str): Identifier of the semantic index to attach the thought to.

        Returns:
            int: The assigned id64 for the new thought.
        """
        embedding = self._embedder.embed(thought.body)
        id64 = self._thought_repository.insert_thought(thought, embedding, index_id)
        self._semantic_store.write(embedding.vector, [id64])
        self._thought_repository.complete_thought_insert(id64)
        return id64

    def query(self, query: str, index_id: str, k: int) -> list[SearchHit]:
        """
        Perform a semantic search over the given index.

        Args:
            query (str): Raw text query to embed and search with.
            index_id (str): Identifier of the semantic index to search.
            k (int): Max number of nearest neighbors to retrieve.

        Returns:
            list[SearchHit]: Ranked list of matching thoughts.
        """
        embedding = self._embedder.embed(query)
        similarities, ids = self._semantic_store.query(embedding.vector, k)
        thoughts = self._thought_repository.retrieve_thoughts(ids, index_id, ThoughtStatus.ACTIVE)
        return self._create_search_hits(thoughts, similarities, ids)
    
    def _create_search_hits(self, thoughts: list[ThoughtRecord], similarities: Distances, ids: Ids) -> list[SearchHit]:
        """
        Pair repository results with semantic map ranking output.

        Args:
            thoughts (list[ThoughtRecord]): Fetched thought records.
            similarities (Distances): Similarity scores for each id.
            ids (Ids): id64s returned by semantic map, ordered by rank.

        Returns:
            list[SearchHit]: Search results with rank, score, and full record.
        """
        thoughts_map = {thought.id64: thought for thought in thoughts}
        search_hits: list[SearchHit] = []
        for i, id in enumerate(ids):
            thought_record = thoughts_map[id]
            similarity_score = float(similarities[i])
            search_hit = SearchHit(
                record=thought_record,
                score=similarity_score,
                rank=i
            )
            search_hits.append(search_hit)
        return search_hits
    
    def create_index(self, index_name: str, algorithm: str) -> str:
        """
        Create a new semantic index in the repository.

        Args:
            index_name (str): Human-readable index name.
            algorithm (str): Indexing algorithm tag (e.g. 'faiss-flat').

        Returns:
            str: The generated index_id.
        """
        return self._thought_repository.create_index(index_name, algorithm)
    
    def get_indexes(self) -> list[Index]:
        """
        Return all known semantic indexes.

        Returns:
            list[Index]: Metadata for each index defined in the repository.
        """
        return self._thought_repository.list_indexes()
    