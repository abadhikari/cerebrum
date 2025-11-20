import logging

from cerebrum.core.thought import Thought
from cerebrum.core.embedder import Embedder
from cerebrum.core.semantic_store import Distances, Ids, SemanticStore
from cerebrum.core.repository import Index, ThoughtRecord, ThoughtRepository, ThoughtStatus
from cerebrum.core.errors import NoEmbeddingsError
from cerebrum.core.search import SearchHit, SearchResult, SearchStatus
from cerebrum.core.resolver import Resolver

logger = logging.getLogger(__name__)


class Service:
    """
    High-level application service coordinating embedding, persistence,
    and semantic search.

    This layer hides infrastructure details and exposes simple operations 
    for adding thoughts and querying them.
    """

    def __init__(self, thought_repository: ThoughtRepository, embedder: Embedder, semantic_store: SemanticStore, resolver: Resolver):
        """
        Initialize the service with its dependencies.

        Args:
            thought_repository (ThoughtRepository):
                Persistent storage for thoughts and index metadata.
            embedder (Embedder):
                Backend capable of converting text into embedding vectors.
            semantic_store (SemanticStore):
                Semantic store for nearest-neighbor search.
            resolver (Resolver):
                Component responsible for filtering and classifying raw search hits
                into a high-level SearchResult.
        """
        self._thought_repository = thought_repository
        self._embedder = embedder
        self._semantic_store = semantic_store
        self._resolver = resolver
    
    def add_thought(self, thought: Thought, index_id: str) -> int:
        """
        Insert a new thought into the system.

        Args:
            thought (Thought): Domain object containing the thought body and metadata.
            index_id (str): Identifier of the semantic index to attach the thought to.

        Returns:
            int: The assigned id64 for the new thought.
        """
        logger.info("AddThought: adding thought to index %s", index_id)
        embedding = self._embedder.embed(thought.body)
        id64 = self._thought_repository.insert_thought(thought, embedding, index_id)
        self._semantic_store.write(embedding.vector, [id64])
        self._thought_repository.complete_thought_insert(id64)
        logger.info("AddThought: completed insert id64=%s into index=%s", id64, index_id)
        return id64

    def query(self, query: str, index_id: str, k: int) -> SearchResult:
        """
        Perform a semantic search over the given index.

        Args:
            query (str): Raw text query to embed and search with.
            index_id (str): Identifier of the semantic index to search.
            k (int): Maximium number of nearest neighbors to retrieve.

        Returns:
            SearchResult: The structured outcome of the search.

        """
        logger.info("Query: starting on index=%s requested_k=%d", index_id, k)
        embedding = self._embedder.embed(query)
        try:
            similarities, ids = self._semantic_store.query(embedding.vector, k)
        except NoEmbeddingsError:
            return SearchResult(
                status=SearchStatus.NO_EMBEDDINGS, 
                hits=[]
            )

        thoughts = self._thought_repository.retrieve_thoughts(ids, index_id, ThoughtStatus.ACTIVE)
        search_hits = self._create_search_hits(thoughts, similarities, ids)
        result = self._resolver.resolve(search_hits)

        logger.info(
            "Query: finished index=%s status=%s raw_hits=%d filtered_hits=%d",
            index_id,
            result.status.name,
            len(search_hits),
            len(result.hits),
        )
        return result
    
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
        index_id = self._thought_repository.create_index(index_name, algorithm)
        logger.info(
            "CreateIndex: created index id=%s name=%s algorithm=%s",
            index_id,
            index_name,
            algorithm,
        )
        return index_id
    
    def get_indexes(self) -> list[Index]:
        """
        Return all known semantic indexes.

        Returns:
            list[Index]: Metadata for each index defined in the repository.
        """
        return self._thought_repository.list_indexes()
    
    def get_index_by_id(self, index_id: str) -> Index:
        """
        Return a semantic index based on the index_id.
        
        Raises:
            KeyError: If no index exists with the given ID.
        """
        indexes = self.get_indexes()
        for index in indexes:
            if index.index_id == index_id:
                return index

        raise KeyError(f"No index found with id: {index_id}")
