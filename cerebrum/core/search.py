from dataclasses import dataclass
from enum import StrEnum

from cerebrum.core.repository import ThoughtRecord


@dataclass(frozen=True, slots=True)
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


class SearchStatus(StrEnum):
    """
    Classification of a semantic search outcome.

    OK:
        At least one hit satisfied the resolver's criteria.
    NO_EMBEDDINGS:
        The index contains no embeddings and cannot be searched.
    NO_MATCHES:
        The search completed, but no hits met the resolver's criteria.
    """
    OK = "ok"
    NO_EMBEDDINGS = "no_embeddings"
    NO_MATCHES = "no_matches"


@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    Structured result of a semantic search.

    Attributes:
        status: A SearchStatus indicating how the search resolved.
        hits: The ordered list of SearchHit objects that passed resolver filtering. 
              This list is non-empty only when status == SearchStatus.OK.
    """
    status: SearchStatus
    hits: list[SearchHit]