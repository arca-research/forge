"""
Query engines: VectorQueryEngine // GraphQueryEngine
"""

from __future__ import annotations

# - util -
import numpy as np
from typing import Optional
from pathlib import Path

# - local -
from ..config import log, VectorDBConfig, GraphConfig
from .state import VectorIndex, MetaIndex
from .graph import Entity, Relationship
from .embed import Embedder
from .util import fetch_doc

from ._schemas import (
    ClaimData,
    RelationshipRecord,
    RelationshipCollisionError
)

from .state import GraphIndex



# === VECTOR QUERY ENGINE ===

class VectorQueryEngine:
    """Handles querying the vector database."""
    
    def __init__(self, cfg: VectorDBConfig):
        self.cfg = cfg
        self.embedder = Embedder(self.cfg.embed_model, self.cfg.batch_size)
        self.vector_index = VectorIndex(self.cfg, self.embedder.dim, rebuild=False)
        self.meta_index = MetaIndex(self.cfg)


    def run_query_engine(self):
        """
        TODO: depr.
        """
        top_k = 3
        res = self.query(query_text="sample query", k=top_k)

        if not res:
            log.info("No results to display.")
            return

        for i in range(len(res)):
            search_result = res[i]
            print(search_result)


    def query(self, query_text: str, k: int = 10, min_score: float = 0.0):
        """
        Search for similar text chunks.
        
        Args:
            query_text: Text to search for
            k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects (TODO)
        
        # TODO: make a single-chunk-per-doc mode (unique retrieval OR oversampling)
        """
        
        query_text = query_text[:800]
        query_vector = self.embedder.embed(query_text)
        
        # search
        distances, indices = self.search(query_vector=query_vector, k=k)
        
        # filter
        valid_results = [(dist, idx) for dist, idx in zip(distances, indices) if idx != -1]
        if not valid_results:
            log.warning("No valid search results found")
            return []
        
        embedding_ids, similarity_scores = [], []
        # convert distances to similarity scores
        for distance, embedding_id in valid_results:
            # distance == squared L2
            similarity = 1.0 - (distance / 2.0)
            
            if similarity >= min_score:
                embedding_ids.append(int(embedding_id))
                similarity_scores.append(float(similarity))
        
        if not embedding_ids:
            log.info("No results above minimum score threshold %s", min_score)
            return []
        
        # fetch metadata
        chunks = []
        for e_id in embedding_ids:
            c_id = self.meta_index.resolve(e_id)
            if c_id is None:
                log.warning("No chunk found for embedding_id: %s", e_id)
                continue
            doc_path, start, end = self.meta_index.get_chunk_metadata(c_id)
            chunk_text = fetch_doc(doc_path, start, end)

            chunk = {
                "embedding_id": e_id,
                "text": chunk_text,
            }
            chunks.append(chunk)

        if not chunks:
            log.warning("No chunk metadata found for embedding IDs")
            return []
        
        # create mapping of embedding_id to similarity score
        score_map = dict(zip(embedding_ids, similarity_scores))
        
        # build search results
        results = []
        for chunk in chunks:
            embedding_id = chunk['embedding_id']
            similarity_score = score_map.get(embedding_id, 0.0)
            result = {
                **chunk,
                "similarity_score": similarity_score
            }
            results.append(result)
        
        # sort by similarity score (descending)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:k]


    def search(self, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        Args:
            query_vector: Query vector of shape (dimension,)
            k: Number of neighbors to return
        Returns:
            Tuple of (distances, indices)
        """
        if self.vector_index.index is None:
            raise RuntimeError("Index not initialized")
        
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        n = self.vector_index.index.ntotal
        distances, indices = self.vector_index.index.search(query_vector, min(k, n))
        return distances[0], indices[0]


# === GRAPH QUERY ENGINE ===

class GraphQueryEngine:
    """Handles querying the graph database."""

    def __init__(self):
        self.graph_config = GraphConfig()
        self.index = GraphIndex(index_path=self.graph_config.graph_index_path)


    def Entity(self, name: str) -> Entity:
        """Create Entity bound to this engine's index"""
        return Entity(name, self.index)


    def Relationship(self,
        src: str, tgt: str,
        directed: bool = False,
        strength: Optional[float] = None
    ) -> Relationship:
        """Create Relationship bound to this engine's index"""
        return Relationship(src, tgt, strength=strength, state=self.index, directed=directed)


    def list_all_entities(self) -> list[str]:
        """List all canonical entity names"""
        return self.index.list_all_entities()


    def list_all_aliases(self, entity_name: Optional[str] = None):
        """List aliases"""
        return self.index.list_all_aliases(entity_name)


    def query(self, query: str):
        """
        TODO: not implemented
        """
        raise NotImplementedError()


    def find_similar_entities(self, name: str) -> list[str]:
        """
        | TODO: not implemented.
        """
        raise NotImplementedError("find_similar_entities requires vector integration")


    def find_similar_claims(self, content: str) -> list[ClaimData]:
        """
        | TODO: not implemented.
        """
        raise NotImplementedError("find_similar_claims requires vector integration")
    

