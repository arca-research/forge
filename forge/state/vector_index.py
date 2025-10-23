from __future__ import annotations

import faiss
import numpy as np

from ...config import log

def debug_only(func):
    """Marks a function as debug/internal use only"""
    func.__debug_only__ = True
    return func


class VectorIndex:
    """Manages FAISS vector index."""
    
    def __init__(self, cfg, dimension: int, rebuild: bool = False):
        self.cfg = cfg
        self.dimension = dimension

        self.index_type = cfg.index_type
        self.index_path = self.cfg.stage_dir / f"{self.index_type}.faiss" # | FIXME

        self.index_size = 0
        self.index = None
        
        if rebuild:
            self._initialize()
        else:
            self._load()


    def _initialize(self) -> None:
        """Initialize a new HNSW index."""
        log.info("Initializing %s index with dimension %s", self.index_type, self.dimension)
        
        if self.index_type == "hnsw":
            # Create HNSW index
            self.index = faiss.IndexHNSWFlat(self.dimension, self.cfg.hnsw_m)
            self.index.hnsw.efConstruction = self.cfg.hnsw_ef_construction
            self.index.hnsw.efSearch = self.cfg.hnsw_ef_search
            self.index_size = 0
        elif self.index_type == "flat":
            pass # stub TODO
        elif self.index_type == "ivf":
            pass # stub TODO


    def _load(self):
        """Create index or load existing index from disk."""
        if self.index_path.exists():   
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.index_size = self.index.ntotal
                log.info("Loaded existing index with %s vectors", self.index_size)
                return self.index
            except Exception as e:
                log.error("Failed to load index: %s", e)
                raise
        else:
            self._initialize()
    
    
    def save(self):
        """Save index to disk."""
        if self.index is None:
            log.warning("No index to save. Please call load().")
            return
            
        try:
            faiss.write_index(self.index, str(self.index_path))
            log.info("Saved index with %s vectors to %s", self.index.ntotal, self.index_path)
        except Exception as e:
            log.error("Failed to save index: %s", e)
            raise


    def add_vectors(self, embeddings: np.ndarray) -> list[int]:
        """
        Add vectors to index and return their IDs.
        Args:
            embeddings: np.ndarray of shape (n, dimension)
        Returns:
            List of vector IDs assigned
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Please call load().")
        if embeddings.size == 0:
            return []
        
        if len(embeddings.shape) == 1: # ensure correct shape
            embeddings = embeddings.reshape(1, -1)
            
        start_id = self.index_size
        self.index.add(embeddings)
        self.index_size = self.index.ntotal
        
        # return list of assigned IDs
        return list(range(start_id, self.index_size))


    def mean(self, ids: list[int]) -> np.ndarray:
        """
        Compute the centroid vector for the given embedding IDs.
        
        Args:
            ids: List of vector IDs to average.
        
        Returns:
            np.ndarray of shape (dimension,) representing the centroid.
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Please call load().")
        if not ids:
            raise ValueError("No IDs provided.")

        # reconstruct each vector from the index
        vectors = [self.index.reconstruct(idx) for idx in ids]

        # convert to numpy array and compute mean
        vectors = np.array(vectors, dtype=np.float32)
        centroid = np.mean(vectors, axis=0)
        return centroid


    def size(self) -> int:
        """Get index size"""
        return self.index_size
