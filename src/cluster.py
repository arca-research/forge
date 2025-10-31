"""
Clustering primitives
"""

from __future__ import annotations

# - util -
import matplotlib.pyplot as plt
import warnings
import random
import textwrap

# - vectors -
import hdbscan
import umap
import numpy as np

# - local -
from ..config import VectorDBConfig, log
from .state import VectorIndex, MetaIndex
from .embed import Embedder
from .llm import SyncLLM
from .util import fetch_doc


# --- config ---

warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", message=".*TBB threading layer.*") # UMAP warning, not HDBSCAN.

random.seed(42)
np.random.seed(42)


# === CLUSTER ENGINE ===

class ClusterEngine:
    """Handles clustering of the vector database."""
    
    def __init__(self, cfg: VectorDBConfig):
        self.cfg = cfg
        self.embedder = Embedder(self.cfg.embed_model, self.cfg.batch_size)
        self.vector_index = VectorIndex(cfg, self.embedder.dim, rebuild=False)
        self.meta_index = MetaIndex(self.cfg)
        self.llm_model = SyncLLM()

        self.min_cluster_size = 7
        self.min_samples = 3
        self.min_documents_per_cluster = 3 # TODO: currently unused.
        self.epsilon = 0.2

        self._embeddings = None
        self._raw_labels = None
    
    
    def cluster(self) -> list[list[int]]:
        """
        generate dynamic clusters.

        return clusters as lists of chunk ids.
        """

        # fetch embeds
        embeddings = self.vector_index.index.reconstruct_n(0, self.vector_index.index_size)
    
        # apply HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_epsilon=self.epsilon
        )

        labels = clusterer.fit_predict(embeddings)

        self._embeddings = embeddings
        self._raw_labels = labels

        clusters = []
        for cluster_id in set(labels):
            if cluster_id != -1:
                embedding_ids = [i for i, label in enumerate(labels) if label == cluster_id]
                clusters.append(embedding_ids)

        return clusters
    

    def centroid(self, ids: list[int]) -> np.ndarray:
        return self.vector_index.mean(ids)


    def categorize(self, clusters: list[list[int]]) -> list[str]:
        """Natural-languge cluster categorization"""
        log.info("Categorizing clusters with LLM...")

        cluster_snippets = [
            [s[:300] for s in self.extract_chunks(cluster, sample_size=10)]
            for cluster in clusters
        ]

        total_chars = sum(len(snippet) for snippets in cluster_snippets for snippet in snippets)
        log.info("Total characters in all snippets: %s", total_chars)
        
        system_prompt = (
            "You are an expert at summarizing and categorizing document clusters. "
            "Given several text snippets, return a short category label (~10 words) "
            "that best describes the overall topic or theme of the cluster."
            "Name specific countries, regions or people if the context requires."
            "Try to be specific, avoid being overly vague."
        )

        self.llm_model.set_system(system_prompt)

        categories = []

        for snippets in cluster_snippets:
            if not snippets:
                categories.append("Unknown")  # fallback for empty cluster
                continue
            user_content = "Snippets:\n" + "\n".join(f"- {s}" for s in snippets)
            user_content += "\n\nCategory (5-10 words):"
            label = self.llm_model.run(prompt=user_content)

            categories.append(label)

        return categories


    def extract_chunks(self, cluster: list[int], sample_size: int = 5) -> list[str]:
        """
        Return metadata for clusters
        Args: cluster list of chunk ids
        """
        sample_size = min(sample_size, len(cluster))
        sampled_embedding_ids = random.sample(cluster, sample_size)
        
        chunk_snippets = []
        
        for e_id in sampled_embedding_ids:
            try:
                # resolve embedding_id to chunk_id
                c_id = self.meta_index.resolve(e_id)
                if c_id is None:
                    log.warning("No chunk found for embedding_id: %s", e_id)
                    continue
                    
                # get chunk metadata and fetch chunk text
                doc_path, start, end = self.meta_index.get_chunk_metadata(c_id)
                chunk_text = fetch_doc(doc_path, start, end)
                chunk_text = chunk_text.replace('\n', ' ').replace('\r', ' ')
                chunk_snippets.append(chunk_text)
                
            except Exception as e:
                log.warning("Error processing embedding_id %s: %s", e_id, e)
                continue
        
        return chunk_snippets


    def extract_docs(self, cluster: list[int]) -> tuple[list[int], list[str]]:
        """
        Return metadata for clusters
        Args: cluster list of chunk ids
        Returns: tuple of (doc_ids, doc_texts)
        """

        doc_paths = set()
        docs = []
        
        for e_id in cluster:
            try:
                # resolve embedding_id to chunk_id
                c_id = self.meta_index.resolve(e_id)
                if c_id is None:
                    log.warning("No chunk found for embedding_id: %s", e_id)
                    continue
                    
                # get chunk metadata and fetch document text
                doc_path, _, _ = self.meta_index.get_chunk_metadata(c_id)
                doc_paths.add(doc_path)
                
            except Exception as e:
                log.warning("Error processing embedding_id %s: %s", e_id, e)
                continue
        
        for doc_path in doc_paths:
            doc_text = fetch_doc(doc_path)
            docs.append(doc_text)
        
        return list(doc_paths), docs


    def visualize(self, categories: list[str] = None):
        """Visualize clusters with meaningful category names, or fallback to raw labels"""
        if not hasattr(self, '_embeddings') or not hasattr(self, '_raw_labels'):
            log.warning("No embeddings/labels stored for visualization")
            return
        
        # NOTE: remove random state for parallelism at the cost of reproducibility.
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(self._embeddings)

        plt.style.use('Solarize_Light2')
        plt.figure(figsize=(12, 8))
        
        unique_labels = set(self._raw_labels)
        
        for label in unique_labels:
            idx = self._raw_labels == label
            if label == -1:
                plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], 
                        s=10, label='Noise', alpha=0.7, c='gray')
            else:
                # Use category name if available, otherwise fall back to cluster number
                if categories and label < len(categories):
                    # label_name = categories[label]
                    label_name = '\n'.join(textwrap.wrap(categories[label], width=60))
                else:
                    label_name = f'Cluster {label}'
                    
                plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], 
                        s=10, label=label_name, alpha=0.7)
    
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('UMAP projection of embeddings')
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.tight_layout()
        plt.savefig("apps/daybreak/clusters.png", bbox_inches='tight')



# --- ENGINE CALLS ---

def run_clustering():
    cfg = VectorDBConfig()
    engine = ClusterEngine(cfg)
    clusters = engine.cluster()

    log.info("%s clusters formed.", len(clusters))

    mapping = {}
    categories = engine.categorize(clusters)
    engine.visualize(categories)

    for n, category in enumerate(categories):
        mapping[n] = category
        print(f"cluster_{n}: {category}")

    for i in range(len(clusters)):
        cluster = clusters[i]
        print(f"\n 📁 cluster_{i} ({mapping.get(i)}): {len(cluster)} chunks")
        print("=" * 100)
        snippets = engine.extract_chunks(cluster)
        for j, snippet in enumerate(snippets, 1):
            print(f"\033[36m{j}.\033[0m. {snippet[:100]}")
        if len(cluster) > 5:
            print(f"... and {len(cluster) - 5} more chunks")
        print()


if __name__ == "__main__":
    run_clustering()
