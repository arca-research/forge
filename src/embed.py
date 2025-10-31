"""
Embedder class

TODO: add support for cloud models (later version) & better async
"""

from __future__ import annotations

import numpy as np

from ..config import log


# === EMBEDDER ===

class Embedder:
    def __init__(self, embed_model: str, batch_size: int):
        from sentence_transformers import SentenceTransformer # NOTE: this takes ~1.5s, which is why it's here
        from transformers import AutoTokenizer, logging as hf_logging

        self.model = SentenceTransformer(embed_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.batch_size = batch_size
        hf_logging.set_verbosity_error()
        log.info("Initialized embedder with dimension %s", self.dim)


    def embed(self, texts: list[str]) -> np.ndarray:
        """
        texts: list of strings
        returns: np.ndarray of shape (len(batch), embed_dim), L2-normalized
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=self.batch_size
        )

        return embeddings.astype(np.float32)


    def encode(self, text) -> list[int]:
        return self.tokenizer.encode(text, truncation=False, add_special_tokens=False)


    def decode(self, encoding: list[int]) -> str:
        return self.tokenizer.decode(encoding, skip_special_tokens=True).strip()


if __name__ == "__main__":
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2", 32)
    text = "Hello World!"
    print(f"text: {text}")
    encoding = embedder.encode(text)
    print(f"encoding: {encoding}")
    decoded = embedder.decode(encoding)
    print(f"decoded: {decoded}")
