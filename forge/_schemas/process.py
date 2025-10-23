from dataclasses import dataclass

@dataclass 
class ProcessingStats:
    """Statistics for processing session."""
    documents_processed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    errors: int = 0