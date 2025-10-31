from dataclasses import dataclass
from typing import Optional

@dataclass
class ChunkData:
    """Container for chunk information."""
    document_id: int
    start_token: int
    end_token: int
    start_char: int
    end_char: int
    source_path: str
    embedding_id: Optional[int] = None
