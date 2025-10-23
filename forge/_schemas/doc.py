from typing import Protocol, Optional
from dataclasses import dataclass
from pathlib import Path

class DocLike(Protocol):
    """Container template for document information."""
    document_id: int
    filepath: Path | str
    date: Optional[str]
    source: Optional[str]
    domain: Optional[str]
    context: Optional[str]


@dataclass
class Doc:
    document_id: int
    filepath: Path | str
    date: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None
    context: Optional[str] = None