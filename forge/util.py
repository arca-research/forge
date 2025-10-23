"""
forge utilities
"""

from pathlib import Path

from ..config import log
from ._schemas import (ChunkData)


# --- doc-reader ---

def fetch_doc(filepath: Path | str, start_char: int | None = None, end_char: int | None = None) -> str:
    """Fetch doc"""
    if isinstance(filepath, str):
        filepath = Path(filepath)

    # read
    try:
        with filepath.open(encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        raise RuntimeError(f"cannot open {filepath}: {e}")

    # fetch chunk if asked
    if (start_char is not None and end_char is not None and start_char < end_char):
        start = max(0, min(start_char, len(text)))
        end = max(start, min(end_char, len(text)))
        return text[start:end]
    
    return text


# --- chunker ---

def chunk(
    filepath: str,
    document_id: int,
    embedder,  # Implements .encode(str)->list[int] & .decode(list[int])->str.
    max_tokens: int = 510,
    overlap: int = 510 * 0.10,
) -> tuple[list[ChunkData], list[list[int]]]:
    """
    Read a document file, chunk its text, and return (chunks_data, chunks_text).
    
    Args:
        filepath: Path to document file
        document_id: Document DB ID
        embedder: Embedder instance
        max_tokens: Max tokens per chunk
        overlap: Overlap in tokens
        
    Returns:
        Tuple of (chunks_data, chunks_text)
    """

    # fetch document text
    try:
        text = fetch_doc(filepath)
    except Exception as e:
        log.error("Failed to read %s: %s", filepath, e)
        return [], []
    
    if not text.strip():
        log.warning("Empty file: %s", filepath)
        return [], []

    encoding = embedder.encode(text) # just tokenizing.
    chunks_data, chunks_text = [], []

    # TODO: SENTENCIZER

    # chunking logic
    if len(encoding) <= max_tokens: # doc is single chunk
        chunk_text = text.strip()
        chunks_data.append(ChunkData(
            document_id=document_id,
            start_token=0,
            end_token=len(encoding),
            start_char=0,
            end_char=len(text),
            source_path=str(filepath)
        ))
        chunks_text.append(text)
    else: # make many chunks
        total_chars = len(text)
        total_tokens = len(encoding)
        step = max_tokens - overlap
        for start in range(0, total_tokens, step):
            end = min(start + max_tokens, total_tokens)
            token_slice = encoding[start:end]
            chunk_text = embedder.decode(token_slice)
            if not chunk_text:
                log.warning("Empty chunk within %s: %s...", filepath, text[:30])
                continue
            start_char = int((start / total_tokens) * total_chars)
            end_char = int((end / total_tokens) * total_chars)
            start_char = max(0, min(start_char, total_chars))
            end_char = max(start_char, min(end_char, total_chars))
            chunks_data.append(ChunkData(
                document_id=document_id,
                start_token=start,
                end_token=end,
                start_char=start_char,
                end_char=end_char,
                source_path=str(filepath)
            ))
            chunks_text.append(chunk_text)

    if not chunks_text:
        log.warning("No chunks generated for %s", filepath)
        return [], []
    
    return chunks_data, chunks_text


# ---

def pricing(words: str, input_price_per_M: float, output_price_per_M: float):
    """TODO: token pricing.
    Print out the price
    """
    return # STUB