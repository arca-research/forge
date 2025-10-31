from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3

from ...config import log
from .._schemas import (
    ChunkData
)


def debug_only(func):
    """Marks a function as debug/internal use only"""
    func.__debug_only__ = True
    return func


class MetaIndex:
    """SQL Meta Index handler"""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self._initialize()


    @contextmanager
    def _conn(self):
        """
        Helper to open a SQLite connection with row access by column name.
        """
        con = sqlite3.connect(self.index_path)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA foreign_keys=ON;")
        con.execute("PRAGMA busy_timeout=5000;")
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()


    def _initialize(self) -> None:
        with self._conn() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS chunk (
                    id              INTEGER PRIMARY KEY,
                    document_id     INTEGER NOT NULL,
                    embedding_id    INTEGER,
                    start_token     INTEGER NOT NULL,
                    end_token       INTEGER NOT NULL,
                    start_char      INTEGER NOT NULL,
                    end_char        INTEGER NOT NULL,
                    source_path     TEXT NOT NULL,
                    UNIQUE (document_id, start_token, end_token)
                )
            """)


    def upsert(self, chunks: list[ChunkData]):
        """upsert a chunk batch into meta."""
        with self._conn() as con:
            con.executemany("""
                INSERT INTO chunk (
                    document_id, embedding_id, start_token, end_token, start_char, end_char, source_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id, start_token, end_token)
                DO UPDATE SET
                    embedding_id = excluded.embedding_id,
                    start_char = excluded.start_char,
                    end_char = excluded.end_char,
                    source_path = excluded.source_path
            """, [(c.document_id, c.embedding_id,
                  c.start_token, c.end_token,
                  c.start_char, c.end_char,
                  c.source_path,) for c in chunks])


    def drop(self):
        """drop all values from the chunk table."""
        with self._conn() as con:
            con.execute("DELETE FROM chunk;")


    def resolve(self, embedding_id: int) -> int:
        """
        Return chunk.id for a given embedding_id.
        Raises KeyError if not found.
        """
        with self._conn() as con:
            row = con.execute(
                "SELECT id FROM chunk WHERE embedding_id = ? LIMIT 1",
                (embedding_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"embedding_id {embedding_id} not found in chunk table")
        return row["id"]


    def get_chunk_metadata(
        self,
        chunk_id: int,
    ) -> tuple[str, int, int]:
        """
        Return (source_path, start_char, end_char) for a given chunk id.
        """
        with self._conn() as con:
            row = con.execute(
                "SELECT source_path, start_char, end_char "
                "FROM chunk WHERE id = ?",
                (chunk_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"chunk {chunk_id} not found")
        return row["source_path"], row["start_char"], row["end_char"]
    
    
    def has_chunks(self, document_id: int) -> bool:
        """
        Return True if any chunks exist for the given document_id, else False.
        """
        with self._conn() as con:
            row = con.execute(
                "SELECT 1 FROM chunk WHERE document_id = ? LIMIT 1",
                (document_id,)
            ).fetchone()
        return row is not None
