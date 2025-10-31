from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import shutil

import numpy as np

from ...config import log


def debug_only(func):
    """Marks a function as debug/internal use only"""
    func.__debug_only__ = True
    return func


class ClusterIndex:
    """Cluster state handler"""

    def __init__(self, stage_dir):
        self.stage_dir = stage_dir
        self.index_path = self.stage_dir / "clusters.sqlite"
        self.centroids_path = self.stage_dir / "centroids"
        self.artifacts_path = self.stage_dir / "artifacts"
        self.sim_threshold = 0.9

        

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
        self.centroids_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

        with self._conn() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS cluster (
                    id              INTEGER PRIMARY KEY,
                    status          INTEGER NOT NULL CHECK (status IN (0,1)),
                    centroid_path   TEXT NOT NULL,
                    artifact_path   TEXT
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS cluster_docs (
                    cluster_id      INTEGER NOT NULL,
                    doc_id          INTEGER NOT NULL,
                    PRIMARY KEY (cluster_id, doc_id),
                    FOREIGN KEY (cluster_id) REFERENCES cluster(id) ON DELETE CASCADE
                );
            """)


    def batch(self, new: list[np.ndarray], doc_ids: list[list[int]]) -> dict[str, int]:
        """
        Insert/match a generation of clusters. Returns stats:
        born, died, active, dormant, total

        Args:
            new_centroids: list of centroid vectors.
            doc_ids: list of lists of document IDs, aligned with centroids.
        """
        existing = self._fetch_all()  # dict: cluster_id -> (centroid ndarray, path)
        matched_existing: set[int] = set()

        born = 0
        died = 0
        revived = 0 # 0->1 flips

        # TODO (optimal assignment): compute similarity matrix once; decide greedy vs optimal.
        for centroid, docs in zip(new, doc_ids):
            cluster_id = self._match(centroid, existing)
            if cluster_id is not None:
                self._replace_centroid(cluster_id, centroid)
            else:
                cluster_id = self.upsert(1, None)
                self._upload_centroid(cluster_id, centroid)
                born += 1

            self._update_docs(cluster_id, docs)
            matched_existing.add(cluster_id)

        # reactivate matched clusters that were dormant (status: 0 -> 1)
        if matched_existing:
            with self._conn() as con:
                before = con.total_changes
                con.executemany(
                    "UPDATE cluster SET status=1 WHERE status=0 AND id=?",
                    [(cid,) for cid in matched_existing]
                )
                revived = con.total_changes - before

        # mark unmatched (existing) clusters as dormant (status=0)
        dormant = set(existing.keys()) - matched_existing
        if dormant:
            with self._conn() as con:
                before = con.total_changes
                con.executemany(
                    "UPDATE cluster SET status=0 WHERE status=1 AND id=?",
                    [(cid,) for cid in dormant]
                )
                died = con.total_changes - before # only counts 1->0 flips

        base_stats = self._stats()
        stats = {
            "born": born,
            "died": died,
            "revived": revived,
            **base_stats,
        }
        return stats


    def upsert(self, status: int, artifact_path: str | None) -> int:
        """insert or update a cluster record."""

        with self._conn() as con:
            cur = con.execute("""
                INSERT INTO cluster (status, centroid_path, artifact_path)
                VALUES (?, ?, ?)
            """, (status, "", artifact_path))
            return cur.lastrowid
        

    def _update_docs(self, cluster_id: int, docs: list[int]) -> None:
        """update membership doc list for a cluster, ignoring duplicates."""
        with self._conn() as con:
            con.executemany(
                "INSERT OR IGNORE INTO cluster_docs (cluster_id, doc_id) VALUES (?, ?)",
                [(cluster_id, d) for d in docs]
            )

    
    def _match(self, centroid: np.ndarray, existing: dict[int, tuple[np.ndarray, str]]) -> int | None:
        """
        Return matching cluster_id if similarity >= threshold, else None.
        
        Time complexity: O(N) (N is multiplied by M matching ops)
        """
        if not existing:
            return None

        # compute cosine similarities against all existing centroids
        sims = {
            cid: float(np.dot(centroid, ex) / (np.linalg.norm(centroid) * np.linalg.norm(ex)))
            for cid, (ex, _) in existing.items()
        }

        # pick best match above threshold
        best_id, best_sim = max(sims.items(), key=lambda kv: kv[1])
        if best_sim >= self.sim_threshold:
            return best_id
        return None

    
    def _upload_centroid(self, cluster_id: int, centroid: np.ndarray) -> str:
        """uploads to new blob storage. returns path for upserting."""
        path = self.centroids_path / f"{cluster_id}.npy"
        np.save(path, centroid)

        with self._conn() as con:
            con.execute("UPDATE cluster SET centroid_path=? WHERE id=?",
                        (str(path), cluster_id))

        return str(path)
    

    def _replace_centroid(self, cluster_id: int, centroid: np.ndarray) -> str:
        """overwrite centroid file for an existing cluster."""
        with self._conn() as con:
            cur = con.execute(
                "SELECT centroid_path FROM cluster WHERE id=?",
                (cluster_id,)
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Cluster {cluster_id} does not exist")
            path = Path(row["centroid_path"])

        # overwrite on disk
        np.save(path, centroid)
        return str(path)
    

    def _fetch_one(self, centroid_path: str) -> np.ndarray:
        """fetch centroid from blob storage.
        NOTE: this will be used for resolve()
        """
        return np.load(centroid_path)


    def _fetch_all(self) -> dict[int, tuple[np.ndarray, str]]:
        """fetch ALL centroids from blob storage (for ease of matching)
        NOTE 1: there will likely only ever be hundreds of cluster records, so I/O is not bottlenecked.
        NOTE 2: might be quite valuable to have a single "all clusters" file, so we just need a single I/O.
        """
        centroids: dict[int, tuple[np.ndarray, str]] = {}
        with self._conn() as con:
            for row in con.execute("SELECT id, centroid_path FROM cluster"):
                path = Path(row["centroid_path"])
                if path.exists():
                    vec = np.load(path)
                    centroids[row["id"]] = (vec, str(path))
        return centroids


    def resolve(self, id: int):
        """return a cluster record.
        haven't decided what this entails (whether centroid or doc path).
        leave as stub for now.
        we'll probably use self._fetch_one here."""
        return


    def upload_artifact(self):
        """stub. TODO much later."""
        return


    def _stats(self) -> dict[str, int]:
        """Compute cluster statistics."""
        with self._conn() as con:
            total = con.execute("SELECT COUNT(*) FROM cluster").fetchone()[0]
            active = con.execute("SELECT COUNT(*) FROM cluster WHERE status=1").fetchone()[0]
            dormant = con.execute("SELECT COUNT(*) FROM cluster WHERE status=0").fetchone()[0]
        return {
            "total": total,
            "active": active,
            "dormant": dormant,
        }


    def drop(self):
        """drop all values from the cluster table."""
        with self._conn() as con:
            con.execute("DELETE FROM cluster;")
            con.execute("DELETE FROM cluster_docs;")
        
        for path in [self.centroids_path, self.artifacts_path]:
            if path.exists():
                shutil.rmtree(path)
