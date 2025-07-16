import hashlib
import sqlite3
from pathlib import Path

class FileTracker:
    """
    Tracks ingested files and skips files that haven't changed.
    Uses a SQLite table to store file path and SHA-256 hash.
    """
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_table()

    def _init_table(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingested_files (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def has_changed(self, path: Path) -> bool:
        h = self._compute_hash(path)
        cur = self.conn.cursor()
        cur.execute("SELECT hash FROM ingested_files WHERE path = ?", (str(path),))
        row = cur.fetchone()
        if row and row[0] == h:
            return False
        # insert or update
        cur.execute(
            "INSERT INTO ingested_files(path, hash) VALUES (?, ?) ON CONFLICT(path) DO UPDATE SET hash=excluded.hash, ingested_at=CURRENT_TIMESTAMP",
            (str(path), h)
        )
        self.conn.commit()
        return True

    def _compute_hash(self, path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()

    def close(self):
        self.conn.close()