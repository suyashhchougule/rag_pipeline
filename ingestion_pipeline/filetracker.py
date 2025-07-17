import hashlib
import sqlite3
from pathlib import Path
import logging
log = logging.getLogger(__name__)

class FileTracker:
    """
    Tracks ingested files and skips files that haven't changed.
    Uses a SQLite table to store file path and SHA-256 hash.
    """
    def __init__(self, db_path: str):
        parent_dir = Path(db_path).parent
        if parent_dir.exists():
            log.info(f"Directory {parent_dir} exists for tracker DB.")
        else:
            log.info(f"Directory {parent_dir} does not exist. Creating now...")
        parent_dir.mkdir(parents=True, exist_ok=True)
        if parent_dir.exists():
            log.info(f"Verified directory exists for DB: {parent_dir}")

        self.conn = sqlite3.connect(db_path)
        self._init_table()
        log.info(f"Initialized FileTracker with DB: {db_path}")

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
        log.debug(f"Computed SHA256 for {path}")
        return sha.hexdigest()

    def close(self):
        self.conn.close()
        log.info("Closed FileTracker DB connection.")