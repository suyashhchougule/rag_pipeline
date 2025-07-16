import sqlite3
from typing import List
import pickle
from langchain.schema import Document

class SQLiteBlobStore:
    """
    Stores parent chunks in SQLite as pickled blobs.
    """
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS parents(uid TEXT PRIMARY KEY, doc BLOB)")
        self.conn.commit()

    def save_parents(self, parents: List[Document]):
        data = [(p.metadata["uid"], pickle.dumps(p)) for p in parents]
        cur = self.conn.cursor()
        cur.executemany("INSERT OR REPLACE INTO parents VALUES (?,?)", data)
        self.conn.commit()

    def load_parents(self, uids: List[str]) -> List[Document]:
        if not uids:
            return []
        placeholders = ",".join("?" * len(uids))
        cur = self.conn.cursor()
        cur.execute(f"SELECT doc FROM parents WHERE uid IN ({placeholders})", uids)
        return [pickle.loads(r[0]) for r in cur.fetchall()]