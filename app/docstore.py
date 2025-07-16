import sqlite3, pickle
from typing import List

class DocStore:
    def __init__(self, sqlite_path):
        self.sqlite_path = sqlite_path

    def load_parents(self, uids: List[str]) -> List:
        if not uids:
            return []
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        q_marks = ",".join("?" * len(uids))
        cur.execute(f"SELECT doc FROM parents WHERE uid IN ({q_marks})", uids)
        docs = [pickle.loads(r[0]) for r in cur.fetchall()]
        conn.close()
        return docs
