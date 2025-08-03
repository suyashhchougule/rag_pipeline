from pathlib import Path
from typing import List
import uuid
import re
from bs4 import BeautifulSoup
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema import Document
from filetracker import FileTracker

class Normalizer:
    @staticmethod
    def html_table_to_markdown(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return html
        grid, spans = [], {}
        for r, row in enumerate(table.find_all("tr")):
            line, c = [], 0
            for cell in row.find_all(["th", "td"]):
                # handle row/col spans
                while (r, c) in spans:
                    txt, rem, span_cols = spans.pop((r, c))
                    line.append(txt)
                    if rem > 1:
                        for sc in range(span_cols):
                            spans[(r+1, c+sc)] = (txt, rem-1, span_cols)
                    c += span_cols
                txt = " ".join(cell.get_text(strip=True).split())
                colspan = int(cell.get("colspan", 1))
                rowspan = int(cell.get("rowspan", 1))
                line.extend([txt] * colspan)
                if rowspan > 1:
                    for sc in range(colspan):
                        spans[(r+1, c+sc)] = (txt, rowspan-1, colspan)
                c += colspan
            grid.append(line)
        max_cols = max(len(r) for r in grid)
        for row in grid:
            row.extend([""] * (max_cols - len(row)))
        # build markdown table
        header = grid[0]
        sep = ["---"] * max_cols
        rows = grid[1:]
        lines = ["| " + " | ".join(header) + " |",
                 "| " + " | ".join(sep) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    @classmethod
    def normalise_tables(cls, text: str) -> str:
        return re.sub(r"<table[\s\S]+?</table>", lambda m: cls.html_table_to_markdown(m.group(0)), text, flags=re.IGNORECASE)

class Loader:
    """
    Discovers files in a folder, normalises and splits into sections.
    """
    def __init__(self, input_folder: str, tracker: FileTracker, patterns: List[str] = ["*.md"], rechunk: bool = False):
        self.input_folder = Path(input_folder)
        self.tracker = tracker
        self.patterns = patterns
        self.rechunk = rechunk

    def discover_files(self) -> List[Path]:
        files: List[Path] = []
        for pat in self.patterns:
            files.extend(self.input_folder.rglob(pat))
        return files

    def load(self) -> List[Document]:
        docs: List[Document] = []
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","H1"),("##","H2")], strip_headers=False)
        for file in self.discover_files():
            if not self.rechunk and not self.tracker.has_changed(file):
                continue
            raw = file.read_text(encoding="utf-8")
            md = Normalizer.normalise_tables(raw)
            sections = splitter.split_text(md)
            for sec in sections:
                sec.metadata["source"] = str(file)
                sec.metadata["uid"] = sec.metadata.get("uid", uuid.uuid4().hex)
                docs.append(sec)
        
        for i, doc in enumerate(docs, 1):
            print(f"\n──── Chunk {i} ▸ UID={doc.metadata['uid'][:8]} … ────\n")
            print(doc.page_content)
        return docs
