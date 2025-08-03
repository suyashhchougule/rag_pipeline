from typing import List, Tuple
import pickle
import uuid
import nltk
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
nltk.download("punkt_tab", quiet=True)

class Chunker:
    """
    Splits sections into parent chunks and sentence chunks.
    """
    def __init__(self, parent_size: int = 2048, parent_overlap: int = 0):
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=parent_overlap)

    def chunk(self, sections: List[Document]) -> Tuple[List[Document], List[Document]]:
        parents, sentences = [], []
        for sec in sections:
            for chunk in self.parent_splitter.split_text(sec.page_content):
                p_uid = uuid.uuid4().hex
                parents.append(Document(page_content=chunk, metadata={"uid":p_uid, "source":sec.metadata["source"]}))
                for sent in nltk.tokenize.sent_tokenize(chunk):
                    sentences.append(Document(page_content=sent, metadata={"uid":uuid.uuid4().hex, "parent_id":p_uid}))
        return parents, sentences