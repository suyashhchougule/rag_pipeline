from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_indexes(sentence_index_dir, parent_index_dir, embed_model_name):
    embed = HuggingFaceEmbeddings(model_name=embed_model_name)
    s_index = FAISS.load_local(sentence_index_dir, embed, allow_dangerous_deserialization=True)
    p_index = FAISS.load_local(parent_index_dir, embed, allow_dangerous_deserialization=True)
    return s_index, p_index
