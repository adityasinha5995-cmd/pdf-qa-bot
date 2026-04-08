import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from typing import List


class TFIDFEmbeddings(Embeddings):
    """Lightweight TF-IDF embeddings — no torch, no API needed."""

    def __init__(self, texts: List[str]):
        self.vectorizer = TfidfVectorizer(max_features=512)
        self.vectorizer.fit(texts)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.vectorizer.transform([text])
        return vector.toarray()[0].tolist()


def get_embedder(texts: List[str]) -> TFIDFEmbeddings:
    return TFIDFEmbeddings(texts)


def build_vectorstore(chunks):
    texts = [doc.page_content for doc in chunks]
    embedder = get_embedder(texts)
    vectorstore = FAISS.from_documents(chunks, embedder)
    return vectorstore


def get_retriever(vectorstore, k=4):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )