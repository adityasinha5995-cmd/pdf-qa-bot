from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embedder = None


def get_embedder():
    """Load embedding model once and reuse (cached)."""
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embedder


def build_vectorstore(chunks):
    """Embed all chunks and store in FAISS in memory."""
    embedder = get_embedder()
    vectorstore = FAISS.from_documents(chunks, embedder)
    return vectorstore


def get_retriever(vectorstore, k=4):
    """Return a retriever that fetches top-k similar chunks."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
