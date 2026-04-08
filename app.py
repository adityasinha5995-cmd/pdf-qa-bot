import streamlit as st
from rag.loader import load_and_chunk
from rag.embedder import build_vectorstore, get_retriever
from rag.chain import build_chain

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📄",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .source-box {
        background: #f8f9fa;
        border-left: 3px solid #6c63ff;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.85em;
        margin-bottom: 8px;
    }
    .chunk-label {
        font-weight: 600;
        color: #6c63ff;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📄 PDF Q&A Bot")
st.caption("Upload any PDF and ask questions about it. Runs fully locally — no API key needed.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    model_choice = st.selectbox(
        "Ollama model",
        ["mistral", "llama3", "llama3.2", "phi3", "gemma"],
        index=0,
        help="Make sure the model is pulled via: ollama pull <model>"
    )

    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=2, max_value=8, value=4,
        help="How many text chunks to feed the LLM per question"
    )

    show_sources = st.toggle("Show source chunks", value=True)

    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
1. PDF split into 500-char chunks
2. Chunks embedded with `all-MiniLM-L6-v2`
3. Stored in FAISS (in memory)
4. Your question matched to top chunks
5. Chunks + question sent to local LLM
6. Answer returned in chat
    """)

    st.divider()
    st.markdown("**Setup**")
    st.code("ollama pull mistral\nstreamlit run app.py", language="bash")

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a PDF to get started",
    type="pdf",
    help="Max ~50 pages works best. Larger files take longer to index."
)

if uploaded_file:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}_{model_choice}_{top_k}"

    if st.session_state.get("file_key") != file_key:
        with st.status("Indexing your PDF...", expanded=True) as status:
            st.write("Loading and chunking PDF...")
            chunks = load_and_chunk(uploaded_file)
            st.write(f"Created {len(chunks)} chunks.")

            st.write("Building embeddings and FAISS index...")
            vectorstore = build_vectorstore(chunks)
            retriever = get_retriever(vectorstore, k=top_k)
            st.write("Embeddings ready.")

            st.write(f"Loading {model_choice} via Ollama...")
            chain = build_chain(retriever, model_name=model_choice)
            st.write("Chain ready.")

            status.update(label=f"Ready — {len(chunks)} chunks indexed.", state="complete")

        st.session_state["chain"] = chain
        st.session_state["file_key"] = file_key
        st.session_state["messages"] = []
        st.session_state["num_chunks"] = len(chunks)

    st.success(
        f"**{uploaded_file.name}** indexed — "
        f"{st.session_state['num_chunks']} chunks · model: {model_choice}"
    )

# ── Chat Interface ────────────────────────────────────────────────────────────
if "chain" in st.session_state:
    st.divider()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                with st.expander("Source chunks"):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(
                            f'<div class="source-box">'
                            f'<div class="chunk-label">Chunk {i+1} · Page {src["page"]}</div>'
                            f'{src["text"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # Chat input
    question = st.chat_input("Ask something about the PDF...")

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {model_choice}..."):
                result = st.session_state["chain"]({"query": question})
                answer = result["result"]
                source_docs = result.get("source_documents", [])

            st.write(answer)

            sources = [
                {
                    "page": doc.metadata.get("page", "?") + 1,
                    "text": doc.page_content
                }
                for doc in source_docs
            ]

            if show_sources and sources:
                with st.expander("Source chunks"):
                    for i, src in enumerate(sources):
                        st.markdown(
                            f'<div class="source-box">'
                            f'<div class="chunk-label">Chunk {i+1} · Page {src["page"]}</div>'
                            f'{src["text"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

    # Clear chat button
    if st.session_state["messages"]:
        if st.button("Clear chat", type="secondary"):
            st.session_state["messages"] = []
            st.rerun()

else:
    st.info("Upload a PDF above to start chatting with it.")
