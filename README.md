# PDF Q&A Bot

A fully local, free RAG (Retrieval-Augmented Generation) pipeline that lets you upload any PDF and ask questions about it in a chat interface.

No API keys. No cloud. Everything runs on your machine.

---

## Demo

Upload a PDF → ask a question → get an answer with source chunks shown.

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| PDF Loading | PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS (in-memory) |
| LLM | Ollama (Mistral / Llama3 / Phi3) |

---

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed

### 1. Install Ollama and pull a model

```bash
# Install from https://ollama.com
ollama pull mistral
```

### 2. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/pdf-qa-bot.git
cd pdf-qa-bot
```

### 3. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How It Works

```
PDF Upload
    │
    ▼
PyPDFLoader → pages
    │
    ▼
RecursiveCharacterTextSplitter → chunks (500 chars, 50 overlap)
    │
    ▼
HuggingFace Embeddings (all-MiniLM-L6-v2) → vectors
    │
    ▼
FAISS Vector Store (in memory)
    │
    ▼
User question → embed → similarity search → top-k chunks
    │
    ▼
Prompt: chunks + question → Ollama LLM
    │
    ▼
Answer + source chunks shown in chat UI
```

---

## Project Structure

```
pdf-qa-bot/
├── app.py              # Streamlit UI + chat interface
├── rag/
│   ├── __init__.py
│   ├── loader.py       # PDF loading and chunking
│   ├── embedder.py     # Embeddings + FAISS vector store
│   └── chain.py        # Prompt template + LangChain QA chain
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Features

- Upload any PDF (research papers, contracts, manuals, notes)
- Persistent chat history within the session
- Source chunk viewer — see exactly which part of the PDF was used
- Model selector — switch between Mistral, Llama3, Phi3, Gemma
- Adjustable top-k retrieval
- Handles re-upload and re-indexing automatically

---

## Skills Demonstrated

- RAG pipeline implementation end-to-end
- Vector database design and retrieval (FAISS)
- LLM integration with prompt engineering
- Unstructured data processing (PDF → chunks → embeddings)
- LangChain orchestration
- Local LLM deployment with Ollama
- Streamlit UI with session state management

---

## License

MIT
