# 📚 PDF-RAG: Retrieval-Augmented Generation for PDF Question Answering

An end-to-end **Retrieval-Augmented Generation (RAG)** system that enables accurate, context-aware question answering over PDF documents. Built with **LangChain**, **ChromaDB**, **Sentence Transformers**, and **Groq LLM**, this project implements a modular and scalable pipeline covering PDF ingestion, chunking, embedding generation, vector storage, semantic retrieval, and LLM-based answer generation.

---

## 🚀 Features

- **Automated PDF Ingestion** — Load single or multiple PDFs and extract clean, structured text.
- **Smart Chunking** — Splits documents into overlapping, context-preserving chunks for better retrieval accuracy.
- **384-Dimensional Embeddings** — Uses Sentence Transformers (`all-MiniLM-L6-v2`) to convert text chunks into dense vector representations.
- **Vector Search with ChromaDB** — Stores embeddings in a persistent vector database for fast, semantic similarity search.
- **Context-Aware Retrieval** — Retrieves the most relevant chunks based on query similarity, not just keyword matching.
- **Groq LLM Integration** — Generates fast, accurate, and grounded answers using Groq's low-latency inference engine.
- **Modular Pipeline Design** — Each stage (ingestion, chunking, embedding, retrieval, generation) is independently testable and swappable.
- **Scalable Architecture** — Easily extendable to support multiple document types, larger corpora, and different LLM/embedding providers.

---

## 🏗️ Architecture

```
                ┌────────────────┐
                │   PDF Files    │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │  PDF Ingestion │  (PyPDFLoader / LangChain loaders)
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │ Text Chunking  │  (RecursiveCharacterTextSplitter)
                └───────┬────────┘
                        │
                ┌───────▼────────────┐
                │ Embedding Manager  │  (Sentence Transformers, 384-dim)
                └───────┬────────────┘
                        │
                ┌───────▼────────┐
                │   ChromaDB     │  (Persistent Vector Store)
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │ Semantic Query │  (Top-k Similarity Search)
                │   Retrieval    │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │   Groq LLM     │  (Context + Query → Answer)
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │  Final Answer  │
                └────────────────┘
```

---

## 🧰 Tech Stack

| Component            | Technology                                  |
|-----------------------|----------------------------------------------|
| Orchestration          | LangChain                                    |
| Vector Database        | ChromaDB                                     |
| Embedding Model        | Sentence Transformers (`all-MiniLM-L6-v2`)   |
| LLM Provider           | Groq (e.g., `llama-3.1-70b`, `mixtral-8x7b`) |
| PDF Parsing            | PyPDFLoader / pypdf                          |
| Language               | Python 3.10+                                 |

---

## 📂 Project Structure

```
pdf-rag/
│
├── data/
│   └── pdfs/                     # Source PDF documents
│
├── chroma_db/                    # Persistent vector store (auto-generated)
│
├── src/
│   ├── ingestion.py               # PDF loading and text extraction
│   ├── chunking.py                # Text splitting logic
│   ├── embedding_manager.py       # Embedding generation & management
│   ├── vector_store.py            # ChromaDB storage & retrieval
│   ├── retriever.py                # Semantic search / context fetching
│   ├── llm_chain.py                # Groq LLM prompt & response generation
│   └── pipeline.py                 # End-to-end orchestration
│
├── app.py                         # Entry point / CLI or UI interface
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/pdf-rag.git
cd pdf-rag
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
CHROMA_DB_DIR=./chroma_db
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

---

## 📦 requirements.txt

```
langchain
langchain-community
langchain-groq
chromadb
sentence-transformers
pypdf
python-dotenv
tiktoken
```

---

## 🧠 Embedding Manager

The `EmbeddingManager` class handles all embedding-related operations, decoupling the embedding logic from the vector store and retrieval components.

```python
# src/embedding_manager.py

from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingManager:
    """
    Manages text-to-vector embedding generation using Sentence Transformers.
    Produces 384-dimensional dense embeddings for semantic search.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of document chunks."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query string."""
        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )
        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
```

---

## ✂️ Chunking Strategy

```python
# src/chunking.py

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size=1000, chunk_overlap=150):
    """
    Splits documents into overlapping chunks to preserve context
    across chunk boundaries, improving retrieval quality.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)
```

---

## 🗄️ Vector Store (ChromaDB)

```python
# src/vector_store.py

import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, persist_directory="./chroma_db", collection_name="pdf_rag"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, ids, embeddings, documents, metadatas):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, query_embedding, top_k=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
```

---

## 🤖 Groq LLM Chain

```python
# src/llm_chain.py

import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
You are an assistant answering questions strictly based on the provided context.
If the answer is not found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""


def get_llm_response(context: str, question: str, model_name="llama-3.1-70b-versatile"):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=model_name,
        temperature=0.2
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content
```

---

## ▶️ Usage

### Ingest PDFs and build the vector store
```bash
python app.py --ingest --pdf_dir ./data/pdfs
```

### Ask a question
```bash
python app.py --query "What are the key findings in the report?"
```

### Example (Python)
```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.ingest("./data/pdfs")

answer = pipeline.ask("Summarize the main conclusions of the document.")
print(answer)
```

---

## 📊 Example Output

```
Query: What is the main objective of the study?

Retrieved Context (Top 3 chunks):
1. "The primary objective of this research is to evaluate..."
2. "This study aims to investigate the impact of..."
3. "Our goal is to develop a framework that..."

Answer:
The main objective of the study is to evaluate the effectiveness of the 
proposed framework in improving retrieval accuracy for domain-specific 
question answering tasks.
```

---

## 🔮 Future Improvements

- Support for multi-format documents (DOCX, TXT, HTML)
- Hybrid search (BM25 + semantic embeddings)
- Re-ranking with cross-encoders
- Streaming responses via Groq
- Web UI (Streamlit / Gradio) for interactive Q&A
- Conversation memory for multi-turn dialogue

---


