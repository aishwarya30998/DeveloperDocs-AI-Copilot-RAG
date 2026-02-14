# 🤖 Developer Docs Copilot

> Production-grade RAG system that answers questions using official techstack documentation (eg:fastapi)

[![Deployed on HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

## 🎯 What This Project Demonstrates

This is a **production-style RAG (Retrieval-Augmented Generation)** system that showcases:

- ✅ **Professional documentation ingestion pipeline** with chunking strategies
- ✅ **Semantic search** using vector embeddings (ChromaDB)
- ✅ **Source attribution** with clickable citations
- ✅ **RAG evaluation metrics** (RAGAS framework)
- ✅ **Dockerized deployment** ready for cloud platforms
- ✅ **Production-grade error handling** and logging

## 🏗️ Architecture

```
┌─────────────┐
│   User      │
│  Question   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  1. Query Embedding                 │
│     (sentence-transformers)         │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  2. Vector Search (ChromaDB)        │
│     - Top 5 relevant chunks         │
│     - Metadata: source, section     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  3. Context Assembly                │
│     - Format chunks                 │
│     - Add instructions              │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  4. LLM Generation (HF Inference)   │
│     - Answer with citations         │
│     - Code examples preserved       │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  5. Response + Source Links         │
└─────────────────────────────────────┘
```

### Local Setup

```bash
# Clone the repository
git clone https://github.com/aishwarya30998/DeveloperDocs-AI-Copilot-RAG.git
cd DeveloperDocs-AI-Copilot-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


# create .env and add your HF_TOKEN


# Run the application
python app.py
```

Visit `http://localhost:7860` in your browser.

## 📦 Project Structure

```
fastapi-docs-copilot/
├── app.py                      # Gradio UI application
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Local container orchestration
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── chunking.py            # Document chunking strategies
│   ├── embeddings.py          # Embedding generation
│   ├── retriever.py           # Vector search logic
│   ├── rag_pipeline.py        # Main RAG orchestration
│   └── prompts.py             # Prompt templates
│
├── scripts/
│   ├── ingest_docs.py         # Documentation ingestion
│   ├── evaluate_rag.py        # RAG metrics evaluation
│   └── test_retrieval.py      # Test retrieval quality
│
├── data/
│   ├── raw/                   # Downloaded documentation
│   ├── processed/             # Chunked documents
│   └── vectordb/              # ChromaDB storage
│
├── tests/
│   ├── test_chunking.py
│   ├── test_retriever.py
│   └── test_rag_pipeline.py
│
└── evals/
    ├── test_queries.json      # Evaluation dataset
    └── results/               # Evaluation outputs
```

## 🎯 Key Features

### 1. Smart Chunking

- **Semantic chunking** with overlap for context preservation
- **Metadata enrichment** (section titles, URLs, code blocks)
- **Configurable chunk sizes** (300-800 tokens)

### 2. Retrieval Quality

- **Hybrid search** (semantic + keyword)
- **Reranking** for improved relevance
- **Source attribution** with confidence scores

### 3. Answer Generation

- **Code-aware formatting** (preserves indentation)
- **Inline citations** with source links
- **Fallback handling** for low-confidence results

### 4. Production Features

- **Health check endpoint** (`/health`)
- **Query logging** for analytics
- **Rate limiting** (basic throttling)
- **Error recovery** with graceful degradation

## 📊 RAG Evaluation

We use **RAGAS** framework to measure:

| Metric                | Description                 | Target Score |
| --------------------- | --------------------------- | ------------ |
| **Faithfulness**      | Answer accuracy vs. context | > 0.8        |
| **Answer Relevancy**  | Response relevance to query | > 0.7        |
| **Context Precision** | Retrieval accuracy          | > 0.75       |
| **Context Recall**    | Context completeness        | > 0.8        |

Run evaluations:

```bash
python evaluate_rag.py
```

## 🐳 Docker Deployment

### Build and run locally:

```bash
docker build -t developerdocs-rag
docker run -p 7860:7860 --name developerdocs-rag-container developerdocs-rag
```

### Deploy to HuggingFace Spaces:

1. Create a new Space on HuggingFace
2. Enable Docker SDK
3. Push this repository
4. Add `HF_TOKEN` as a Space secret
5. Deploy automatically

## 🧪 Testing

```bash
# Run all tests


# Test chunking strategy
pytest test_chunking.py -v

# Test retrieval quality
python test_retrieval.py
```

## 📈 Performance Benchmarks

On HuggingFace Spaces (free tier):

- **Query latency**: ~2-3 seconds
- **Vector DB size**: ~150MB (FastAPI docs)
- **Memory usage**: ~800MB
- **Concurrent users**: 5-10

## 🛠️ Technology Stack

| Component      | Technology                               | Why?                               |
| -------------- | ---------------------------------------- | ---------------------------------- |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Fast, lightweight, good quality    |
| **Vector DB**  | ChromaDB                                 | Easy setup, persistent storage     |
| **LLM**        | HuggingFace Inference API (Mistral-7B)   | Free tier, good code understanding |
| **Framework**  | LangChain                                | Industry standard, modular         |
| **UI**         | Gradio                                   | Rapid prototyping, HF integration  |
| **Deployment** | Docker + HF Spaces                       | Free, scalable, shareable          |

## 🔮 Future Enhancements

- [ ] Multi-documentation support (React, Django, etc.)
- [ ] Conversation memory for follow-up questions
- [ ] Advanced retrieval (HyDE, Multi-Query)
- [ ] User feedback loop for continuous improvement
- [ ] Analytics dashboard for query patterns

## 📝 License

MIT License - feel free to use for your portfolio!

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome via issues.

## 📧 Contact

Built by Aishwarya as a portfolio demonstration of production RAG systems.

- Portfolio: https://aishwarya30998.github.io/projects.html
- LinkedIn: https://www.linkedin.com/in/aishwarya-pentyala/

---

⭐ If this helped you understand production RAG, give it a star!
