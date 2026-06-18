# RAG Project Planning Document

**Project:** Insurellm Expert Knowledge Worker Assistant
**Type:** Retrieval-Augmented Generation (RAG) application
**Author:** _your name_
**Date:** _date_

---

## 1. Project Objective and Use Case

### Objective
Build an AI assistant that answers questions about Insurellm — a fictional insurance
technology company — by retrieving information from company documents and generating
grounded, accurate answers. The goal is to demonstrate a complete RAG pipeline, from data
ingestion through deployment.

### Problem
Insurellm employees and customers ask the same kinds of questions every day: what products
exist, what policy limits apply, how the claims process works, and who is eligible. The
answers live across many documents. A plain LLM either does not know Insurellm-specific
details or invents them (hallucination).

### Use Case
An internal "knowledge worker" assistant. An employee asks a natural-language question and
receives a concise answer drawn from the official knowledge base, with the source documents
cited so the answer can be trusted and verified.

### Why RAG (not a fine-tuned or plain LLM)
- **Accuracy:** answers are grounded in real documents.
- **Freshness:** updating knowledge means adding a file, not retraining.
- **Low cost:** no fine-tuning; uses a free hosted LLM and local embeddings.
- **Trust:** every answer lists its sources.

### Success Criteria
- Correctly answers the benchmark questions (e.g. health waiting period = 90 days, cyber
  Growth-tier limit = $1,000,000).
- Returns "I don't know" rather than hallucinating when information is absent.
- Publicly accessible and functional on Hugging Face Spaces.

---

## 2. Dataset and Data Sources

### Dataset
A curated knowledge base of Insurellm company documents stored as Markdown under
`knowledge-base/`, organised into three categories:

| Category   | Documents | Example content |
|------------|-----------|-----------------|
| `company/` | overview  | mission, products list, contact details |
| `products/`| Healthllm, Cyberllm, Autollm, Bizllm | coverage, limits, eligibility, pricing |
| `policies/`| claims process, eligibility, employee handbook | procedures and rules |

Insurellm is fictional, so the documents are illustrative sample data created for this
assignment. The pipeline also accepts `.txt` and `.pdf` files, so real documents can be
added later without code changes.

### Data Characteristics
- ~10 documents, short to medium length, well-structured prose.
- Documents are split into ~800-character chunks with ~120-character overlap to keep
  retrieved passages topically coherent while preserving context across boundaries.

### Data Flow
Raw documents → text extraction → chunking → embedding → stored in the vector database.

---

## 3. RAG Architecture and Workflow

### Indexing (offline, runs once at startup)
1. **Load** every document in `knowledge-base/`.
2. **Chunk** each document into overlapping passages.
3. **Embed** each chunk into a vector using a sentence-transformer model.
4. **Store** the vectors, text, and source metadata in ChromaDB.

### Query (online, per question)
1. **Embed** the user's question with the same model.
2. **Retrieve** the top-k (default 4) most similar chunks via cosine similarity.
3. **Augment** a prompt with the retrieved context and the question.
4. **Generate** an answer with the LLM, constrained to use only the provided context.
5. **Cite** the source documents alongside the answer.

```
            ┌─────────────── Indexing (startup) ───────────────┐
documents → load → chunk → embed → ChromaDB (vectors + sources)
            └───────────────────────────────────────────────────┘

            ┌──────────────── Query (per question) ─────────────┐
question → embed → similarity search → top-k chunks
                                              │
              system prompt + context + question
                                              │
                                            LLM → grounded answer + sources
            └───────────────────────────────────────────────────┘
```

### Design Decisions
- **Local embeddings** keep retrieval free and fast and require no API key.
- **Constrained prompt** ("answer only from the context") reduces hallucination.
- **Graceful fallback:** if the LLM is unreachable, the app returns the top retrieved
  passage so it never hard-fails during demos or grading.

---

## 4. Tools, Frameworks, and Technologies

| Component        | Technology | Reason |
|------------------|------------|--------|
| Language         | Python 3.10+ | Standard for ML/RAG |
| Embeddings       | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, local, CPU-friendly |
| Vector database  | ChromaDB | Lightweight, persistent, part of the course curriculum |
| LLM              | Hugging Face Inference Providers via `huggingface_hub.InferenceClient` | Free tier, OpenAI-compatible, no separate paid key |
| Default model    | `Qwen/Qwen2.5-7B-Instruct` (configurable) | Ungated, widely served, good quality |
| PDF parsing      | `pypdf` | Adds support for custom PDF documents |
| Web UI           | Gradio (`ChatInterface`) | Native fit for Hugging Face Spaces |
| Version control  | Git + GitHub | Source control and documentation |
| Deployment       | Hugging Face Spaces | Free, public, simple Gradio hosting |

---

## 5. Project Plan and Milestones

| Milestone | Deliverable |
|-----------|-------------|
| 1. Planning | This document |
| 2. Data | Knowledge base assembled in `knowledge-base/` |
| 3. Pipeline | Ingestion, embeddings, vector store, retrieval, generation |
| 4. App | Gradio chat interface |
| 5. Repository | GitHub repo with README and regular commits |
| 6. Deployment | Live, functional Hugging Face Space |
| 7. Submission | Share GitHub + Space links; email both URLs |

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| LLM endpoint unavailable | Extractive fallback returns the top retrieved passage |
| Model gated / unavailable | `LLM_MODEL` is configurable via environment variable |
| Irrelevant retrieval | Tune `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Hallucination | Prompt constrains the model to the provided context only |

---

## 7. Future Improvements
- Hybrid search (keyword + vector) and a reranking step.
- Metadata filtering (e.g. restrict to a product line).
- Conversation memory for multi-turn follow-ups.
- Evaluation harness measuring answer accuracy against a question/answer set.
