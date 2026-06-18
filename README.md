---
title: Insurellm Knowledge Assistant
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: mit
---

# 🛡️ Insurellm Expert Knowledge Worker Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers employee and customer
questions about **Insurellm**, a fictional insurance technology company. Instead of
relying on an LLM's memory, it **searches the company knowledge base first** and then
generates an answer grounded in the retrieved documents — more accurate, company-specific,
and far less prone to hallucination.

> The YAML block at the top of this file configures the Hugging Face Space. GitHub
> ignores it, so the same README works for both the repo and the deployment.

## Live Demo
- **Hugging Face Space:** _add your Space URL here_
- **GitHub repo:** _add your repo URL here_

## What It Does
Ask things like:
- "What is the waiting period for our health insurance plan?"
- "What is our cyber insurance coverage limit?"
- "What insurance products does Insurellm offer for small businesses?"
- "How do I file a claim?"

The assistant retrieves the most relevant document chunks and answers from them, listing
the source files it used.

## Architecture

```
User question
     │
     ▼
Embed query  ──►  ChromaDB similarity search  ──►  Top-k relevant chunks
                                                          │
                                                          ▼
                              Prompt = system + context + question
                                                          │
                                                          ▼
                        LLM (HF Inference Providers)  ──►  Grounded answer + sources
```

| Stage          | Choice                                              | Why |
|----------------|-----------------------------------------------------|-----|
| Embeddings     | `sentence-transformers/all-MiniLM-L6-v2` (local)    | Free, fast, CPU-friendly, no API key |
| Vector store   | ChromaDB (persistent, cosine)                       | Simple, beginner-friendly, in the course curriculum |
| LLM            | HF Inference Providers via `InferenceClient`        | Free tier, no separate paid key, OpenAI-compatible |
| UI             | Gradio `ChatInterface`                              | Native fit for Hugging Face Spaces |
| Docs           | Markdown / PDF in `knowledge-base/`                 | Easy to extend with your own files |

## Project Structure

```
insurellm-rag-assistant/
├── app.py                  # Gradio app (HF Spaces entry point)
├── requirements.txt
├── PROJECT_PLAN.md         # planning document
├── rag/
│   ├── config.py           # settings (env-overridable)
│   ├── ingest.py           # load + chunk documents
│   ├── vectorstore.py      # embeddings + ChromaDB
│   └── pipeline.py         # retrieval + LLM generation
└── knowledge-base/         # the company documents (the "data")
    ├── company/
    ├── products/
    └── policies/
```

## Run Locally

```bash
git clone <your-repo-url>
cd insurellm-rag-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export HF_TOKEN=your_hf_token        # Windows: set HF_TOKEN=your_hf_token
python app.py
```

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

## Deploy to Hugging Face Spaces
1. Create a new Space at https://huggingface.co/new-space → SDK: **Gradio**.
2. Push this repository to the Space (or connect it to your GitHub repo).
3. In **Settings → Variables and secrets**, add a secret named `HF_TOKEN`
   (create a token at https://huggingface.co/settings/tokens).
4. The Space builds automatically and the app goes live.

The vector index is built from `knowledge-base/` on first startup, so there is nothing
extra to upload.

## Add Your Own Documents
Drop `.md`, `.txt`, or `.pdf` files anywhere under `knowledge-base/` and restart the app.
They are automatically chunked, embedded, and made searchable.

## Configuration
All settings live in `rag/config.py` and can be overridden with environment variables
(`LLM_MODEL`, `EMBEDDING_MODEL`, `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`). See `.env.example`.

## Notes
Insurellm is a fictional company; the documents in `knowledge-base/` are illustrative
sample data created for this assignment.

## License
MIT
