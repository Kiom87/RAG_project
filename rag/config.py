"""Central configuration for the Insurellm RAG assistant.

All values can be overridden with environment variables, which is convenient
when deploying to Hugging Face Spaces (set them as Space secrets/variables).
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.environ.get("KB_DIR", os.path.join(BASE_DIR, "knowledge-base"))
CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.join(BASE_DIR, ".chroma"))
COLLECTION_NAME = "insurellm"

# --- Embeddings (runs locally, free, CPU-friendly) ---
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- LLM via Hugging Face Inference Providers ---
# A Hugging Face token (HF_TOKEN) is read automatically by InferenceClient.
# Set it as a secret named HF_TOKEN in your Space settings.
# Known-good ungated options: "Qwen/Qwen2.5-7B-Instruct",
# "meta-llama/Llama-3.1-8B-Instruct", "openai/gpt-oss-20b".
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")  # optional; picked up automatically too

# --- Chunking ---
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))      # characters per chunk
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "120"))  # overlap between chunks

# --- Retrieval ---
TOP_K = int(os.environ.get("TOP_K", "4"))  # number of chunks to retrieve

# --- Generation ---
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))
