import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.environ.get("KB_DIR", os.path.join(BASE_DIR, "knowledge-base"))
CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.join(BASE_DIR, ".chroma"))
COLLECTION_NAME = "insurellm"

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "120"))
TOP_K = int(os.environ.get("TOP_K", "4"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))
