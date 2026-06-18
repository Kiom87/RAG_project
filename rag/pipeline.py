"""The RAG pipeline: retrieve relevant chunks, then generate a grounded answer.

The LLM is called through Hugging Face Inference Providers via
huggingface_hub.InferenceClient (OpenAI-compatible chat completions).
If no token is configured or the call fails, the pipeline falls back to
returning the retrieved context so the app keeps working.
"""

from typing import List, Dict, Tuple

from . import config
from .ingest import build_chunks
from .vectorstore import VectorStore

SYSTEM_PROMPT = (
    "You are the Insurellm Expert Knowledge Worker Assistant. "
    "Answer employee and customer questions about Insurellm's insurance products and "
    "policies using ONLY the provided context. "
    "If the answer is not in the context, say you don't have that information and "
    "suggest who to contact. Be concise, accurate, and cite the source document names."
)


class RAGPipeline:
    def __init__(self):
        self.store = VectorStore()
        # Build the index from the knowledge base if it is empty.
        if self.store.count() == 0:
            self.reindex()

    def reindex(self) -> int:
        chunks = build_chunks()
        self.store.index(chunks)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        return self.store.search(query, top_k=top_k)

    def _build_context(self, hits: List[Dict]) -> str:
        blocks = []
        for hit in hits:
            blocks.append(f"[Source: {hit['source']}]\n{hit['text']}")
        return "\n\n---\n\n".join(blocks)

    def generate(self, query: str, hits: List[Dict]) -> str:
        context = self._build_context(hits)
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above."
        )

        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=config.HF_TOKEN)  # token optional
            completion = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            # Graceful fallback: return the most relevant context directly.
            print(f"[pipeline] LLM call failed, using extractive fallback: {exc}")
            top = hits[0]["text"] if hits else "No relevant information found."
            return (
                "I couldn't reach the language model, but here is the most relevant "
                f"information I found:\n\n{top}\n\n"
                "(Set an HF_TOKEN secret in your Space to enable full answers.)"
            )

    def answer(self, query: str, top_k: int = None) -> Tuple[str, List[Dict]]:
        """Return (answer_text, source_hits)."""
        if not query or not query.strip():
            return "Please ask a question about Insurellm.", []
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return "I don't have any documents that match that question.", []
        answer = self.generate(query, hits)
        return answer, hits
