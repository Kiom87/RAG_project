import gradio as gr
from rag import RAGPipeline

pipeline = RAGPipeline()

EXAMPLES = [
    "What is the waiting period for our health insurance plan?",
    "What is our cyber insurance coverage limit?",
    "What insurance products does Insurellm offer for small businesses?",
    "How do I file a claim?",
    "Who is eligible for Autollm?",
]


def respond(message, history):
    answer, hits = pipeline.answer(message)
    if hits:
        sources = "\n".join(f"- `{h['source']}` (relevance {h['score']})" for h in hits)
        answer = f"{answer}\n\n---\n**Sources**\n{sources}"
    return answer


with gr.Blocks(title="Insurellm Knowledge Assistant") as demo:
    gr.Markdown(
        "# 🛡️ Insurellm Expert Knowledge Worker Assistant\n"
        "Ask questions about Insurellm's insurance products and policies. "
        "Answers are retrieved from the company knowledge base (RAG), grounded in real documents."
    )
    gr.ChatInterface(fn=respond, examples=EXAMPLES)
    gr.Markdown(
        "_Built with sentence-transformers (embeddings), ChromaDB (vector store), "
        "and Hugging Face Inference Providers (LLM)._"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False, theme=gr.themes.Soft())
