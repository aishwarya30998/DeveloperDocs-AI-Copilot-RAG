"""
Developer Docs AI Copilot - Gradio UI Application

Production-grade RAG chatbot interface for any developer documentation.

Two-tab UI:
  Setup tab  — enter a docs URL, trigger ingestion/embedding
  Chat tab   — ask questions, get answers with source citations
"""
import logging
import sys
import queue
import threading
from pathlib import Path
from typing import List, Tuple, Optional
import gradio as gr
from datetime import datetime
import json
from urllib.parse import urlparse

from src import create_rag_pipeline, settings
from src.config import RESULTS_DIR
from ingest_docs import run_ingestion

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global pipeline state
rag_pipeline = None
pipeline_stats: dict = {}
current_docs_name: str = settings.docs_name  # may be updated after ingestion


def _try_load_pipeline():
    """Attempt to load the RAG pipeline from an existing vector DB."""
    global rag_pipeline, pipeline_stats
    try:
        rag_pipeline = create_rag_pipeline()
        pipeline_stats = rag_pipeline.get_stats()
        logger.info(f"Pipeline loaded. {pipeline_stats.get('total_chunks', 0)} chunks indexed.")
    except Exception as e:
        logger.warning(f"Could not load pipeline on startup (run Setup first): {e}")
        rag_pipeline = None
        pipeline_stats = {}


_try_load_pipeline()


# Query logging

QUERY_LOG_FILE = RESULTS_DIR / "query_log.jsonl"


def log_query(question: str, response: dict):
    try:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "docs_name": current_docs_name,
            "question": question,
            "answer": response.get("answer", ""),
            "source_count": response.get("source_count", 0),
            "confidence": response.get("confidence", "unknown"),
            "chunks_retrieved": response.get("chunks_retrieved", 0),
        }
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log query: {e}")


# Chat helpers

def format_sources(sources: List[dict]) -> str:
    if not sources:
        return "No sources available."
    formatted = "### Sources\n\n"
    for i, source in enumerate(sources, 1):
        title = source.get("title", "Unknown")
        section = source.get("section", "")
        url = source.get("url", "#")
        score = source.get("score", 0.0)
        formatted += f"{i}. **{title}**"
        if section:
            formatted += f" ({section})"
        formatted += f"\n   - Relevance: {score:.2%}\n"
        if url and url != "#":
            formatted += f"   - [View Documentation]({url})\n"
        formatted += "\n"
    return formatted


def process_query(question: str, history: List[Tuple[str, str]]) -> Tuple[str, str]:
    if not rag_pipeline:
        return (
            "Pipeline not ready. Please go to the **Setup** tab and ingest documentation first.",
            "No sources available.",
        )
    if not question or not question.strip():
        return "Please enter a question.", ""

    try:
        logger.info(f"Processing query: {question[:100]}...")
        response = rag_pipeline.query(question, top_k=5)
        log_query(question, response)

        answer = response["answer"]
        confidence = response.get("confidence", "unknown")
        chunks_retrieved = response.get("chunks_retrieved", 0)
        answer += f"\n\n---\n*Confidence: {confidence.upper()} | Retrieved {chunks_retrieved} chunks*"

        sources_text = format_sources(response.get("sources", []))
        return answer, sources_text

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return f"Error: {str(e)}", "No sources available."


# Ingestion helper — runs in a background thread, streams log lines via queue
def _derive_docs_name(url: str) -> str:
    hostname = urlparse(url).hostname or ""
    return hostname.split(".")[0].replace("-", " ").title()


def ingest_and_stream(docs_url: str, docs_name: str, url_patterns_raw: str):
    """
    Generator function: runs ingestion in a background thread and streams
    status lines to the Gradio Textbox.
    """
    global rag_pipeline, pipeline_stats, current_docs_name

    docs_url = docs_url.strip().rstrip("/")
    docs_name = docs_name.strip() or _derive_docs_name(docs_url)
    url_patterns = [p.strip() for p in url_patterns_raw.split(",") if p.strip()]

    if not docs_url:
        yield "Please enter a documentation URL."
        return

    # Queue used to pass log lines from the worker thread to the generator
    log_q: queue.Queue = queue.Queue()
    result_holder: dict = {}
    error_holder: dict = {}

    def worker():
        try:
            stats = run_ingestion(
                docs_url=docs_url,
                docs_name=docs_name,
                url_patterns=url_patterns or None,
                progress_callback=lambda msg: log_q.put(msg),
            )
            result_holder["stats"] = stats
        except Exception as exc:
            error_holder["error"] = str(exc)
            logger.error(f"Ingestion failed: {exc}", exc_info=True)
        finally:
            log_q.put(None)  # sentinel

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Stream log lines as they arrive
    accumulated = ""
    while True:
        try:
            line = log_q.get(timeout=120)
        except queue.Empty:
            yield accumulated + "\n[Timed out waiting for ingestion]"
            return

        if line is None:  # sentinel → done
            break

        accumulated += line + "\n"
        yield accumulated

    thread.join(timeout=5)

    if "error" in error_holder:
        yield accumulated + f"\n\nIngestion failed: {error_holder['error']}"
        return

    # Reload the RAG pipeline with the newly ingested docs
    accumulated += "\nReloading RAG pipeline..."
    yield accumulated

    try:
        # Update settings so the pipeline and prompts use the new docs name
        settings.docs_url = docs_url
        settings.docs_name = docs_name
        current_docs_name = docs_name

        rag_pipeline = create_rag_pipeline()
        pipeline_stats = rag_pipeline.get_stats()

        accumulated += f"\nPipeline ready — {pipeline_stats.get('total_chunks', 0)} chunks indexed."
        accumulated += f"\n\nSwitch to the Chat tab and start asking questions about {docs_name}!"
        yield accumulated

    except Exception as e:
        accumulated += f"\n\nPipeline reload failed: {e}"
        yield accumulated


# UI
def create_ui():
    custom_css = """
    .stats-box {
        background: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(
        title="Developer Docs AI Copilot",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as app:

        gr.Markdown("# Developer Docs AI Copilot")
        gr.Markdown(
            "Ingest any developer documentation and ask questions answered directly from the source."
        )

        with gr.Tabs() as tabs:

            # TAB 1 — Setup

            with gr.Tab("⚙️ Setup — Ingest Docs", id="setup"):
                gr.Markdown(
                    "Enter the URL of any developer documentation site. "
                    "The system will scrape, chunk, embed, and index it for Q&A."
                )

                with gr.Row():
                    docs_url_input = gr.Textbox(
                        label="Documentation URL",
                        placeholder="e.g. https://docs.djangoproject.com/en/stable/",
                        scale=3,
                    )
                    docs_name_input = gr.Textbox(
                        label="Docs Name (optional — auto-derived if empty)",
                        placeholder="e.g. Django",
                        scale=1,
                    )

                url_patterns_input = gr.Textbox(
                    label="URL Path Patterns to include (optional, comma-separated)",
                    placeholder="e.g. /topics,/ref,/howto   — leave empty to include all pages",
                )

                ingest_btn = gr.Button("Ingest Documentation", variant="primary")

                ingest_status = gr.Textbox(
                    label="Ingestion Log",
                    lines=20,
                    interactive=False,
                    placeholder="Status will appear here when you click Ingest...",
                )

                # Wire up the button to the streaming generator
                ingest_btn.click(
                    fn=ingest_and_stream,
                    inputs=[docs_url_input, docs_name_input, url_patterns_input],
                    outputs=ingest_status,
                )

                gr.Markdown("""
                **Tips:**
                - Most documentation sites (FastAPI, Django, React, Stripe, etc.) work out of the box
                - Use URL patterns to ingest only a specific section (faster)
                - Re-run ingestion any time to switch to a different documentation source
                - Default page cap is **50 pages** — sufficient for most demos
                """)

            # TAB 2 — Chat
            with gr.Tab("💬 Chat", id="chat"):

                # Live status bar
                status_text = (
                    f"Ready — {pipeline_stats.get('total_chunks', 0)} chunks indexed "
                    f"({current_docs_name})"
                    if rag_pipeline
                    else "Not ready — please ingest documentation in the Setup tab first."
                )
                status_md = gr.Markdown(f"**Status:** {status_text}")

                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=420,
                            show_copy_button=True,
                        )

                        with gr.Row():
                            question_input = gr.Textbox(
                                label="Ask a question",
                                placeholder="e.g. How do I get started?",
                                lines=2,
                                scale=4,
                            )
                            submit_btn = gr.Button("Ask", variant="primary", scale=1)

                        gr.Examples(
                            examples=[
                                "How do I get started?",
                                "What are the core concepts?",
                                "Show me a basic example",
                                "How do I handle authentication?",
                                "What is the recommended project structure?",
                            ],
                            inputs=question_input,
                            label="Example Questions",
                        )

                    with gr.Column(scale=1):
                        sources_display = gr.Markdown(
                            value="Sources will appear here after asking a question."
                        )

                clear_btn = gr.Button("Clear Conversation")

                def respond(message, chat_history):
                    answer, sources = process_query(message, chat_history)
                    chat_history.append((message, answer))
                    return "", chat_history, sources

                submit_btn.click(
                    respond,
                    inputs=[question_input, chatbot],
                    outputs=[question_input, chatbot, sources_display],
                )
                question_input.submit(
                    respond,
                    inputs=[question_input, chatbot],
                    outputs=[question_input, chatbot, sources_display],
                )
                clear_btn.click(
                    lambda: ([], "Sources will appear here after asking a question."),
                    outputs=[chatbot, sources_display],
                )

        gr.Markdown(
            "---\n*Built with: ChromaDB · Sentence Transformers · HuggingFace · Gradio*"
        )

    return app


def health_check():
    return {"status": "healthy", "pipeline_ready": rag_pipeline is not None}


if __name__ == "__main__":
    logger.info("Starting Developer Docs AI Copilot...")
    app = create_ui()
    logger.info(f"Launching on port {settings.app_port}")
    app.launch(
        server_name="0.0.0.0",
        server_port=settings.app_port,
        share=False,
        show_error=True,
    )
