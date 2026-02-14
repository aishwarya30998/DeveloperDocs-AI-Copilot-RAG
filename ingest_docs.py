"""
Ingest developer documentation into the vector database.

This script:
1. Scrapes documentation from any URL (via sitemap or recursive crawl)
2. Chunks the content semantically
3. Generates embeddings
4. Stores in ChromaDB

Usage:
    python ingest_docs.py

Configure via environment variables (or .env):
    DOCS_URL          - Base URL of the documentation (required)
    DOCS_NAME         - auto-derived if empty
    DOCS_URL_PATTERNS - Comma-separated path patterns to include, e.g. "/tutorial,/guide"
                        Leave empty to include all pages under the base URL.
    COLLECTION_NAME   - ChromaDB collection name
"""
import logging
import re
from pathlib import Path
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import json

from src.config import settings, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.chunking import create_chunker
from src.retriever import create_retriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocsScraper:
    """
    Generic documentation scraper that works with any documentation site.

    Discovers pages via sitemap.xml first; falls back to recursive same-domain
    crawling if no sitemap is available.
    """

    def __init__(
        self,
        base_url: str,
        url_patterns: Optional[List[str]] = None,
        max_pages: int = 200,
    ):
        """
        Args:
            base_url:     Root URL of the documentation site.
            url_patterns: Optional list of path substrings to include
                          (e.g. ["/tutorial", "/guide"]).  When empty/None,
                          all pages whose URL starts with base_url are included.
            max_pages:    Safety cap on the number of pages to scrape.
        """
        self.base_url = base_url.rstrip("/")
        self.url_patterns = url_patterns or []
        self.max_pages = max_pages

        parsed = urlparse(self.base_url)
        self.base_domain = parsed.netloc

    # URL discovery
    def get_doc_urls(self) -> List[str]:
        """Return a deduplicated list of documentation page URLs."""
        urls = self._urls_from_sitemap()
        if not urls:
            logger.warning("No sitemap found or empty — falling back to recursive crawl")
            urls = self._urls_from_crawl()

        urls = self._filter_urls(urls)
        logger.info(f"Discovered {len(urls)} documentation pages")
        return urls[: self.max_pages]

    def _urls_from_sitemap(self) -> List[str]:
        """Try to fetch all URLs from sitemap.xml."""
        sitemap_url = f"{self.base_url}/sitemap.xml"
        logger.info(f"Fetching sitemap: {sitemap_url}")
        try:
            resp = requests.get(sitemap_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "xml")
            urls = [loc.text.strip() for loc in soup.find_all("loc")]
            logger.info(f"Found {len(urls)} URLs in sitemap")
            return urls
        except Exception as e:
            logger.warning(f"Could not load sitemap: {e}")
            return []

    def _urls_from_crawl(self, start_url: Optional[str] = None) -> List[str]:
        """
        Recursively crawl same-domain links starting from base_url.
        Limited to self.max_pages pages to avoid runaway crawls.
        """
        start = start_url or self.base_url
        visited: set = set()
        queue: List[str] = [start]
        found: List[str] = []

        while queue and len(found) < self.max_pages * 2:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.content, "html.parser")
                found.append(url)

                for tag in soup.find_all("a", href=True):
                    href = tag["href"].strip()
                    absolute = urljoin(url, href).split("#")[0]
                    if (
                        absolute not in visited
                        and urlparse(absolute).netloc == self.base_domain
                        and absolute.startswith("http")
                    ):
                        queue.append(absolute)
            except Exception as e:
                logger.debug(f"Crawl error for {url}: {e}")

        return found

    def _filter_urls(self, urls: List[str]) -> List[str]:
        """
        Keep only URLs that belong to the same domain and, if url_patterns
        is set, match at least one pattern.
        """
        filtered = []
        for url in urls:
            parsed = urlparse(url)
            if parsed.netloc != self.base_domain:
                continue
            if self.url_patterns:
                if not any(p in parsed.path for p in self.url_patterns):
                    continue
            filtered.append(url)
        seen = set()
        unique = []
        for u in filtered:
            if u not in seen:
                seen.add(u)
                unique.append(u)
        return unique

    # Page scraping
    def scrape_page(self, url: str) -> Dict[str, Any]:
        """
        Scrape a single documentation page.

        Returns a dict with keys: url, title, section, content, success.
        """
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")

            main_content = (
                soup.find("main")
                or soup.find("article")
                or soup.find(attrs={"role": "main"})
                or soup.find("div", class_=re.compile(r"content|doc|page|main", re.I))
                or soup.find("body")
            )

            if not main_content:
                logger.warning(f"No content container found for {url}")
                return {"url": url, "success": False}

            # Strip navigation / chrome elements
            for unwanted in main_content.find_all(
                ["nav", "header", "footer", "script", "style", "aside"]
            ):
                unwanted.decompose()

            text = main_content.get_text(separator="\n", strip=True)

            h1 = soup.find("h1")
            if h1:
                title_text = h1.get_text(strip=True)
            elif soup.title:
                title_text = soup.title.get_text(strip=True)
            else:
                parts = [p for p in urlparse(url).path.split("/") if p]
                title_text = parts[-1].replace("-", " ").replace("_", " ").title() if parts else url

            path_parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
            section = path_parts[0].replace("-", " ").replace("_", " ").title() if path_parts else "General"

            return {
                "url": url,
                "title": title_text,
                "section": section,
                "content": text,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {"url": url, "success": False, "error": str(e)}


# Helpers

def _safe_filename(name: str) -> str:
    """Convert a docs name into a safe filename prefix."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).lower()


# Programmatic ingestion API (used by app.py UI)
def run_ingestion(
    docs_url: str,
    docs_name: str,
    url_patterns: Optional[List[str]] = None,
    max_pages: int = 50,
    progress_callback=None,
) -> dict:
    """
    Run the full ingestion pipeline programmatically.

    Args:
        docs_url:          Base URL of the documentation site.
        docs_name:         Human-readable name.
        url_patterns:      Optional list of path substrings to filter pages.
        max_pages:         Maximum number of pages to scrape.
        progress_callback: Optional callable(message: str) for live status updates.

    Returns:
        Stats dict with keys: total_chunks, collection_name, embedding_dimension,
        metadata_fields, pages_scraped.
    """
    def emit(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    safe_name = _safe_filename(docs_name)
    url_patterns = url_patterns or []

    emit("=" * 50)
    emit(f"Ingestion Pipeline: {docs_name}")
    emit(f"Source: {docs_url}")
    if url_patterns:
        emit(f"URL patterns: {url_patterns}")
    emit("=" * 50)

    # Step 1: Scrape
    emit(f"\n[1/4] Discovering and scraping {docs_name} documentation...")
    scraper = DocsScraper(
        base_url=docs_url,
        url_patterns=url_patterns,
        max_pages=max_pages * 4,
    )
    urls = scraper.get_doc_urls()
    urls = urls[:max_pages]
    emit(f"      Scraping {len(urls)} pages...")

    documents = []
    for i, url in enumerate(urls, 1):
        doc = scraper.scrape_page(url)
        if doc.get("success"):
            documents.append(doc)
        if i % 10 == 0 or i == len(urls):
            emit(f"      Scraped {i}/{len(urls)} pages ({len(documents)} succeeded)")

    emit(f"[1/4] Done — {len(documents)} pages scraped successfully")

    # Save raw documents
    raw_file = RAW_DATA_DIR / f"{safe_name}_docs_raw.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    # Step 2: Chunk
    emit(f"\n[2/4] Chunking {len(documents)} documents...")
    chunker = create_chunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    all_chunks = []
    for doc in documents:
        metadata = {
            "source": doc["url"],
            "title": doc["title"],
            "section": doc["section"],
            "url": doc["url"],
            "docs_name": docs_name,
        }
        chunks = chunker.chunk_document(text=doc["content"], metadata=metadata)
        all_chunks.extend(chunks)

    emit(f"[2/4] Done — {len(all_chunks)} chunks created")

    processed_file = PROCESSED_DATA_DIR / f"{safe_name}_docs_chunks.json"
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(
            [chunk.to_dict() for chunk in all_chunks],
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Step 3: Embed + store
    emit(f"\n[3/4] Generating embeddings and storing in ChromaDB...")
    emit(f"      This may take a few minutes for large doc sets...")
    retriever = create_retriever()

    try:
        retriever.reset_collection()
    except Exception:
        pass

    batch_size = 100
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    for idx, i in enumerate(range(0, len(all_chunks), batch_size), 1):
        batch = all_chunks[i : i + batch_size]
        retriever.add_documents(batch)
        emit(f"      Stored batch {idx}/{total_batches}")

    # Step 4: Verify
    emit(f"\n[4/4] Verifying ingestion...")
    stats = retriever.get_collection_stats()
    stats["pages_scraped"] = len(documents)

    emit("\n" + "=" * 50)
    emit("Ingestion Complete!")
    emit(f"  Pages scraped  : {len(documents)}")
    emit(f"  Chunks indexed : {stats['total_chunks']}")
    emit(f"  Collection     : {stats['collection_name']}")
    emit(f"  Embedding dim  : {stats['embedding_dimension']}")
    emit("=" * 50)

    return stats

# CLI entry point
def main():
    """CLI entry point — reads config from settings / .env."""
    url_patterns: List[str] = []
    if settings.docs_url_patterns.strip():
        url_patterns = [p.strip() for p in settings.docs_url_patterns.split(",") if p.strip()]

    run_ingestion(
        docs_url=settings.docs_url,
        docs_name=settings.docs_name,
        url_patterns=url_patterns,
    )
    logger.info("Ready to use! Run 'python app.py' to start the UI")


if __name__ == "__main__":
    main()
