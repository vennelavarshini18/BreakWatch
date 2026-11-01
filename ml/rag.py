"""
ml/rag.py — Retrieval over ICAR district contingency plans.

Usage
-----
Build (or rebuild) the index:
    python -m ml.rag --build

Query:
    python -m ml.rag "mid season dry spell cotton" --district Nalgonda

Architecture
------------
* Chunks by *condition block* (early season drought, mid season drought,
  terminal drought, delayed onset, flood, excess rain, heat wave, cold wave …)
  when the PDF tables parse reliably.  Falls back to page-level chunks and
  logs which districts needed the fallback.

* Primary retrieval: ChromaDB (persisted at data/model/chroma) with the
  all-MiniLM-L6-v2 sentence-transformer embedding model.

* Fallback: rank_bm25 BM25 index, rebuilt in memory from the same chunks.
  Active when ChromaDB or the embedding model fail to load.

* District filtering is applied *before* semantic ranking so we never return
  passages from the wrong district.

* Returns contracts.Passage objects.  On any internal error, returns []
  and logs the reason.

PDF corpus
----------
data/corpus/icar/<district>_<year>.pdf

District names (canonical, matching data/train_points.json / boundaries):
  adilabad, karimnagar, khammam, mahabubnagar, medak, nalgonda,
  nizamabad, rangareddy, warangal

All 9 PDFs are the ICAR 2011 Andhra Pradesh contingency plans (finalised
31 Jan 2011, covering 2008-09 data).  Year found from PDF metadata title
field: "Microsoft Word - AP*-<District> 31.1.2011.doc".

Leap-year note (for chunk_id):
  None here; chunk IDs use page numbers (0-indexed) and sequential counters
  within the page; no date arithmetic is done in this module.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml.rag")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "corpus" / "icar"
CHROMA_DIR = REPO_ROOT / "data" / "model" / "chroma"

EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "icar_contingency"

# Minimum extractable characters per page; below this we flag as scan-like
MIN_TEXT_CHARS_PER_PAGE = 50

# How many characters count as "almost no text" for the whole PDF
MIN_TOTAL_CHARS = 500

STALE_NDVI_DAYS = 20  # mirrored from common/features; not used here but noted

# Condition keywords used to split text into condition blocks
_CONDITION_PATTERNS = [
    r"early\s+season\s+drought",
    r"mid\s+season\s+drought",
    r"terminal\s+drought",
    r"delayed\s+release\s+of\s+water",
    r"unusual\s+rains?",
    r"continuous\s+high\s+rainfall",
    r"heavy\s+rainfall",
    r"floods?",
    r"extreme\s+events?",
    r"heat\s+wave",
    r"cold\s+wave",
]

CONDITION_RE = re.compile(
    r"(?:" + "|".join(_CONDITION_PATTERNS) + r")",
    re.IGNORECASE,
)

# Crops to detect in chunk text
_CROP_PATTERNS = [
    "cotton", "redgram", "red gram", "paddy", "rice", "maize", "corn",
    "groundnut", "castor", "soybean", "jowar", "greengram", "green gram",
    "sugarcane", "sunflower", "turmeric", "bengalgram", "wheat", "bajra",
]
CROP_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(c) for c in _CROP_PATTERNS) + r")\b",
    re.IGNORECASE,
)

# canonical district name -> plan year (from PDF metadata)
_DISTRICT_YEAR: dict[str, int] = {
    "adilabad": 2011,
    "karimnagar": 2011,
    "khammam": 2011,
    "mahabubnagar": 2011,
    "medak": 2011,
    "nalgonda": 2011,
    "nizamabad": 2011,
    "rangareddy": 2011,
    "warangal": 2011,
}

# ---------------------------------------------------------------------------
# District name normalisation
# ---------------------------------------------------------------------------

def _normalise_district(name: str) -> str:
    """Return the canonical lowercase district key from any spelling variant.

    Handles:
    - Leading/trailing whitespace
    - Mixed case: 'Ranga Reddy', 'RANGAREDDY', 'rangareddy'
    - Spaces: 'Ranga Reddy' -> 'rangareddy'
    - Common aliases: 'RangaReddy' -> 'rangareddy', 'Mahbubnagar' -> 'mahabubnagar'
    """
    key = re.sub(r"\s+", "", name).lower()
    # Common spelling variants
    aliases = {
        "mahbubnagar": "mahabubnagar",
        "mahbubnager": "mahabubnagar",
        "ranga reddy": "rangareddy",
        "rangareddy": "rangareddy",
        "rangareddi": "rangareddy",
    }
    return aliases.get(key, key)


# ---------------------------------------------------------------------------
# PDF extraction helpers
# ---------------------------------------------------------------------------


def _extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text per page from a PDF.  Returns list of {page, text} dicts."""
    try:
        import pymupdf  # noqa: PLC0415
    except ImportError:
        log.error("pymupdf not installed; cannot extract PDFs")
        return []

    pages = []
    try:
        doc = pymupdf.open(str(pdf_path))
        for pg in range(len(doc)):
            text = doc[pg].get_text("text")
            pages.append({"page": pg, "text": text})
        doc.close()
    except Exception as exc:
        log.error("Failed to open %s: %s", pdf_path, exc)
    return pages


def _is_scan_like(pages: list[dict]) -> bool:
    """Return True when the PDF has almost no extractable text (likely a scan)."""
    total = sum(len(p["text"]) for p in pages)
    return total < MIN_TOTAL_CHARS


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _condition_from_text(text: str) -> str:
    """Extract the first condition keyword found in text, or 'general'."""
    m = CONDITION_RE.search(text)
    if m:
        return re.sub(r"\s+", " ", m.group(0)).strip().lower()
    return "general"


def _crops_from_text(text: str) -> Optional[str]:
    """Return comma-separated crop names found in text, or None."""
    found = list(dict.fromkeys(
        m.group(0).lower().replace(" ", "")
        for m in CROP_RE.finditer(text)
    ))
    return ", ".join(found) if found else None


def _chunk_by_condition(
    pages: list[dict],
    district: str,
    plan_year: int,
    doc_name: str,
) -> tuple[list[dict], bool]:
    """Split contingency table pages into condition-level chunks.

    Returns (chunks, used_fallback).  used_fallback=True means we fell back
    to page-level chunking because no condition keywords were found at all.
    """
    chunks: list[dict] = []
    any_condition_found = False

    for page_info in pages:
        pg = page_info["page"]
        text = page_info["text"]
        if not text.strip():
            continue

        # Find all condition match positions in this page
        matches = list(CONDITION_RE.finditer(text))
        if not matches:
            # No conditions on this page → page-level chunk if non-trivial
            if len(text.strip()) >= 80:
                chunk_id = f"{district}_{plan_year}_p{pg}_c0"
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc": doc_name,
                    "district": district,
                    "plan_year": plan_year,
                    "condition": "general",
                    "crop": _crops_from_text(text),
                    "page": pg,
                    "text": text.strip(),
                })
            continue

        any_condition_found = True
        # Text before first match → general chunk if non-trivial
        prefix = text[: matches[0].start()].strip()
        if len(prefix) >= 80:
            chunk_id = f"{district}_{plan_year}_p{pg}_c0"
            chunks.append({
                "chunk_id": chunk_id,
                "doc": doc_name,
                "district": district,
                "plan_year": plan_year,
                "condition": "general",
                "crop": _crops_from_text(prefix),
                "page": pg,
                "text": prefix,
            })

        # Split page text at each condition match start
        boundaries = [m.start() for m in matches] + [len(text)]
        for ci, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            block = text[start:end].strip()
            if len(block) < 60:
                continue
            condition = _condition_from_text(block)
            chunk_id = f"{district}_{plan_year}_p{pg}_c{ci + 1}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc": doc_name,
                "district": district,
                "plan_year": plan_year,
                "condition": condition,
                "crop": _crops_from_text(block),
                "page": pg,
                "text": block,
            })

    used_fallback = not any_condition_found
    return chunks, used_fallback


def build_chunks() -> tuple[list[dict], list[str]]:
    """Parse all PDFs in CORPUS_DIR and return (all_chunks, fallback_districts).

    fallback_districts lists districts where page-level fallback was used.
    """
    all_chunks: list[dict] = []
    fallback_districts: list[str] = []
    stats: dict[str, int] = {}

    pdf_files = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDFs found in %s", CORPUS_DIR)
        return [], []

    for pdf_path in pdf_files:
        # Parse filename: <district>_<year>.pdf
        stem = pdf_path.stem  # e.g. "nalgonda_2011"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            log.warning("Skipping unexpected filename: %s", pdf_path.name)
            continue
        district = parts[0]
        try:
            plan_year = int(parts[1])
        except ValueError:
            plan_year = 0

        pages = _extract_pages(pdf_path)
        if not pages:
            log.warning("No pages extracted from %s — skipping", pdf_path.name)
            continue

        if _is_scan_like(pages):
            log.warning(
                "%s appears to be a scan (almost no text); excluded from index",
                pdf_path.name,
            )
            continue

        doc_name = pdf_path.name
        chunks, used_fallback = _chunk_by_condition(pages, district, plan_year, doc_name)

        if used_fallback:
            fallback_districts.append(district)
            log.info("%s: used page-level fallback chunking", district)

        # De-duplicate chunk IDs within this document (shouldn't happen but safety)
        seen: set[str] = set()
        for i, ch in enumerate(chunks):
            while ch["chunk_id"] in seen:
                ch["chunk_id"] += "_dup"
            seen.add(ch["chunk_id"])

        all_chunks.extend(chunks)
        stats[district] = len(chunks)
        log.info("%s: %d chunks", district, len(chunks))

    # Summary
    log.info("=== Chunk summary ===")
    for d, n in sorted(stats.items()):
        fb = " [page-fallback]" if d in fallback_districts else ""
        log.info("  %s: %d chunks%s", d, n, fb)
    if fallback_districts:
        log.info("Page-level fallback used for: %s", ", ".join(fallback_districts))

    return all_chunks, fallback_districts


# ---------------------------------------------------------------------------
# Index management — ChromaDB + BM25 fallback
# ---------------------------------------------------------------------------


class _BM25Index:
    """In-memory BM25 index over chunks."""

    def __init__(self, chunks: list[dict]) -> None:
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        self._chunks = chunks
        tokenised = [ch["text"].lower().split() for ch in chunks]
        self._bm25 = BM25Okapi(tokenised)

    def query(
        self,
        query: str,
        district: str,
        crop: Optional[str],
        condition: Optional[str],
        k: int,
    ) -> list[dict]:
        filtered = [
            (i, ch) for i, ch in enumerate(self._chunks)
            if ch["district"] == district
            and (crop is None or crop.lower() in (ch.get("crop") or "").lower())
            and (condition is None or condition.lower() in ch["condition"].lower())
        ]
        if not filtered:
            return []

        indices = [i for i, _ in filtered]
        cands = [ch for _, ch in filtered]
        tokenised_q = query.lower().split()

        # Score only the filtered subset
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        subset_tokenised = [ch["text"].lower().split() for ch in cands]
        bm25_sub = BM25Okapi(subset_tokenised)
        scores = bm25_sub.get_scores(tokenised_q)

        ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)
        return [{"chunk": ch, "score": float(sc)} for sc, ch in ranked[:k]]


_chroma_client = None
_chroma_collection = None
_embed_fn = None
_bm25_index: Optional[_BM25Index] = None
_all_chunks: list[dict] = []
_using_bm25_fallback = False


def _load_embed_fn():
    """Load the sentence-transformer embedding function for ChromaDB."""
    try:
        from chromadb.utils.embedding_functions import (  # noqa: PLC0415
            SentenceTransformerEmbeddingFunction,
        )
        return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    except Exception as exc:
        log.warning("Could not load embedding function: %s", exc)
        return None


def _try_load_chroma():
    """Try to connect to the persisted Chroma collection.  Returns True on success."""
    global _chroma_client, _chroma_collection, _embed_fn
    try:
        import chromadb  # noqa: PLC0415

        _embed_fn = _load_embed_fn()
        if _embed_fn is None:
            return False

        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = _chroma_client.get_collection(
            name=CHROMA_COLLECTION,
            embedding_function=_embed_fn,
        )
        count = _chroma_collection.count()
        log.info("ChromaDB loaded: %d documents in collection", count)
        return True
    except Exception as exc:
        log.warning("ChromaDB not available (%s); using BM25 fallback", exc)
        return False


def _ensure_loaded() -> None:
    """Load index (Chroma or BM25 fallback) lazily on first retrieve call."""
    global _using_bm25_fallback, _bm25_index, _all_chunks

    if _chroma_collection is not None or _bm25_index is not None:
        return  # already loaded

    # Try Chroma first
    if _try_load_chroma():
        _using_bm25_fallback = False
        return

    # BM25 fallback — load chunks from PDFs
    log.info("Building BM25 in-memory index from PDFs (fallback mode)…")
    if not _all_chunks:
        _all_chunks, _ = build_chunks()
    if _all_chunks:
        _bm25_index = _BM25Index(_all_chunks)
        _using_bm25_fallback = True
    else:
        log.warning("No chunks available for BM25 index")


def build_index(chunks: Optional[list[dict]] = None) -> None:
    """Build (or rebuild) the ChromaDB index.  Called by --build."""
    global _chroma_client, _chroma_collection, _embed_fn, _all_chunks

    if chunks is None:
        chunks, fallback = build_chunks()
        _all_chunks = chunks

    if not chunks:
        log.error("No chunks to index")
        return

    import chromadb  # noqa: PLC0415

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    _embed_fn = _load_embed_fn()
    if _embed_fn is None:
        log.error("Cannot build Chroma index: embedding model failed to load")
        return

    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if present
    try:
        _chroma_client.delete_collection(CHROMA_COLLECTION)
        log.info("Deleted existing collection '%s'", CHROMA_COLLECTION)
    except Exception:
        pass

    _chroma_collection = _chroma_client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=_embed_fn,
    )

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        _chroma_collection.add(
            ids=[ch["chunk_id"] for ch in batch],
            documents=[ch["text"] for ch in batch],
            metadatas=[
                {
                    "doc": ch["doc"],
                    "district": ch["district"],
                    "plan_year": ch["plan_year"],
                    "condition": ch["condition"],
                    "crop": ch.get("crop") or "",
                    "page": ch["page"],
                }
                for ch in batch
            ],
        )
        log.info(
            "Indexed batch %d-%d / %d",
            i + 1, min(i + batch_size, len(chunks)), len(chunks),
        )

    log.info("Index built: %d chunks", len(chunks))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available_districts() -> list[dict]:
    """Return list of {district, plan_year} dicts for districts with a plan."""
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    result = []
    for pdf in pdfs:
        parts = pdf.stem.rsplit("_", 1)
        if len(parts) == 2:
            district = parts[0]
            try:
                year = int(parts[1])
            except ValueError:
                year = 0
            result.append({"district": district, "plan_year": year})
    return result


def retrieve(
    query: str,
    district: str,
    crop: Optional[str] = None,
    condition: Optional[str] = None,
    k: int = 5,
) -> list:
    """Retrieve up to k Passage objects for the given query and district.

    Parameters
    ----------
    query:
        Free-text query describing the agronomic situation.
    district:
        District name; accepted case-insensitively and with/without spaces.
        Example: 'Ranga Reddy', 'rangareddy', 'NALGONDA'.
    crop:
        Optional crop filter (e.g. 'cotton').  If None, no crop filter.
    condition:
        Optional condition filter (e.g. 'mid season drought').
    k:
        Maximum number of passages to return.

    Returns
    -------
    List of contracts.Passage objects.  Returns [] if:
    - The district has no plan.
    - No matching chunks found.
    - Any internal error occurs (logged).

    Never raises.
    """
    try:
        from contracts import Passage  # noqa: PLC0415

        district_key = _normalise_district(district)

        # Check the district has a PDF
        known = {d["district"] for d in available_districts()}
        if district_key not in known:
            log.info(
                "retrieve: district '%s' (normalised: '%s') has no plan — returning []",
                district, district_key,
            )
            return []

        _ensure_loaded()

        passages: list[Passage] = []

        if not _using_bm25_fallback and _chroma_collection is not None:
            # --- ChromaDB path ---
            where: dict = {"district": {"$eq": district_key}}
            if crop:
                where["crop"] = {"$contains": crop.lower()}  # type: ignore[assignment]
            if condition:
                where["condition"] = {"$contains": condition.lower()}  # type: ignore[assignment]

            # ChromaDB doesn't support $contains on metadata strings in all versions;
            # do a broader district-only filter and post-filter
            where = {"district": {"$eq": district_key}}

            try:
                results = _chroma_collection.query(
                    query_texts=[query],
                    n_results=min(k * 5, _chroma_collection.count()),
                    where=where,
                )
            except Exception as exc:
                log.warning("ChromaDB query failed (%s); falling back to BM25", exc)
                # Fall through to BM25
                results = None

            if results is not None:
                ids = results["ids"][0]
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                dists = results["distances"][0]

                for cid, doc_text, meta, dist in zip(ids, docs, metas, dists):
                    # Post-filter by crop and condition
                    if crop and crop.lower() not in (meta.get("crop") or "").lower():
                        continue
                    if condition and condition.lower() not in meta.get("condition", "").lower():
                        continue

                    score = max(0.0, 1.0 - dist)  # cosine distance → similarity

                    passages.append(
                        Passage(
                            chunk_id=cid,
                            doc=meta.get("doc", ""),
                            district=meta.get("district", district_key),
                            plan_year=meta.get("plan_year") or None,
                            condition=meta.get("condition", "general"),
                            crop=meta.get("crop") or None,
                            page=meta.get("page", 0),
                            text=doc_text,
                            score=score,
                        )
                    )
                    if len(passages) >= k:
                        break

                return passages

        # --- BM25 fallback path ---
        if _bm25_index is None:
            log.warning("No index available (neither Chroma nor BM25)")
            return []

        results = _bm25_index.query(query, district_key, crop, condition, k)
        for r in results:
            ch = r["chunk"]
            passages.append(
                Passage(
                    chunk_id=ch["chunk_id"],
                    doc=ch["doc"],
                    district=ch["district"],
                    plan_year=ch.get("plan_year") or None,
                    condition=ch.get("condition", "general"),
                    crop=ch.get("crop") or None,
                    page=ch.get("page", 0),
                    text=ch["text"],
                    score=r["score"],
                )
            )
        return passages

    except Exception as exc:
        log.error("retrieve() error: %s", exc, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_query(query: str, district: str, k: int = 5) -> None:
    passages = retrieve(query, district, k=k)
    if not passages:
        print(f"No passages found for district '{district}'.")
        return
    for p in passages:
        print(f"chunk_id  : {p.chunk_id}")
        print(f"page      : {p.page}")
        print(f"condition : {p.condition}")
        print(f"plan_year : {p.plan_year}")
        print(f"score     : {p.score:.4f}")
        print(f"text      : {p.text[:300]}")
        print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BreakWatch RAG — ICAR contingency plan retrieval"
    )
    parser.add_argument("query", nargs="?", default=None, help="Query string")
    parser.add_argument("--district", default=None, help="District name")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--build",
        action="store_true",
        help="(Re)build the ChromaDB index from PDFs and exit",
    )
    parser.add_argument(
        "--list-districts",
        action="store_true",
        help="List districts with plans",
    )
    args = parser.parse_args()

    if args.build:
        chunks, fallback = build_chunks()
        build_index(chunks)
        log.info("Build complete.")
        # Print summary table
        print("\n=== Chunks per district ===")
        from collections import Counter  # noqa: PLC0415

        counts = Counter(ch["district"] for ch in chunks)
        for d, n in sorted(counts.items()):
            fb = " [page-fallback]" if d in fallback else ""
            print(f"  {d}: {n}{fb}")
        return

    if args.list_districts:
        for d in available_districts():
            print(f"  {d['district']}: plan_year={d['plan_year']}")
        return

    if args.query is None or args.district is None:
        parser.print_help()
        sys.exit(1)

    _cli_query(args.query, args.district, k=args.k)


if __name__ == "__main__":
    main()

