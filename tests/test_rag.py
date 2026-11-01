"""
tests/test_rag.py — Tests for ml/rag.py.

Tests run on the real PDFs in data/corpus/icar/.  If the PDFs are absent,
every test is skipped with a clear message.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CORPUS_DIR = REPO_ROOT / "data" / "corpus" / "icar"

_PDFS_PRESENT = any(CORPUS_DIR.glob("*.pdf")) if CORPUS_DIR.exists() else False
_SKIP_MSG = (
    f"PDFs absent in {CORPUS_DIR} — run `python -m ml.rag --build` "
    "after placing the ICAR PDF files there."
)

pytestmark = pytest.mark.skipif(not _PDFS_PRESENT, reason=_SKIP_MSG)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chunks():
    """Build chunks from real PDFs (module-scoped to avoid re-parsing)."""
    from ml.rag import build_chunks  # noqa: PLC0415

    all_chunks, fallback = build_chunks()
    assert len(all_chunks) > 0, "build_chunks() returned no chunks"
    return all_chunks, fallback


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chunks_non_empty(chunks):
    """build_chunks() must produce at least one chunk per district PDF."""
    all_chunks, _ = chunks
    assert len(all_chunks) > 0


def test_chunk_ids_unique(chunks):
    """Every chunk_id must be globally unique."""
    all_chunks, _ = chunks
    ids = [ch["chunk_id"] for ch in all_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"


def test_chunk_required_fields(chunks):
    """Every chunk must have the required metadata fields."""
    all_chunks, _ = chunks
    required = {"chunk_id", "doc", "district", "plan_year", "condition", "page", "text"}
    for ch in all_chunks:
        missing = required - set(ch.keys())
        assert not missing, f"Chunk {ch.get('chunk_id')} missing fields: {missing}"
        assert isinstance(ch["text"], str) and ch["text"].strip(), (
            f"Chunk {ch['chunk_id']} has empty text"
        )


def test_retrieve_returns_passages(chunks):
    """retrieve() must return Passage objects with non-empty text."""
    from contracts import Passage  # noqa: PLC0415
    from ml.rag import retrieve  # noqa: PLC0415

    passages = retrieve("mid season drought cotton", "nalgonda", k=3)
    assert isinstance(passages, list)
    assert len(passages) > 0, "Expected at least one passage for Nalgonda"
    for p in passages:
        assert isinstance(p, Passage), f"Expected Passage, got {type(p)}"
        assert p.text.strip(), "Passage text must be non-empty"
        assert p.district == "nalgonda"
        assert p.chunk_id, "Passage must have a chunk_id"


def test_retrieve_k_respected(chunks):
    """retrieve() must return at most k passages."""
    from ml.rag import retrieve  # noqa: PLC0415

    for k in (1, 2, 5):
        passages = retrieve("drought contingency measures", "khammam", k=k)
        assert len(passages) <= k, f"Expected <= {k} passages, got {len(passages)}"


def test_retrieve_district_no_plan():
    """A district with no plan must return [] and never raise."""
    from ml.rag import retrieve  # noqa: PLC0415

    # Hyderabad has no ICAR contingency PDF in corpus
    result = retrieve("drought", "Hyderabad", k=5)
    assert result == [], f"Expected [] for Hyderabad, got {result}"

    # Completely unknown district
    result2 = retrieve("flood", "NonExistentDistrict", k=5)
    assert result2 == []


def test_retrieve_case_insensitive(chunks):
    """District matching must be case-insensitive and handle spaces."""
    from ml.rag import retrieve  # noqa: PLC0415

    variants = ["Nalgonda", "NALGONDA", "nalgonda"]
    results = [retrieve("drought", v, k=3) for v in variants]
    # All should return the same number of passages (same district)
    sizes = [len(r) for r in results]
    assert len(set(sizes)) == 1 or all(s > 0 for s in sizes), (
        f"Case-insensitive district matching inconsistent: {sizes}"
    )
    # Ranga Reddy with space
    rr = retrieve("drought", "Ranga Reddy", k=3)
    rr2 = retrieve("drought", "rangareddy", k=3)
    assert len(rr) == len(rr2), "Space-insensitive matching failed for Ranga Reddy"


def test_retrieve_never_raises():
    """retrieve() must never raise, even on bad inputs."""
    from ml.rag import retrieve  # noqa: PLC0415

    # Empty query
    result = retrieve("", "nalgonda", k=5)
    assert isinstance(result, list)

    # Very long nonsense query
    result2 = retrieve("x " * 1000, "adilabad", k=1)
    assert isinstance(result2, list)

    # k=0
    result3 = retrieve("drought", "nalgonda", k=0)
    assert isinstance(result3, list)


def test_available_districts():
    """available_districts() must list all districts with a PDF."""
    from ml.rag import available_districts  # noqa: PLC0415

    districts = available_districts()
    assert isinstance(districts, list)
    assert len(districts) == 9  # 9 PDFs in corpus
    names = {d["district"] for d in districts}
    expected = {
        "adilabad", "karimnagar", "khammam", "mahabubnagar", "medak",
        "nalgonda", "nizamabad", "rangareddy", "warangal",
    }
    assert names == expected, f"Expected {expected}, got {names}"
    for d in districts:
        assert d["plan_year"] == 2011

