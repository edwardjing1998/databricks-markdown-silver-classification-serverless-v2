import json
import pytest
from markdown_classification.bible_scenes import build_page_input
from markdown_classification.layout_input import compact_layout, PROCESSOR_VERSION


def test_bible_layout_included():
    result = json.loads(build_page_input("id", 1, "Hello", json.dumps({"pages": []})))
    assert result["layout"] == {"pages": []}


def test_old_input_still_supported():
    assert json.loads(build_page_input("id", 1, "Hello"))["layout"] is None


def test_compact_huge_raw_layout_preserves_figures_and_coordinates():
    raw = {"analyzeResult": {
        "content": "OCR text " * 100_000,
        "pages": [{"pageNumber": 1, "width": 800, "height": 1200, "unit": "pixel",
                   "words": [{"content": "unused"}] * 100}],
        "figures": [{"id": "1.1", "boundingRegions": [{"pageNumber": 1,
                        "polygon": [0, 0, 100, 0, 100, 200, 0, 200]}]}],
        "paragraphs": [{"content": "Caption " + "x" * 20000,
                        "spans": [{"offset": 10, "length": 5}],
                        "boundingRegions": [{"pageNumber": 1, "polygon": [1, 2, 3, 4]}]}],
    }}
    compact = json.loads(compact_layout(json.dumps(raw)))
    assert compact["layout_processor"] == PROCESSOR_VERSION
    assert compact["figures"][0]["id"] == "1.1"
    assert compact["paragraphs"][0]["text_anchor"].startswith("Caption")
    assert "words" not in json.dumps(compact)
    assert len(json.dumps(compact)) < 2000


def test_malformed_layout_fails_closed():
    with pytest.raises(ValueError):
        compact_layout("[]")


def test_compact_limit_never_silently_drops_regions():
    with pytest.raises(ValueError, match="Compacted layout exceeds"):
        compact_layout({"paragraphs": [{"content": str(i)} for i in range(100)]},
                       max_compact_bytes=100)


def test_figures_caption_and_tables_are_preserved():
    data = {"figures": [{"id": "f", "caption": {"content": "Hannah",
            "boundingRegions": [{"pageNumber": 1, "polygon": [0, 1]}]}}],
            "tables": [{"rowCount": 2, "columnCount": 2}]}
    compact = json.loads(compact_layout(data))
    assert compact["figures"][0]["caption"]["content"] == "Hannah"
    assert compact["tables"][0]["columnCount"] == 2
