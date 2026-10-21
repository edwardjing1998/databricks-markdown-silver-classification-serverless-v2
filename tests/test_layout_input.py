import json
import pytest
from markdown_classification.bible_scenes import build_page_input
from markdown_classification.layout_input import compact_layout, PROCESSOR_VERSION, resolve_layout_key, expected_layout_path, permitted_scan_root


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



def test_actual_azure_di_wrapper_three_figures_and_seven_paragraphs():
    polygons = [
        [184, 481, 1261, 481, 1261, 1350, 184, 1350],
        [1321, 478, 2398, 478, 2398, 1359, 1321, 1358],
        [176, 1868, 2383, 1870, 2381, 2747, 173, 2745],
    ]
    texts = [
        "Eve gave Adam some of the fruit. He also chose to eat it.",
        "Genesis 3:6-7; Moses 4:12",
        "God and the Lord visited them, but Adam and Eve were afraid and hid.",
        "Genesis 3:8-13; Moses 4:13-14",
        "Adam and Eve told God that they chose to eat the fruit.",
        "Genesis 3:16-24; Moses 4:15-31",
        "Adam and Eve 15",
    ]
    wrapped = {"status": "succeeded", "analyzeResult": {
        "pages": [{"pageNumber": 1, "width": 2550, "height": 3300, "unit": "pixel"}],
        "figures": [{"id": f"1.{i}", "boundingRegions": [{"pageNumber": 1, "polygon": polygon}]}
                    for i, polygon in enumerate(polygons, 1)],
        "paragraphs": [{"content": v, "role": "pageFooter" if i == 6 else None,
                        "boundingRegions": [{"pageNumber": 1, "polygon": [1, 2, 3, 2, 3, 4, 1, 4]}]}
                       for i, v in enumerate(texts)],
        "content": "full original OCR text " * 100000,
    }}
    parsed = json.loads(compact_layout(json.dumps(wrapped).encode("utf-8")))
    assert len(parsed["pages"]) == 1
    assert [f["id"] for f in parsed["figures"]] == ["1.1", "1.2", "1.3"]
    assert len(parsed["paragraphs"]) == 7
    assert parsed["figures"][2]["boundingRegions"][0]["polygon"] == polygons[2]
    assert parsed["paragraphs"][6]["role"] == "pageFooter"
    assert "full original OCR text" not in json.dumps(parsed)


def test_utf8_bom_supported():
    payload = b"\xef\xbb\xbf" + json.dumps({"analyzeResult": {"figures": [{"id": "1.1"}]}}).encode()
    assert json.loads(compact_layout(payload))["figures"][0]["id"] == "1.1"


def test_math_uuid_source_uses_canonical_document_id():
    assert resolve_layout_key(
        "adam_math_v2_cleansed/maths_time_2/maths-time-2_20260923215214283",
        "f6acee40-3e86-40ae-84b8-9585892cd141/adam_math_v2_cleansed/maths_time_2/maths-time-2_20260923215214283.jpeg"
    ) == "adam_math_v2_cleansed/maths_time_2/maths-time-2_20260923215214283"


def test_math_nested_source_prefers_document_id_over_intermediate_directories():
    assert resolve_layout_key(
        "adam_math_v2/maths_calculation/maths-calculation-1",
        "c95b095e-131a-4e46-87e0-e7a2738b66c8/adam_math_v2_cleansed_v2/adam_math_v2/maths_calculation/maths-calculation-1.jpeg"
    ) == "adam_math_v2/maths_calculation/maths-calculation-1"


def test_source_blob_fallback_handles_multiple_directory_depths():
    assert resolve_layout_key(None, "uuid/series/book/chapter/page.jpeg") == "book/chapter/page"


def test_bible_document_id_remains_unchanged():
    assert resolve_layout_key("bible_studies/old_testament_stories/page_20261019191024512") == (
        "bible_studies/old_testament_stories/page_20261019191024512"
    )


@pytest.mark.parametrize("markdown_path,layout_path", [
    (
        "abfss://education@account.dfs.core.windows.net/generated/book-01/chapter-01/page-07/content.md",
        "abfss://education@account.dfs.core.windows.net/generated/book-01/chapter-01/page-07/layout.json",
    ),
    (
        "abfss://education@account.dfs.core.windows.net/generated/adam-maths-course/chapter-6/book-01/chapter-01/page-07/content.md",
        "abfss://education@account.dfs.core.windows.net/generated/adam-maths-course/chapter-6/book-01/chapter-01/page-07/layout.json",
    ),
    (
        "abfss://education@account.dfs.core.windows.net/generated/bible_studies/old_testament_stories/page_20261019191024512/content.md",
        "abfss://education@account.dfs.core.windows.net/generated/bible_studies/old_testament_stories/page_20261019191024512/layout.json",
    ),
])
def test_expected_layout_path_exact_sibling(markdown_path, layout_path):
    assert expected_layout_path(markdown_path) == layout_path


def test_math_nested_paths_remain_distinct():
    root = "abfss://education@account.dfs.core.windows.net/generated/"
    simple = expected_layout_path(root + "book-01/chapter-01/page-07/content.md")
    nested = expected_layout_path(root + "adam-maths-course/chapter-6/book-01/chapter-01/page-07/content.md")
    assert simple != nested


@pytest.mark.parametrize("bad_path", [None, "", "book/chapter/page.jpeg", "generated/book/chapter/page/file.txt"])
def test_bad_markdown_source_path_rejected(bad_path):
    assert expected_layout_path(bad_path) == ""


def test_standard_and_uuid_generated_roots():
    root = "abfss://education@account.dfs.core.windows.net/generated"
    normal = root + "/adam-maths-course/chapter-1/page-01/content.md"
    uuid_root = "abfss://education@account.dfs.core.windows.net/f6acee40-3e86-40ae-84b8-9585892cd141/generated"
    uuid_page = uuid_root + "/adam_math_v2_cleansed/maths_time_2/page/content.md"
    assert permitted_scan_root(normal, root) == root
    assert permitted_scan_root(uuid_page, root) == uuid_root
    assert expected_layout_path(uuid_page) == uuid_root + "/adam_math_v2_cleansed/maths_time_2/page/layout.json"


def test_reject_other_accounts_and_arbitrary_prefixes():
    root = "abfss://education@account.dfs.core.windows.net/generated"
    assert permitted_scan_root("abfss://other@account.dfs.core.windows.net/generated/a/b/content.md", root) == ""
    assert permitted_scan_root("abfss://education@other.dfs.core.windows.net/generated/a/b/content.md", root) == ""
    assert permitted_scan_root("abfss://education@account.dfs.core.windows.net/arbitrary/generated/a/b/content.md", root) == ""


def test_bible_path_does_not_require_uuid():
    root = "abfss://education@account.dfs.core.windows.net/generated"
    bible = root + "/bible_studies/old_testament_stories/page/content.md"
    assert permitted_scan_root(bible, root) == root


def test_legacy_bible_entrypoint_preserved():
    from pathlib import Path
    code = Path(__file__).parents[1].joinpath("src/markdown_classification/bible_job.py").read_text()
    assert '.select("document_id","book_id","chapter_id","page_number","markdown_content")' in code


def test_optional_exact_layout_only_errors_for_malformed_present_files():
    """Regression guard: a missing sibling layout must not abort classification."""
    import inspect
    from markdown_classification.layout_input import attach_layout
    source = inspect.getsource(attach_layout)
    assert 'result.filter(F.col("_layout_error").isNotNull())' in source
    assert '.load(scan_roots)' in source


def test_optional_legacy_bible_layout_only_errors_for_malformed_present_files():
    """Bible's legacy call must continue accepting absent optional layouts."""
    import inspect
    from markdown_classification.layout_input import _attach_legacy_layout
    source = inspect.getsource(_attach_legacy_layout)
    assert '.filter(F.col("_layout_error").isNotNull())' in source
