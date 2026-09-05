from markdown_silver.windowing import build_page_windows


def pages(count: int, size: int = 10):
    return [
        {"document_id": f"page-{number:03d}", "page_number": number,
         "markdown_content": "x" * size}
        for number in range(1, count + 1)
    ]


def test_forty_five_pages_use_overlapping_windows():
    windows = build_page_windows(pages(45), 15, 2, 40000)
    assert [(w["first_page"], w["last_page"]) for w in windows] == [
        (1, 15), (14, 28), (27, 41), (40, 45)
    ]


def test_character_limit_shortens_window_and_preserves_overlap():
    windows = build_page_windows(pages(6, size=600), 15, 1, 1500)
    assert [w["source_document_ids"] for w in windows] == [
        ["page-001", "page-002"],
        ["page-002", "page-003"],
        ["page-003", "page-004"],
        ["page-004", "page-005"],
        ["page-005", "page-006"],
    ]


def test_single_oversized_page_is_not_dropped():
    windows = build_page_windows(pages(1, size=5000), 15, 2, 1000)
    assert windows[0]["source_document_ids"] == ["page-001"]
    assert windows[0]["input_characters"] == 5000
