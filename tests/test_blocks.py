import json

from markdown_classification.blocks import (
    build_page_windows,
    extract_json_object,
    materialize_classifications,
    split_markdown_blocks,
)


def page(number: int, markdown: str):
    return {
        "page_number": number,
        "document_id": f"book-01/chapter-01/page-{number:03d}",
        "markdown_content": markdown,
        "content_hash": f"hash-{number}",
        "image_references": [],
    }


def test_blocks_are_stable_and_front_matter_is_removed():
    blocks = split_markdown_blocks("doc-1", '---\nsourceBlob: "x"\n---\n\n# Title\n\nQuestion')
    assert [item["block_id"] for item in blocks] == ["doc-1#b001", "doc-1#b002"]
    assert blocks[0]["markdown"] == "# Title"


def test_windows_have_no_duplicate_target_pages():
    windows = build_page_windows([page(i, f"Page {i}") for i in range(1, 9)], 5)
    assert [len(item["source_document_ids"]) for item in windows] == [5, 3]
    ids = [document_id for window in windows for document_id in window["source_document_ids"]]
    assert len(ids) == len(set(ids)) == 8


def test_json_code_fence_is_accepted():
    assert extract_json_object('```json\n{"sections": []}\n```') == {"sections": []}


def test_one_page_one_exercise_preserves_original_markdown():
    source = page(1, "# Exercise\n\nFind x.\n\n![diagram](figures/figure-1.png)")
    source["image_references"] = [{"original_link": "figures/figure-1.png", "blob_path": "generated/figure-1.png"}]
    window = build_page_windows([source], 5)[0]
    pages = json.loads(window["pages_json"])
    block_ids = [block["block_id"] for block in pages[0]["blocks"]]
    result = materialize_classifications(window["pages_json"], json.dumps({
        "sections": [{
            "section_id": "exercise-1", "content_type": "EXERCISE",
            "title": "Find x", "knowledge_id": "angles", "confidence": 0.95,
            "block_ids": block_ids,
        }]
    }))
    assert result["status"] == "SUCCEEDED"
    assert result["sections"][0]["preserved_original"] is True
    assert result["sections"][0]["markdown_content"] == source["markdown_content"]


def test_unassigned_blocks_are_preserved_as_other():
    window = build_page_windows([page(1, "Knowledge\n\nExample")], 5)[0]
    pages = json.loads(window["pages_json"])
    first = pages[0]["blocks"][0]["block_id"]
    result = materialize_classifications(window["pages_json"], json.dumps({
        "sections": [{
            "section_id": "knowledge-1", "content_type": "KNOWLEDGE",
            "title": "Knowledge", "knowledge_id": "k1", "confidence": 0.9,
            "block_ids": [first],
        }]
    }))
    assert result["status"] == "SUCCEEDED"
    assert [item["content_type"] for item in result["sections"]] == ["KNOWLEDGE", "OTHER"]


def test_single_exercise_page_is_preserved_inside_multi_page_window():
    first = page(1, "# Knowledge\n\nAngles add to 180 degrees.")
    second = page(2, "# Exercise\n\nFind x.")
    window = build_page_windows([first, second], 5)[0]
    pages = json.loads(window["pages_json"])
    first_ids = [block["block_id"] for block in pages[0]["blocks"]]
    second_ids = [block["block_id"] for block in pages[1]["blocks"]]
    result = materialize_classifications(window["pages_json"], json.dumps({"sections": [
        {"section_id": "knowledge-1", "content_type": "KNOWLEDGE", "title": "Angles",
         "knowledge_id": "angles", "confidence": 0.9, "block_ids": first_ids},
        {"section_id": "exercise-1", "content_type": "EXERCISE", "title": "Find x",
         "knowledge_id": "angles", "confidence": 0.9, "block_ids": second_ids},
    ]}))
    assert result["status"] == "SUCCEEDED"
    assert result["sections"][1]["preserved_original"] is True
    assert result["sections"][1]["markdown_content"] == second["markdown_content"]
