from markdown_classification.blocks import split_markdown_blocks, build_page_windows, materialize_classifications
from markdown_classification.prompt_loader import classification_prompt
import json


def test_english_prompt_has_clear_categories():
    text = classification_prompt(5)
    for key in ("KNOWLEDGE", "EXAMPLE", "EXERCISE", "OTHER", "block_id"):
        assert key in text
    assert "at most 5 pages" in text


def test_heading_without_blank_line_splits():
    text = "# Geometry\nA line goes forever.\n## Worked Example\nFind x.\nSolution: x = 3.\n## Practice\n1. Find y.\n2. Find z."
    blocks = split_markdown_blocks("doc", text)
    assert len(blocks) >= 5
    assert any("2. Find z." == b["markdown"] for b in blocks)


def test_assign_all_mixed_type_blocks():
    page = {"document_id": "doc", "page_number": 1, "markdown_content": "# Knowledge\n\nA theorem.\n\n# Example\n\n3 + 4 = 7\n\n# Practice\n\n1. Find x.", "image_references": []}
    w = build_page_windows([page], 1)[0]
    blocks = json.loads(w["pages_json"])[0]["blocks"]
    groups = [[blocks[i]["block_id"] for i in indices] for indices in ((0,1),(2,3),(4,5))]
    result = materialize_classifications(w["pages_json"], json.dumps({"sections": [{"section_id": str(i), "content_type": ty, "block_ids": ids, "confidence": 0.9} for i, (ty, ids) in enumerate(zip(("KNOWLEDGE","EXAMPLE","EXERCISE"),groups))]}))
    assert result["status"] == "SUCCEEDED", result["error"]
    assert [section["content_type"] for section in result["sections"]] == ["KNOWLEDGE","EXAMPLE","EXERCISE"]
