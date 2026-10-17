import json
from markdown_classification.blocks import build_page_windows, materialize_classifications


def _page(number, markdown):
    return {"document_id": f"d{number}", "page_number": number,
            "markdown_content": markdown, "image_references": []}


def _classify(pages):
    w = build_page_windows(pages, 5)[0]
    payload = json.loads(w["pages_json"])
    ids = [[b["block_id"] for b in p["blocks"]] for p in payload]
    response = {"sections": [
        {"section_id": "question", "content_type": "EXERCISE", "confidence": .9,
         "block_ids": [ids[0][0]]},
        {"section_id": "figure", "content_type": "OTHER", "confidence": .7,
         "block_ids": [ids[-1][-1]]},
    ]}
    return materialize_classifications(w["pages_json"], json.dumps(response))


def test_same_page_diagram_stays_with_exercise():
    result = _classify([_page(1, "Find angle x in the diagram below.\n\n![triangle](figures/f.png)")])
    assert result["status"] == "SUCCEEDED", result["error"]
    assert len(result["sections"]) == 1
    assert "figures/f.png" in result["sections"][0]["image_links"]


def test_next_page_figure_linked():
    result = _classify([_page(1, "As shown in the figure, calculate x."),
                        _page(2, "![triangle](figures/f.png)")])
    assert result["status"] == "SUCCEEDED", result["error"]
    assert result["sections"][0]["source_page_numbers"] == [1, 2]


def test_unrelated_next_page_figure_is_not_linked():
    result = _classify([_page(1, "Solve 5 + 7."),
                        _page(2, "![picture](figures/f.png)")])
    assert result["status"] == "SUCCEEDED"
    assert len(result["sections"]) == 2


def test_algebra_graph_not_geometry_specific():
    result = _classify([_page(1, "Use the graph below to find the intercept.\n\n![graph](figures/graph.png)")])
    assert result["status"] == "SUCCEEDED"
    assert len(result["sections"]) == 1
