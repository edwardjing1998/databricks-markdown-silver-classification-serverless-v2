from __future__ import annotations

import json
from typing import Any


def build_page_windows(
    pages: list[Any] | None,
    max_pages: int,
    overlap: int,
    max_characters: int,
) -> list[dict[str, Any]]:
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    if overlap < 0 or overlap >= max_pages:
        raise ValueError("overlap must be from 0 through max_pages - 1")
    if max_characters < 1000:
        raise ValueError("max_characters must be at least 1000")

    ordered = list(pages or [])
    results: list[dict[str, Any]] = []
    start = 0
    while start < len(ordered):
        selected: list[Any] = []
        characters = 0
        cursor = start
        while cursor < len(ordered) and len(selected) < max_pages:
            page = ordered[cursor]
            markdown = page["markdown_content"] or ""
            if selected and characters + len(markdown) > max_characters:
                break
            selected.append(page)
            characters += len(markdown)
            cursor += 1

        payload = [
            {
                "document_id": page["document_id"],
                "page_number": page["page_number"],
                "markdown": page["markdown_content"] or "",
            }
            for page in selected
        ]
        results.append(
            {
                "window_number": len(results) + 1,
                "first_page": selected[0]["page_number"],
                "last_page": selected[-1]["page_number"],
                "source_document_ids": [page["document_id"] for page in selected],
                "pages_json": json.dumps(payload, ensure_ascii=False),
                "input_characters": characters,
            }
        )
        if cursor >= len(ordered):
            break
        start = max(start + 1, cursor - min(overlap, len(selected) - 1))
    return results
