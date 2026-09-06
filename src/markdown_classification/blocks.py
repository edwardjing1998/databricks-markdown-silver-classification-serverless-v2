from __future__ import annotations

import hashlib
import json
import re
from typing import Any

IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
ALLOWED_TYPES = {"KNOWLEDGE", "EXAMPLE", "EXERCISE", "OTHER"}


def strip_front_matter(markdown: str) -> str:
    return re.sub(r"\A\s*---\s*\n.*?\n---\s*\n", "", markdown or "", count=1, flags=re.S)


def split_markdown_blocks(document_id: str, markdown: str) -> list[dict[str, Any]]:
    """Create stable, ordered blocks without asking AI to reproduce Markdown."""
    body = strip_front_matter(markdown)
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", body) if chunk.strip()]
    return [
        {
            "block_id": f"{document_id}#b{index:03d}",
            "block_index": index,
            "markdown": chunk,
            "image_links": IMAGE_PATTERN.findall(chunk),
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def build_page_windows(pages: list[Any] | None, max_pages: int) -> list[dict[str, Any]]:
    normalized = []
    for page in pages or []:
        value = page.asDict(recursive=True) if hasattr(page, "asDict") else dict(page)
        value["blocks"] = split_markdown_blocks(value["document_id"], value.get("markdown_content") or "")
        normalized.append(value)
    normalized.sort(key=lambda item: (item.get("page_number") or 0, item["document_id"]))
    windows = []
    for start in range(0, len(normalized), max_pages):
        selected = normalized[start:start + max_pages]
        source_ids = [item["document_id"] for item in selected]
        source_material = "|".join(
            f"{item['document_id']}:{item.get('content_hash') or item.get('source_content_hash') or ''}"
            for item in selected
        )
        windows.append({
            "window_number": len(windows) + 1,
            "first_page": selected[0].get("page_number"),
            "last_page": selected[-1].get("page_number"),
            "source_document_ids": source_ids,
            "source_hash": hashlib.sha256(source_material.encode()).hexdigest(),
            "pages_json": json.dumps(selected, ensure_ascii=False, default=str),
            "input_characters": sum(len(item.get("markdown_content") or "") for item in selected),
        })
    return windows


def extract_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        raise ValueError("AI returned an empty result")
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", value.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("AI result does not contain a JSON object")
    return json.loads(cleaned[start:end + 1])


def materialize_classifications(pages_json: str, raw_result: str | None) -> dict[str, Any]:
    try:
        pages = json.loads(pages_json)
        response = extract_json_object(raw_result)
        sections = response.get("sections")
        if not isinstance(sections, list) or not sections:
            raise ValueError("AI result must contain a non-empty sections array")

        ordered_blocks = [
            {**block, "document_id": page["document_id"], "page_number": page.get("page_number")}
            for page in pages for block in page.get("blocks", [])
        ]
        block_by_id = {block["block_id"]: block for block in ordered_blocks}
        section_count_by_document: dict[str, int] = {}
        for candidate in sections:
            candidate_ids = candidate.get("block_ids") or []
            candidate_documents = {
                block_by_id[block_id]["document_id"]
                for block_id in candidate_ids if block_id in block_by_id
            }
            for document_id in candidate_documents:
                section_count_by_document[document_id] = section_count_by_document.get(document_id, 0) + 1
        used: set[str] = set()
        output = []

        for position, section in enumerate(sections, start=1):
            content_type = str(section.get("content_type", "")).upper()
            block_ids = section.get("block_ids")
            if content_type not in ALLOWED_TYPES:
                raise ValueError(f"Unsupported content_type: {content_type}")
            if not isinstance(block_ids, list) or not block_ids:
                raise ValueError("Every section must have at least one block_id")
            unknown = [block_id for block_id in block_ids if block_id not in block_by_id]
            if unknown:
                raise ValueError(f"AI returned unknown block IDs: {unknown[:3]}")
            duplicate = [block_id for block_id in block_ids if block_id in used]
            if duplicate:
                raise ValueError(f"Blocks were assigned more than once: {duplicate[:3]}")
            used.update(block_ids)
            chosen = [block for block in ordered_blocks if block["block_id"] in set(block_ids)]
            markdown_content = "\n\n".join(block["markdown"] for block in chosen)
            source_ids = list(dict.fromkeys(block["document_id"] for block in chosen))
            page_numbers = list(dict.fromkeys(block["page_number"] for block in chosen))
            image_links = list(dict.fromkeys(link for block in chosen for link in block.get("image_links", [])))
            references = []
            for page in pages:
                if page["document_id"] not in source_ids:
                    continue
                for reference in page.get("image_references") or []:
                    if not image_links or reference.get("original_link") in image_links:
                        references.append(reference)
            preserve_original = (
                content_type == "EXERCISE"
                and len(source_ids) == 1
                and section_count_by_document.get(source_ids[0]) == 1
                and set(block_ids) == {
                    block["block_id"] for block in ordered_blocks
                    if block["document_id"] == source_ids[0]
                }
            )
            if preserve_original:
                markdown_content = next(page["markdown_content"] for page in pages if page["document_id"] == source_ids[0])
            confidence = max(0.0, min(1.0, float(section.get("confidence", 0.0))))
            section_id = str(section.get("section_id") or f"section-{position}")
            output.append({
                "section_id": section_id,
                "content_type": content_type,
                "title": str(section.get("title") or section_id),
                "knowledge_id": str(section.get("knowledge_id") or "unassigned"),
                "confidence": confidence,
                "block_ids": block_ids,
                "source_document_ids": source_ids,
                "source_page_numbers": page_numbers,
                "markdown_content": markdown_content,
                "image_links": image_links,
                "image_references_json": json.dumps(references, ensure_ascii=False, default=str),
                "preserved_original": preserve_original,
            })

        missing = [block["block_id"] for block in ordered_blocks if block["block_id"] not in used]
        if missing:
            chosen = [block for block in ordered_blocks if block["block_id"] in set(missing)]
            output.append({
                "section_id": "unclassified-remainder",
                "content_type": "OTHER",
                "title": "Unclassified content",
                "knowledge_id": "unassigned",
                "confidence": 0.0,
                "block_ids": missing,
                "source_document_ids": list(dict.fromkeys(x["document_id"] for x in chosen)),
                "source_page_numbers": list(dict.fromkeys(x["page_number"] for x in chosen)),
                "markdown_content": "\n\n".join(x["markdown"] for x in chosen),
                "image_links": list(dict.fromkeys(link for x in chosen for link in x.get("image_links", []))),
                "image_references_json": "[]",
                "preserved_original": False,
            })
        return {"status": "SUCCEEDED", "error": None, "sections": output}
    except Exception as error:
        return {"status": "FAILED", "error": str(error)[:2000], "sections": []}
