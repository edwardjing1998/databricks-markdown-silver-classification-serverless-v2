"""Isolated Bible illustrated-scene extraction and AI response validation.

No mathematics classifier imports or behavior are changed.
"""
from __future__ import annotations
import hashlib
import json
from importlib.resources import files
from markdown_classification.blocks import split_markdown_blocks, extract_json_object

SCENE_TYPES = frozenset({"SCRIPTURE_STORY", "SCRIPTURE_REFERENCE", "DOCTRINAL_TEACHING", "LIFE_APPLICATION", "DISCUSSION_QUESTION", "PRAYER_REFLECTION", "TABLE_OF_CONTENTS", "OTHER"})


def build_page_input(document_id: str, page_number: int | None, markdown: str, layout_json: str | None = None) -> str:
    blocks = split_markdown_blocks(document_id, markdown)
    return json.dumps({"document_id": document_id, "page_number": page_number,
                       "blocks": blocks, "layout": json.loads(layout_json) if layout_json else None}, ensure_ascii=False)


def bible_prompt() -> str:
    return (files("markdown_classification.prompts") / "bible_scene_classifier_v1.txt").read_text(encoding="utf-8")


def validate_scenes(page_json: str, ai_result: str | None) -> dict:
    """Fail closed for invented/duplicated image links and block references."""
    try:
        source = json.loads(page_json)
        parsed = extract_json_object(ai_result)
        scenes = parsed.get("scenes")
        if not isinstance(scenes, list):
            raise ValueError("AI response must have scenes array")
        blocks = source["blocks"]
        by_id = {b["block_id"]: b for b in blocks}
        image_order = list(dict.fromkeys(link for b in blocks for link in b["image_links"]))
        # A model may select an image but omit the Markdown block containing it.
        # Recover source ownership deterministically, without inventing links.
        image_sources: dict[str, list[str]] = {}
        for block in blocks:
            for link in block["image_links"]:
                image_sources.setdefault(link, []).append(block["block_id"])
        used_blocks: set[str] = set()
        used_images: set[str] = set()
        used_ids: set[str] = set()
        materialized = []
        for i, candidate in enumerate(scenes, 1):
            sid = str(candidate.get("scene_id") or f"scene-{i:02d}")
            kind = str(candidate.get("scene_type") or "").upper()
            ids = candidate.get("block_ids")
            links = candidate.get("image_links", [])
            if sid in used_ids or kind not in SCENE_TYPES:
                raise ValueError("Duplicate scene_id or unsupported scene_type")
            if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or any(x not in by_id or x in used_blocks for x in ids):
                raise ValueError("Invalid, unknown or duplicate block_ids")
            if not isinstance(links, list) or len(links) != len(set(links)) or any(x not in image_order or x in used_images for x in links):
                raise ValueError("Unknown or duplicate image link")
            # Ensure each claimed image belongs to the scene's assigned source
            # blocks. A missing image-only block may be attached once, but never
            # steal a block already assigned to another scene.
            assigned_ids = set(ids)
            for link in links:
                if any(link in by_id[bid]["image_links"] for bid in assigned_ids):
                    continue
                owners = image_sources[link]  # link already checked above
                available = [bid for bid in owners if bid not in used_blocks]
                if not available:
                    raise ValueError("Image source block already assigned to another scene")
                # Ambiguous multiple available owners must not be chosen randomly.
                if len(available) != 1:
                    raise ValueError("Ambiguous image source blocks")
                assigned_ids.add(available[0])
            # Source document order is stable, even when the model returns IDs
            # out of order or omits an image source block.
            ids = [block["block_id"] for block in blocks if block["block_id"] in assigned_ids]
            if any(bid in used_blocks for bid in ids):
                raise ValueError("Block already assigned to another scene")
            refs = candidate.get("scripture_references", [])
            if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs):
                raise ValueError("scripture_references must be strings")
            try:
                confidence = max(0.0, min(1.0, float(candidate.get("confidence", 0))))
            except (TypeError, ValueError):
                raise ValueError("Invalid confidence")
            used_ids.add(sid)
            used_blocks.update(ids)
            used_images.update(links)
            materialized.append({"scene_id": sid, "scene_type": kind,
                "title": str(candidate.get("title") or ""),
                "caption": str(candidate.get("caption") or ""),
                "scripture_references": refs, "block_ids": ids, "image_links": links,
                "markdown_content": "\n\n".join(by_id[x]["markdown"] for x in ids),
                "confidence": confidence})
        unassigned = [x for x in image_order if x not in used_images]
        return {"status": "SUCCEEDED", "error": None,
                "scenes": materialized, "unassigned_image_links": unassigned}
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return {"status": "FAILED", "error": str(exc), "scenes": [], "unassigned_image_links": []}


def stable_scene_id(document_id: str, source_hash: str, prompt_version: str, scene_id: str) -> str:
    return hashlib.sha256(f"{document_id}|{source_hash}|{prompt_version}|{scene_id}".encode()).hexdigest()
