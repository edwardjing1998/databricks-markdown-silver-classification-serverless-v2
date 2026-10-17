"""Conservative link hints for mathematical figures, charts and diagrams.

No visual interpretation is claimed: links are based on Markdown ordering and
explicit figure-referencing language. Ambiguous assets remain unlinked.
"""
from __future__ import annotations

import re
from typing import Any

from markdown_classification.blocks import IMAGE_PATTERN

# Language-neutral mathematics pipeline; these are optional English/Chinese cues.
FIGURE_CUE = re.compile(
    r"\b(?:figure|fig\.?|diagram|graph|chart|table|shown|below|above)\b"
    r"|如图|见图|下图|上图|图中|图示|根据图|统计图|函数图|图表|下表",
    re.IGNORECASE,
)


def _image_only(block: dict[str, Any]) -> bool:
    content = block.get("markdown", "").strip()
    return bool(IMAGE_PATTERN.search(content)) and not IMAGE_PATTERN.sub("", content).strip()


def annotate_figure_links(pages: list[dict[str, Any]]) -> None:
    """Mutate already-sorted pages with conservative figure ownership hints.

    Images on a subsequent page are linked only when they are the first
    substantive block and the preceding page ends in an explicit diagram cue.
    A page/chapter boundary without such a cue is never linked automatically.
    """
    previous_page = None
    for page in pages:
        blocks = page.get("blocks", [])
        for idx, block in enumerate(blocks):
            block["candidate_parent_block_id"] = None
            if not _image_only(block):
                continue
            preceding = blocks[:idx]
            # Pick last textual block only; do not jump past other figures.
            local = preceding[-1] if preceding else None
            if local and not _image_only(local):
                block["candidate_parent_block_id"] = local["block_id"]
            elif idx == 0 and previous_page is not None:
                prev_blocks = previous_page.get("blocks", [])
                if (prev_blocks and not _image_only(prev_blocks[-1])
                        and FIGURE_CUE.search(prev_blocks[-1].get("markdown", ""))
                        and page.get("page_number") is not None
                        and previous_page.get("page_number") is not None
                        and page["page_number"] == previous_page["page_number"] + 1):
                    block["candidate_parent_block_id"] = prev_blocks[-1]["block_id"]
        previous_page = page


def reconcile_figure_assignments(
    sections: list[dict[str, Any]], ordered_blocks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep referenced standalone figures in their parent's classified unit.

    Only move a figure when its owner is unambiguous and another section
    currently owns the figure. This does not join arbitrary adjacent text.
    """
    block_owner: dict[str, int] = {}
    for idx, section in enumerate(sections):
        for block_id in section.get("block_ids") or []:
            block_owner.setdefault(block_id, idx)
    for block in ordered_blocks:
        parent = block.get("candidate_parent_block_id")
        owner_idx = block_owner.get(parent)
        figure_idx = block_owner.get(block["block_id"])
        if owner_idx is None or figure_idx is None or owner_idx == figure_idx:
            continue
        # Only attach to instructional units. A heading by itself is insufficient.
        if sections[owner_idx].get("content_type", "").upper() not in {
            "EXERCISE", "EXAMPLE", "KNOWLEDGE"
        }:
            continue
        sections[figure_idx]["block_ids"].remove(block["block_id"])
        sections[owner_idx]["block_ids"].append(block["block_id"])
        block_owner[block["block_id"]] = owner_idx
    return [section for section in sections if section.get("block_ids")]
