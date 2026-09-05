from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*(?:\n|\Z)", re.DOTALL)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass(frozen=True)
class ParsedSection:
    section_number: int
    heading_level: int | None
    title: str | None
    content: str
    image_links: list[str]
    rule_type: str
    rule_confidence: float


def rule_classify(title: str | None, content: str) -> tuple[str, float]:
    value = f"{title or ''}\n{content}".strip().lower()
    rules = [
        (r"\b(exercise|practice|problem|question)\b", "EXERCISE", 0.93),
        (r"\b(example|worked example)\b", "EXAMPLE", 0.95),
        (r"\b(definition|define)\b", "DEFINITION", 0.95),
        (r"\b(theorem|lemma|corollary|proof)\b", "THEOREM", 0.94),
        (r"\b(answer|solution)\b", "ANSWER", 0.92),
        (r"\b(chapter|unit|lesson)\b", "CHAPTER_HEADING", 0.90),
    ]
    for pattern, label, confidence in rules:
        if re.search(pattern, value):
            return label, confidence
    if title and not content.strip():
        return "TITLE", 0.88
    return "OTHER", 0.35


def parse_markdown(markdown: str | None) -> list[ParsedSection]:
    text = HTML_COMMENT.sub("", FRONT_MATTER.sub("", markdown or "")).strip()
    if not text:
        return []
    groups: list[tuple[int | None, str | None, list[str]]] = []
    level: int | None = None
    title: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            if title is not None or any(part.strip() for part in body):
                groups.append((level, title, body))
            level, title, body = len(match.group(1)), match.group(2).strip(), []
        else:
            body.append(line)
    if title is not None or any(part.strip() for part in body):
        groups.append((level, title, body))

    sections = []
    for index, (heading_level, heading, lines) in enumerate(groups, 1):
        content = "\n".join(lines).strip()
        links = IMAGE.findall(content)
        label, confidence = rule_classify(heading, content)
        sections.append(ParsedSection(index, heading_level, heading, content, links, label, confidence))
    return sections


def section_id(document_id: str, section_number: int) -> str:
    return hashlib.sha256(f"{document_id}#{section_number}".encode()).hexdigest()

