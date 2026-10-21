"""Attach compact, source-matched Azure Document Intelligence layouts to Bronze pages.

Only matched page layouts are validated. Raw OCR text is already in Markdown and
is deliberately omitted from the model's compact positional representation.
"""
from __future__ import annotations

import json

_SOURCE = r"(?m)^sourceBlob:\s*[\"']?(?:[^/\s\"']*/)?([^/\s\"']+/[^/\s\"']+/[^/\s\"']+)"
_LAYOUT = r"/generated/([^/]+/[^/]+/[^/]+)/layout\.json$"
PROCESSOR_VERSION = "compact-layout-v2"


def _geometry(region):
    """Preserve polygons/page coordinates without repetitive OCR payloads."""
    if not isinstance(region, dict):
        return {}
    out = {}
    for name in ("pageNumber", "boundingRegions", "polygon", "unit", "width", "height", "angle"):
        if name in region:
            out[name] = region[name]
    return out


def compact_layout(raw: bytes | str | dict, *, max_compact_bytes: int = 200_000) -> str:
    """Loss-aware extraction of structural layout, not a blind string truncation.

    Supports Azure DI AnalyzeResult and its common {'analyzeResult': ...} wrapper.
    Raises on malformed/oversized compact output rather than hiding figures.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(obj, dict):
        raise ValueError("layout.json root must be a JSON object")
    data = obj.get("analyzeResult", obj)
    if not isinstance(data, dict):
        raise ValueError("layout.json analyzeResult must be an object")

    pages = []
    for page in data.get("pages", []):
        if not isinstance(page, dict):
            continue
        pages.append({key: page[key] for key in
                      ("pageNumber", "width", "height", "unit", "angle") if key in page})

    figures = []
    for figure in data.get("figures", []):
        if not isinstance(figure, dict):
            continue
        item = {key: figure[key] for key in ("id", "elements", "spans") if key in figure}
        if "boundingRegions" in figure:
            item["boundingRegions"] = figure["boundingRegions"]
        caption = figure.get("caption")
        if isinstance(caption, dict):
            # Caption geometry and references; full text lives in Markdown.
            item["caption"] = {key: caption[key] for key in
                               ("content", "elements", "boundingRegions", "spans") if key in caption}
        figures.append(item)

    paragraphs = []
    for index, paragraph in enumerate(data.get("paragraphs", [])):
        if not isinstance(paragraph, dict):
            continue
        item = {"index": index}
        for key in ("role", "spans", "boundingRegions"):
            if key in paragraph:
                item[key] = paragraph[key]
        # Retain a small anchor for spatial/Markdown matching; not all OCR text.
        if isinstance(paragraph.get("content"), str):
            item["text_anchor"] = paragraph["content"][:160]
        paragraphs.append(item)

    tables = []
    for index, table in enumerate(data.get("tables", [])):
        if not isinstance(table, dict):
            continue
        item = {"index": index}
        for key in ("rowCount", "columnCount", "boundingRegions", "spans"):
            if key in table:
                item[key] = table[key]
        tables.append(item)

    result = {"layout_processor": PROCESSOR_VERSION,
              "pages": pages, "figures": figures,
              "paragraphs": paragraphs, "tables": tables}
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(encoded.encode("utf-8")) > max_compact_bytes:
        raise ValueError("Compacted layout exceeds model input limit; subdivide page by layout regions")
    return encoded


def attach_layout(spark, bronze_rows, layout_root: str, *, max_layout_bytes: int = 20_000_000,
                  max_compact_bytes: int = 200_000):
    """Match layout.json to exactly the selected source pages and compact it.

    `max_layout_bytes` protects per-page raw decoding, NOT AI prompt size.
    No layout root retains the previous Markdown-only behavior.
    """
    from pyspark.sql import functions as F, types as T
    if not layout_root:
        return bronze_rows.withColumn("layout_json", F.lit(None).cast("string"))
    root = layout_root.rstrip("/")
    rows = (bronze_rows
        .withColumn("_source_key", F.regexp_extract("markdown_content", _SOURCE, 1))
        .withColumn("_layout_key", F.regexp_replace("_source_key", r"\.[^.\/]+$", "")))
    if rows.filter(F.col("_layout_key") == "").limit(1).count():
        raise ValueError("Markdown sourceBlob is missing or cannot be mapped to a layout.json path")
    selected_keys = rows.select("_layout_key").distinct()

    # Spark binaryFile scans the root, but only matched files are decoded,
    # size-checked or passed to the classification pipeline.
    layouts = (spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "layout.json")
        .load(root)
        .select("path", "length", "content")
        .withColumn("_layout_key", F.regexp_extract("path", _LAYOUT, 1))
        .filter(F.col("_layout_key") != "")
        .join(selected_keys, "_layout_key", "inner"))

    if layouts.groupBy("_layout_key").count().filter("count > 1").limit(1).count():
        raise ValueError("Duplicate layout.json page keys under layout-root")
    oversized = layouts.filter(F.col("length") > max_layout_bytes).select("path", "length").limit(1).collect()
    if oversized:
        raise ValueError(f"Matched layout.json exceeds {max_layout_bytes} raw bytes: {oversized[0]['path']}")

    compact_udf = F.udf(lambda b: compact_layout(b, max_compact_bytes=max_compact_bytes), T.StringType())
    compact = layouts.select("_layout_key", compact_udf("content").alias("layout_json"))
    result = rows.join(compact, "_layout_key", "left")
    if result.filter(F.col("layout_json").isNull()).limit(1).count():
        raise ValueError("At least one selected Bronze page has no matching layout.json; check sourceBlob and layout-root")
    return result.drop("_source_key", "_layout_key")
