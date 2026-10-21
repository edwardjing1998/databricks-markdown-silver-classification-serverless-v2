"""Attach compact, source-matched Azure Document Intelligence layouts to Bronze pages.

Only matched page layouts are validated. Raw OCR text is already in Markdown and
is deliberately omitted from the model's compact positional representation.
"""
from __future__ import annotations

import json
import re

_SOURCE = r"""(?im)^[ \t]*sourceBlob[ \t]*:[ \t]*["']?([^"'\r\n]+)"""
_LAYOUT = r"/generated/(?:[^/]+/)*([^/]+/[^/]+/[^/]+)/layout\.json$"
PROCESSOR_VERSION = "compact-layout-v3"



def resolve_layout_key(document_id: str | None, source_blob: str | None = None) -> str:
    """Resolve the canonical book/chapter/page key without a UUID prefix.

    Prefer Bronze document_id: original sourceBlob can contain upload UUIDs,
    legacy versions or intermediate directories unrelated to the output path.
    """
    if document_id:
        candidate = str(document_id).strip().strip("/")
        parts = candidate.split("/")
        if len(parts) == 3 and all(p not in ("", ".", "..") for p in parts):
            return candidate

    # Backwards-compatible fallback for older rows without canonical IDs.
    source = (source_blob or "").strip().strip("\"'")
    source = source.split("?", 1)[0].strip("/")
    parts = source.split("/")
    if len(parts) >= 3:
        tail = parts[-3:]
        tail[-1] = re.sub(r"\.[^./]+$", "", tail[-1])
        if all(p not in ("", ".", "..") for p in tail):
            return "/".join(tail)
    return ""

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
    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw = bytes(raw).decode("utf-8-sig")
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
            item["text_anchor"] = paragraph["content"][:320]
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


def expected_layout_path(source_markdown_path: str | None) -> str:
    """Return the exact sibling layout.json URI for a Bronze Markdown source.

    Never infer identity from sourceBlob, page filename, or truncated directories.
    """
    path = (source_markdown_path or "").strip()
    if not re.search(r"/content\.md$", path, flags=re.IGNORECASE):
        return ""
    return re.sub(r"/content\.md$", "/layout.json", path, flags=re.IGNORECASE)




def permitted_scan_root(markdown_path: str | None, layout_root: str) -> str:
    """Return an authorized generated/ scan root, optionally prefixed by one UUID.

    The UUID is a storage directory *outside* generated/, not part of a page key.
    A different container/account and arbitrary parent directories are forbidden.
    """
    path = (markdown_path or "").strip()
    configured = (layout_root or "").rstrip("/")
    if not expected_layout_path(path) or not configured:
        return ""
    if path.startswith(configured + "/"):
        return configured
    prefix, sep, suffix = configured.rpartition("/generated")
    if not sep or suffix:
        return ""
    match = re.match(
        r"^" + re.escape(prefix) +
        r"/([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})/generated/",
        path,
    )
    return prefix + "/" + match.group(1) + "/generated" if match else ""

def attach_layout(spark, bronze_rows, layout_root: str, *, max_layout_bytes: int = 20_000_000,
                  max_compact_bytes: int = 200_000):
    """Attach layouts by exact sibling URI from Bronze source_markdown_path.

    Missing or ambiguous layouts fail explicitly; never borrow another book's
    layout based on a common page name. The Bible and Math pipelines must pass
    source_markdown_path through their Bronze selection.
    """
    from pyspark.sql import functions as F, types as T
    if not layout_root:
        return bronze_rows.withColumn("layout_json", F.lit(None).cast("string"))
    if "source_markdown_path" not in bronze_rows.columns:
        # Original Bible job projects only document_id + markdown_content.
        # Keep that pre-existing calling convention and matching behavior.
        return _attach_legacy_layout(
            spark, bronze_rows, layout_root,
            max_layout_bytes=max_layout_bytes,
            max_compact_bytes=max_compact_bytes,
        )

    root = layout_root.rstrip("/")
    path_udf = F.udf(expected_layout_path, T.StringType())
    rows = bronze_rows.withColumn(
        "_expected_layout_path", path_udf(F.col("source_markdown_path"))
    )
    scan_root_udf = F.udf(lambda path: permitted_scan_root(path, root), T.StringType())
    rows = rows.withColumn("_scan_root", scan_root_udf(F.col("source_markdown_path")))
    invalid = (rows.filter(
        F.col("_expected_layout_path").isNull()
        | (F.col("_expected_layout_path") == "")
        | F.col("_scan_root").isNull()
        | (F.col("_scan_root") == "")
    ).select("document_id", "source_markdown_path").limit(5).collect())
    if invalid:
        details = "; ".join(
            f"document_id={r['document_id']} markdown={r['source_markdown_path']}"
            for r in invalid
        )
        raise ValueError("Invalid source_markdown_path for layout root: " + details)

    selected_paths = rows.select("_expected_layout_path").distinct()
    scan_roots = [r["_scan_root"] for r in rows.select("_scan_root").distinct().collect()]
    # Scan only authorized generated roots, including the distinct upload UUID roots.
    # Never strip UUIDs from exact layout paths or reuse similarly named pages.
    layouts = (spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "layout.json")
        .load(*scan_roots)
        .select(F.col("path").alias("_expected_layout_path"), "length", "content")
        .join(selected_paths, "_expected_layout_path", "inner"))

    dupes = (layouts.groupBy("_expected_layout_path")
        .count().filter(F.col("count") > 1).limit(5).collect())
    if dupes:
        raise ValueError("Duplicate exact layout.json paths: " +
                         "; ".join(r["_expected_layout_path"] for r in dupes))

    oversized = (layouts.filter(F.col("length") > max_layout_bytes)
        .select("_expected_layout_path", "length").limit(1).collect())
    if oversized:
        raise ValueError(
            f"Matched layout.json exceeds {max_layout_bytes} raw bytes: "
            f"{oversized[0]['_expected_layout_path']}"
        )

    result_type = T.StructType([
        T.StructField("layout_json", T.StringType(), True),
        T.StructField("error", T.StringType(), True),
    ])

    def _safe_compact(content, path):
        try:
            return (compact_layout(content, max_compact_bytes=max_compact_bytes), None)
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return (None, f"{path}: {type(exc).__name__}: {exc}")

    compact_udf = F.udf(_safe_compact, result_type)
    compact = (layouts
        .select("_expected_layout_path",
                compact_udf(F.col("content"), F.col("_expected_layout_path")).alias("_parsed"))
        .select("_expected_layout_path",
                F.col("_parsed.layout_json").alias("layout_json"),
                F.col("_parsed.error").alias("_layout_error")))
    result = rows.join(compact, "_expected_layout_path", "left")
    errors = (result.filter(F.col("layout_json").isNull())
        .select("document_id", "_expected_layout_path", "_layout_error")
        .limit(5).collect())
    if errors:
        details = "; ".join(
            f"document_id={r['document_id']} path={r['_expected_layout_path']} "
            f"error={r['_layout_error'] or 'NO_MATCHING_LAYOUT'}"
            for r in errors
        )
        raise ValueError(f"Layout compaction/matching failed: {details}")
    return result.drop("_expected_layout_path", "_scan_root", "_layout_error")


def _attach_legacy_layout(spark, bronze_rows, layout_root: str, *, max_layout_bytes: int = 20_000_000,
                  max_compact_bytes: int = 200_000):
    """Match layout.json to exactly the selected source pages and compact it.

    `max_layout_bytes` protects per-page raw decoding, NOT AI prompt size.
    No layout root retains the previous Markdown-only behavior.
    """
    from pyspark.sql import functions as F, types as T
    if not layout_root:
        return bronze_rows.withColumn("layout_json", F.lit(None).cast("string"))
    root = layout_root.rstrip("/")
    key_udf = F.udf(resolve_layout_key, T.StringType())
    rows = (bronze_rows
        .withColumn("_source_key", F.regexp_extract("markdown_content", _SOURCE, 1))
        .withColumn("_layout_key", key_udf(F.col("document_id"), F.col("_source_key"))))
    invalid = (rows.filter(F.col("_layout_key") == "")
        .select("document_id", "_source_key").limit(5).collect())
    if invalid:
        raise ValueError(
            "Cannot determine book/chapter/page layout key: " +
            "; ".join(f"document_id={r['document_id']} sourceBlob={r['_source_key']}" for r in invalid)
        )
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

    # Return a structured result rather than throwing from a distributed UDF.
    # This preserves the offending blob path and avoids an opaque worker stack trace.
    result_type = T.StructType([
        T.StructField("layout_json", T.StringType(), True),
        T.StructField("error", T.StringType(), True),
    ])

    def _safe_compact(content, path):
        try:
            return (compact_layout(content, max_compact_bytes=max_compact_bytes), None)
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return (None, f"{path}: {type(exc).__name__}: {exc}")

    compact_udf = F.udf(_safe_compact, result_type)
    compact = (layouts
        .select("_layout_key", compact_udf(F.col("content"), F.col("path")).alias("_parsed"))
        .select("_layout_key", F.col("_parsed.layout_json").alias("layout_json"),
                F.col("_parsed.error").alias("_layout_error")))
    result = rows.join(compact, "_layout_key", "left")
    errors = (result
        .filter(F.col("layout_json").isNull())
        .select("document_id", "_layout_key", "_layout_error")
        .limit(5).collect())
    if errors:
        details = "; ".join(
            f"document_id={r['document_id']} key={r['_layout_key']} "
            f"error={r['_layout_error'] or 'NO_MATCHING_LAYOUT'}"
            for r in errors
        )
        raise ValueError(f"Layout compaction/matching failed: {details}")
    return result.drop("_source_key", "_layout_key", "_layout_error")
