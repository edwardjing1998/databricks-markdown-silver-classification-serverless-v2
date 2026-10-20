"""Read Azure Document Intelligence layout siblings and attach to Bronze rows.

Opt-in via --layout-root. Uses the YAML sourceBlob emitted by the Java
Document Intelligence writer; fails closed when mappings are ambiguous/missing.
"""
from __future__ import annotations
from pyspark.sql import functions as F

# Both source and layout must use generated/{book}/{chapter}/{page_stem}.
_SOURCE = r"(?m)^sourceBlob:\s*[\"\']?(?:[^/\s\"\']*/)?([^/\s\"\']+/[^/\s\"\']+/[^/\s\"\']+)"
_LAYOUT = r"/generated/([^/]+/[^/]+/[^/]+)/layout\.json$"


def attach_layout(spark, bronze_rows, layout_root: str, *, max_layout_bytes: int = 250000):
    """Attach layout_json to original Bronze rows; no layout-root means disabled.

    Bronze markdown's sourceBlob frontmatter must identify source/<book>/<chapter>/<file>.
    Layout must be located at generated/<book>/<chapter>/<file_stem>/layout.json.
    We intentionally reject unmatched pages, duplicates and oversize blobs, never
    silently downgrade a requested layout-enhanced run to Markdown-only.
    """
    if not layout_root:
        return bronze_rows.withColumn("layout_json", F.lit(None).cast("string"))
    root = layout_root.rstrip("/")
    layouts = (spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "layout.json")
        .load(root)
        .select("path", "length", "content")
        .withColumn("_layout_key", F.regexp_extract("path", _LAYOUT, 1))
        .filter(F.col("_layout_key") != ""))
    if layouts.filter(F.col("length") > max_layout_bytes).limit(1).count():
        raise ValueError("layout.json exceeds max_layout_bytes; configure a compact layout extraction first")
    if layouts.groupBy("_layout_key").count().filter("count > 1").limit(1).count():
        raise ValueError("Duplicate layout.json page keys under layout-root")
    layouts = layouts.select("_layout_key", F.decode("content", "UTF-8").alias("layout_json"))
    rows = (bronze_rows.withColumn("_source_key", F.regexp_extract("markdown_content", _SOURCE, 1))
        .withColumn("_layout_key", F.regexp_replace("_source_key", r"\.[^.\/]+$", "")))
    result = rows.join(layouts, "_layout_key", "left")
    if result.filter(F.col("layout_json").isNull()).limit(1).count():
        raise ValueError("At least one Bronze Markdown page has no matching generated layout.json; check sourceBlob and layout-root")
    return result.drop("_source_key", "_layout_key")
