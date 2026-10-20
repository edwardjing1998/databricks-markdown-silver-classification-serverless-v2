from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import Row, SparkSession, functions as F, types as T

from markdown_classification.blocks import (
    build_page_windows,
    materialize_classifications,
)
from markdown_classification.config import parse_settings
from markdown_classification.source_filters import parse_book_ids
from markdown_classification.prompt_loader import classification_prompt


WINDOWS_TYPE = T.ArrayType(
    T.StructType(
        [
            T.StructField("window_number", T.IntegerType(), False),
            T.StructField("first_page", T.LongType(), True),
            T.StructField("last_page", T.LongType(), True),
            T.StructField(
                "source_document_ids",
                T.ArrayType(T.StringType()),
                False,
            ),
            T.StructField("source_hash", T.StringType(), False),
            T.StructField("pages_json", T.StringType(), False),
            T.StructField("input_characters", T.LongType(), False),
        ]
    )
)

SECTION_TYPE = T.StructType(
    [
        T.StructField("section_id", T.StringType(), False),
        T.StructField("content_type", T.StringType(), False),
        T.StructField("title", T.StringType(), False),
        T.StructField("knowledge_id", T.StringType(), False),
        T.StructField("confidence", T.DoubleType(), False),
        T.StructField(
            "block_ids",
            T.ArrayType(T.StringType()),
            False,
        ),
        T.StructField(
            "source_document_ids",
            T.ArrayType(T.StringType()),
            False,
        ),
        T.StructField(
            "source_page_numbers",
            T.ArrayType(T.LongType()),
            False,
        ),
        T.StructField("markdown_content", T.StringType(), False),
        T.StructField(
            "image_links",
            T.ArrayType(T.StringType()),
            False,
        ),
        T.StructField("image_references_json", T.StringType(), False),
        T.StructField("preserved_original", T.BooleanType(), False),
    ]
)

MATERIALIZED_TYPE = T.StructType(
    [
        T.StructField("status", T.StringType(), False),
        T.StructField("error", T.StringType(), True),
        T.StructField(
            "sections",
            T.ArrayType(SECTION_TYPE),
            False,
        ),
    ]
)


def merge_windows(
    spark: SparkSession,
    table: str,
    rows,
) -> None:
    names = [
        "window_id",
        "chapter_key",
        "book_id",
        "chapter_id",
        "window_number",
        "first_page",
        "last_page",
        "source_document_ids",
        "source_hash",
        "pages_json",
        "input_characters",
        "raw_ai_result",
        "model_endpoint",
        "prompt_version",
        "processing_status",
        "processing_error",
    ]

    values = {
        name: f"source.{name}"
        for name in names
    }

    updates = {
        key: value
        for key, value in values.items()
        if key != "window_id"
    }
    updates["updated_at"] = "source.batch_time"

    inserts = {
        **values,
        "created_at": "source.batch_time",
        "updated_at": "source.batch_time",
    }

    (
        DeltaTable.forName(spark, table)
        .alias("target")
        .merge(
            rows.alias("source"),
            "target.window_id = source.window_id",
        )
        .whenMatchedUpdate(set=updates)
        .whenNotMatchedInsert(values=inserts)
        .execute()
    )


def merge_fine_pages(
    spark: SparkSession,
    table: str,
    rows,
) -> None:
    names = [
        "fine_page_id",
        "window_id",
        "section_id",
        "book_id",
        "chapter_id",
        "page_type",
        "sequence_number",
        "title",
        "knowledge_id",
        "markdown_content",
        "block_ids",
        "source_document_ids",
        "source_page_numbers",
        "image_links",
        "image_references_json",
        "preserved_original",
        "classification_confidence",
        "classification_method",
        "classification_status",
        "classification_error",
        "is_active",
        "source_hash",
        "model_endpoint",
        "prompt_version",
    ]

    values = {
        name: f"source.{name}"
        for name in names
    }

    updates = {
        key: value
        for key, value in values.items()
        if key != "fine_page_id"
    }
    updates["updated_at"] = "source.batch_time"

    inserts = {
        **values,
        "created_at": "source.batch_time",
        "updated_at": "source.batch_time",
    }

    (
        DeltaTable.forName(spark, table)
        .alias("target")
        .merge(
            rows.alias("source"),
            "target.fine_page_id = source.fine_page_id",
        )
        .whenMatchedUpdate(set=updates)
        .whenNotMatchedInsert(values=inserts)
        .execute()
    )


def build_current_windows(spark: SparkSession, settings):
    """Recompute authoritative window IDs from currently ingested Bronze pages."""

    # Do not depend on the Bronze table having a physical content_hash
    # column. Generate a stable SHA-256 hash directly from the Markdown.
    bronze = (
        spark.table(settings.bronze_table)
        .filter(
            (F.col("ingestion_status") == "INGESTED")
            & F.col("markdown_content").isNotNull()
        )
        .withColumn(
            "_content_hash",
            F.sha2(
                F.coalesce(
                    F.col("markdown_content"),
                    F.lit(""),
                ),
                256,
            ),
        )
    )

    # Exclude Bible content from the original mathematics classifier.
    excluded = parse_book_ids(settings.exclude_book_ids)
    if excluded:
        bronze = bronze.filter(~F.col("book_id").isin(*excluded))

    # Rename the calculated hash to content_hash inside the page structure
    # because build_page_windows() expects that field.
    page_struct = F.struct(
        F.col("page_number").alias("page_number"),
        F.col("document_id").alias("document_id"),
        F.col("markdown_content").alias("markdown_content"),
        F.col("image_references").alias("image_references"),
        F.col("_content_hash").alias("content_hash"),
    )

    chapters = (
        bronze
        .groupBy(
            "book_id",
            "chapter_id",
        )
        .agg(
            F.sort_array(
                F.collect_list(page_struct)
            ).alias("pages")
        )
        .withColumn(
            "chapter_key",
            F.concat_ws(
                "/",
                F.col("book_id"),
                F.col("chapter_id"),
            ),
        )
    )

    window_udf = F.udf(
        lambda pages: [
            Row(**item)
            for item in build_page_windows(
                pages,
                settings.max_pages_per_window,
            )
        ],
        WINDOWS_TYPE,
    )

    windows = (
        chapters
        .withColumn(
            "window",
            F.explode(window_udf(F.col("pages"))),
        )
        .select(
            "chapter_key",
            "book_id",
            "chapter_id",
            "window.*",
        )
        .withColumn(
            "window_id",
            F.sha2(
                F.concat_ws(
                    "|",
                    F.col("chapter_key"),
                    F.col("source_hash"),
                    F.col("window_number"),
                    F.lit(settings.model_endpoint),
                    F.lit(settings.prompt_version),
                ),
                256,
            ),
        )
    )
    return windows


def main() -> None:
    settings = parse_settings()
    spark = SparkSession.builder.getOrCreate()
    windows = build_current_windows(spark, settings)

    existing = (
        spark.table(settings.windows_table)
        .select(
            "window_id",
            F.col("processing_status").alias("existing_status"),
        )
    )

    pending = (
        windows
        .join(
            existing,
            "window_id",
            "left",
        )
        .filter(
            F.coalesce(
                F.col("existing_status"),
                F.lit("FAILED"),
            )
            != "SUCCEEDED"
        )
    )

    materialize_udf = F.udf(materialize_classifications, MATERIALIZED_TYPE)
    has_pending = pending.limit(1).count() > 0
    if not has_pending:
        print("No new or failed page windows require AI classification; reconciling active results.")

    if has_pending:
        prompt_instructions = classification_prompt(settings.max_pages_per_window)

        prompt = F.concat(
            F.lit(prompt_instructions),
            F.col("pages_json"),
        )

        queried = (
            pending
            .withColumn(
                "prompt",
                prompt,
            )
            .withColumn(
                "ai",
                F.expr(
                    "ai_query("
                    f"'{settings.model_endpoint}', "
                    "prompt, "
                    "failOnError => false"
                    ")"
                ),
            )
        )

        evaluated = queried.withColumn(
            "materialized",
            materialize_udf(
                F.col("pages_json"),
                F.col("ai.result"),
            ),
        )

        window_rows = evaluated.select(
            "window_id",
            "chapter_key",
            "book_id",
            "chapter_id",
            "window_number",
            "first_page",
            "last_page",
            "source_document_ids",
            "source_hash",
            "pages_json",
            "input_characters",
            F.col("ai.result").alias("raw_ai_result"),
            F.lit(settings.model_endpoint).alias("model_endpoint"),
            F.lit(settings.prompt_version).alias("prompt_version"),
            F.col("materialized.status").alias("processing_status"),
            F.coalesce(
                F.col("ai.errorMessage"),
                F.col("materialized.error"),
            ).alias("processing_error"),
            F.current_timestamp().alias("batch_time"),
        )

        # Persist the AI response once. This prevents ai_query from being
        # evaluated a second time while fine-grained pages are generated.
        merge_windows(
            spark,
            settings.windows_table,
            window_rows,
        )

    # Read the stored AI results back from the Delta table.
    stored = (
        spark.table(settings.windows_table)
        .join(windows.select("window_id"), "window_id", "left_semi")
        .filter(
            (F.col("processing_status") == "SUCCEEDED")
            & (
                F.col("model_endpoint")
                == settings.model_endpoint
            )
            & (
                F.col("prompt_version")
                == settings.prompt_version
            )
        )
        .withColumn(
            "materialized",
            materialize_udf(
                F.col("pages_json"),
                F.col("raw_ai_result"),
            ),
        )
    )

    successful = (
        stored
        .filter(
            F.col("materialized.status") == "SUCCEEDED"
        )
        .select(
            "window_id",
            "book_id",
            "chapter_id",
            "source_hash",
            F.posexplode(
                F.col("materialized.sections")
            ).alias(
                "position",
                "section",
            ),
        )
    )

    fine_rows = successful.select(
        F.sha2(
            F.concat_ws(
                "|",
                F.col("window_id"),
                F.col("section.section_id"),
            ),
            256,
        ).alias("fine_page_id"),
        F.col("window_id"),
        F.col("section.section_id").alias("section_id"),
        F.col("book_id"),
        F.col("chapter_id"),
        F.col("section.content_type").alias("page_type"),
        (F.col("position") + 1).cast("int").alias(
            "sequence_number"
        ),
        F.col("section.title").alias("title"),
        F.col("section.knowledge_id").alias("knowledge_id"),
        F.col("section.markdown_content").alias(
            "markdown_content"
        ),
        F.col("section.block_ids").alias("block_ids"),
        F.col("section.source_document_ids").alias(
            "source_document_ids"
        ),
        F.col("section.source_page_numbers").alias(
            "source_page_numbers"
        ),
        F.col("section.image_links").alias("image_links"),
        F.col("section.image_references_json").alias(
            "image_references_json"
        ),
        F.col("section.preserved_original").alias(
            "preserved_original"
        ),
        F.col("section.confidence").alias(
            "classification_confidence"
        ),
        F.lit("AI_BLOCK_ASSIGNMENT").alias(
            "classification_method"
        ),
        F.when(
            F.col("section.confidence")
            >= settings.confidence_threshold,
            F.lit("CLASSIFIED"),
        )
        .otherwise(
            F.lit("REVIEW_REQUIRED")
        )
        .alias("classification_status"),
        F.when(
            F.col("section.confidence")
            < settings.confidence_threshold,
            F.lit(
                "Classification confidence is below "
                "the configured threshold"
            ),
        ).alias("classification_error"),
        F.lit(True).alias("is_active"),
        F.col("source_hash"),
        F.lit(settings.model_endpoint).alias("model_endpoint"),
        F.lit(settings.prompt_version).alias("prompt_version"),
        F.current_timestamp().alias("batch_time"),
    )

    # Deactivate rows whose source windows are no longer current. Preserve
    # audit history while preventing superseded windows from being retrieved.
    current_window_ids = windows.select("window_id").distinct()
    current_chapters = windows.select("book_id", "chapter_id").distinct()
    stale_rows = (
        spark.table(settings.fine_pages_table)
        .filter(F.col("is_active") == True)
        .join(current_chapters, ["book_id", "chapter_id"], "inner")
        .join(current_window_ids, "window_id", "left_anti")
        .select("fine_page_id").distinct()
        .withColumn("deactivate", F.lit(True))
    )
    if stale_rows.limit(1).count() > 0:
        (
            DeltaTable.forName(spark, settings.fine_pages_table)
            .alias("target")
            .merge(stale_rows.alias("source"),
                   "target.fine_page_id = source.fine_page_id")
            .whenMatchedUpdate(set={"is_active": "false",
                                    "updated_at": "current_timestamp()"})
            .execute()
        )

    # The current run must never send two records with the same MERGE key.
    duplicate_keys = (
        fine_rows.groupBy("fine_page_id").count()
        .filter(F.col("count") > 1)
    )
    if duplicate_keys.limit(1).count() > 0:
        raise RuntimeError("Duplicate fine_page_id in current classification batch")

    if fine_rows.limit(1).count() > 0:
        merge_fine_pages(spark, settings.fine_pages_table, fine_rows)

    print("Page-window classification completed.")


if __name__ == "__main__":
    main()