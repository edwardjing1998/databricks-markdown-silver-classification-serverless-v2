from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import Row, SparkSession, functions as F, types as T

from markdown_classification.blocks import (
    build_page_windows,
    materialize_classifications,
)
from markdown_classification.config import parse_settings


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


def main() -> None:
    settings = parse_settings()
    spark = SparkSession.builder.getOrCreate()

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

    if pending.limit(1).count() == 0:
        print("No new or failed page windows require classification.")
        return

    prompt_instructions = (
        "You classify ordered Markdown blocks from related mathematics-book "
        f"pages. This request contains at most "
        f"{settings.max_pages_per_window} pages. "
        "Return exactly one JSON object and no code fences or explanation. "
        "The required JSON schema is: "
        '{"sections":['
        "{"
        '"section_id":"short-stable-id",'
        '"content_type":"KNOWLEDGE|EXAMPLE|EXERCISE|OTHER",'
        '"title":"short title",'
        '"knowledge_id":"shared-concept-id",'
        '"confidence":0.0,'
        '"block_ids":["exact-input-block-id"]'
        "}"
        "]}. "
        "Classify every substantive block exactly once. "
        "A section may use blocks from multiple related pages. "
        "Separate every individual exercise into its own section. "
        "Use the same knowledge_id for related knowledge, examples, and "
        "exercises. "
        "Never copy Markdown into the response. "
        "Never invent block IDs. "
        "Use only block IDs present in the input. "
        "Preserve the original reading order. "
        "Headers, footers, and unrelated material may be OTHER."
        "\nINPUT:\n"
    )

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

    materialize_udf = F.udf(
        materialize_classifications,
        MATERIALIZED_TYPE,
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

    if fine_rows.limit(1).count() > 0:
        # Rows generated by older prompt versions remain available for
        # auditing, but they are no longer active.
        spark.sql(
            f"""
            UPDATE {settings.fine_pages_table}
            SET is_active = false
            WHERE coalesce(prompt_version, '') <>
                  '{settings.prompt_version}'
            """
        )

        merge_fine_pages(
            spark,
            settings.fine_pages_table,
            fine_rows,
        )

    print("Page-window classification completed.")


if __name__ == "__main__":
    main()