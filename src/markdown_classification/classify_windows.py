from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import Row, SparkSession, functions as F, types as T

from markdown_classification.blocks import build_page_windows, materialize_classifications
from markdown_classification.config import parse_settings


WINDOWS_TYPE = T.ArrayType(T.StructType([
    T.StructField("window_number", T.IntegerType(), False),
    T.StructField("first_page", T.LongType(), True),
    T.StructField("last_page", T.LongType(), True),
    T.StructField("source_document_ids", T.ArrayType(T.StringType()), False),
    T.StructField("source_hash", T.StringType(), False),
    T.StructField("pages_json", T.StringType(), False),
    T.StructField("input_characters", T.LongType(), False),
]))

SECTION_TYPE = T.StructType([
    T.StructField("section_id", T.StringType(), False),
    T.StructField("content_type", T.StringType(), False),
    T.StructField("title", T.StringType(), False),
    T.StructField("knowledge_id", T.StringType(), False),
    T.StructField("confidence", T.DoubleType(), False),
    T.StructField("block_ids", T.ArrayType(T.StringType()), False),
    T.StructField("source_document_ids", T.ArrayType(T.StringType()), False),
    T.StructField("source_page_numbers", T.ArrayType(T.LongType()), False),
    T.StructField("markdown_content", T.StringType(), False),
    T.StructField("image_links", T.ArrayType(T.StringType()), False),
    T.StructField("image_references_json", T.StringType(), False),
    T.StructField("preserved_original", T.BooleanType(), False),
])

MATERIALIZED_TYPE = T.StructType([
    T.StructField("status", T.StringType(), False),
    T.StructField("error", T.StringType(), True),
    T.StructField("sections", T.ArrayType(SECTION_TYPE), False),
])


def merge_windows(spark, table, rows):
    names = [
        "window_id", "chapter_key", "book_id", "chapter_id", "window_number",
        "first_page", "last_page", "source_document_ids", "source_hash", "pages_json",
        "input_characters", "raw_ai_result", "model_endpoint", "prompt_version",
        "processing_status", "processing_error",
    ]
    values = {name: f"s.{name}" for name in names}
    updates = {key: value for key, value in values.items() if key != "window_id"}
    updates["updated_at"] = "s.batch_time"
    inserts = {**values, "created_at": "s.batch_time", "updated_at": "s.batch_time"}
    (DeltaTable.forName(spark, table).alias("t")
     .merge(rows.alias("s"), "t.window_id=s.window_id")
     .whenMatchedUpdate(set=updates)
     .whenNotMatchedInsert(values=inserts)
     .execute())


def merge_fine_pages(spark, table, rows):
    names = [
        "fine_page_id", "window_id", "section_id", "book_id", "chapter_id",
        "page_type", "sequence_number", "title", "knowledge_id", "markdown_content",
        "block_ids", "source_document_ids", "source_page_numbers", "image_links",
        "image_references_json", "preserved_original", "classification_confidence",
        "classification_method", "classification_status", "classification_error",
        "is_active", "source_hash", "model_endpoint", "prompt_version",
    ]
    values = {name: f"s.{name}" for name in names}
    updates = {key: value for key, value in values.items() if key != "fine_page_id"}
    updates["updated_at"] = "s.batch_time"
    inserts = {**values, "created_at": "s.batch_time", "updated_at": "s.batch_time"}
    (DeltaTable.forName(spark, table).alias("t")
     .merge(rows.alias("s"), "t.fine_page_id=s.fine_page_id")
     .whenMatchedUpdate(set=updates)
     .whenNotMatchedInsert(values=inserts)
     .execute())


def main() -> None:
    s = parse_settings()
    spark = SparkSession.builder.getOrCreate()
    page_struct = F.struct(
        "page_number", "document_id", "markdown_content", "image_references", "content_hash"
    )
    chapters = (
        spark.table(s.bronze_table)
        .filter((F.col("ingestion_status") == "INGESTED") & F.col("markdown_content").isNotNull())
        .groupBy("book_id", "chapter_id")
        .agg(F.sort_array(F.collect_list(page_struct)).alias("pages"))
        .withColumn("chapter_key", F.concat_ws("/", "book_id", "chapter_id"))
    )
    window_udf = F.udf(
        lambda pages: [Row(**item) for item in build_page_windows(pages, s.max_pages_per_window)],
        WINDOWS_TYPE,
    )
    windows = (
        chapters.withColumn("window", F.explode(window_udf("pages")))
        .select("chapter_key", "book_id", "chapter_id", "window.*")
        .withColumn(
            "window_id",
            F.sha2(F.concat_ws("|", "chapter_key", "source_hash", "window_number",
                              F.lit(s.model_endpoint), F.lit(s.prompt_version)), 256),
        )
    )
    existing = spark.table(s.windows_table).select(
        "window_id", F.col("processing_status").alias("existing_status")
    )
    pending = windows.join(existing, "window_id", "left").filter(
        F.coalesce("existing_status", F.lit("FAILED")) != "SUCCEEDED"
    )
    if pending.limit(1).count() == 0:
        print("No new or failed page windows require classification.")
        return

    prompt = F.concat(
        F.lit(
            "You classify ordered Markdown blocks from 1 to 5 related mathematics-book pages. "
            "Return exactly one JSON object and no code fences or explanation. Schema: "
            "{\"sections\":[{\"section_id\":\"short-stable-id\","
            "\"content_type\":\"KNOWLEDGE|EXAMPLE|EXERCISE|OTHER\","
            "\"title\":\"short title\",\"knowledge_id\":\"shared-concept-id\","
            "\"confidence\":0.0,\"block_ids\":[\"exact-input-block-id\"]}]}. "
            "Classify every substantive block exactly once. A section may use blocks from multiple "
            "pages. Separate every individual exercise. Use the same knowledge_id for related "
            "knowledge, examples and exercises. Never copy Markdown into the response. Never invent "
            "block IDs. Preserve reading order. Headers and footers may be OTHER.\nINPUT:\n"
        ),
        F.col("pages_json"),
    )
    queried = pending.withColumn("prompt", prompt).withColumn(
        "ai", F.expr(f"ai_query('{s.model_endpoint}', prompt, failOnError => false)")
    )
    materialize_udf = F.udf(materialize_classifications, MATERIALIZED_TYPE)
    evaluated = queried.withColumn("materialized", materialize_udf("pages_json", "ai.result"))
    window_rows = evaluated.select(
        "window_id", "chapter_key", "book_id", "chapter_id", "window_number",
        "first_page", "last_page", "source_document_ids", "source_hash", "pages_json", "input_characters",
        F.col("ai.result").alias("raw_ai_result"),
        F.lit(s.model_endpoint).alias("model_endpoint"),
        F.lit(s.prompt_version).alias("prompt_version"),
        F.col("materialized.status").alias("processing_status"),
        F.coalesce(F.col("ai.errorMessage"), F.col("materialized.error")).alias("processing_error"),
        F.current_timestamp().alias("batch_time"),
    )
    merge_windows(spark, s.windows_table, window_rows)

    stored = (
        spark.table(s.windows_table)
        .filter(
            (F.col("processing_status") == "SUCCEEDED")
            & (F.col("model_endpoint") == s.model_endpoint)
            & (F.col("prompt_version") == s.prompt_version)
        )
        .withColumn("materialized", materialize_udf("pages_json", "raw_ai_result"))
    )
    successful = (
        stored.filter(F.col("materialized.status") == "SUCCEEDED")
        .select("window_id", "book_id", "chapter_id", "source_hash",
                F.posexplode("materialized.sections").alias("position", "section"))
    )
    fine_rows = successful.select(
        F.sha2(F.concat_ws("|", "window_id", F.col("section.section_id")), 256).alias("fine_page_id"),
        "window_id", F.col("section.section_id").alias("section_id"), "book_id", "chapter_id",
        F.col("section.content_type").alias("page_type"),
        (F.col("position") + 1).cast("int").alias("sequence_number"),
        F.col("section.title").alias("title"), F.col("section.knowledge_id").alias("knowledge_id"),
        F.col("section.markdown_content").alias("markdown_content"),
        F.col("section.block_ids").alias("block_ids"),
        F.col("section.source_document_ids").alias("source_document_ids"),
        F.col("section.source_page_numbers").alias("source_page_numbers"),
        F.col("section.image_links").alias("image_links"),
        F.col("section.image_references_json").alias("image_references_json"),
        F.col("section.preserved_original").alias("preserved_original"),
        F.col("section.confidence").alias("classification_confidence"),
        F.lit("AI_BLOCK_ASSIGNMENT").alias("classification_method"),
        F.when(F.col("section.confidence") >= s.confidence_threshold, "CLASSIFIED")
         .otherwise("REVIEW_REQUIRED").alias("classification_status"),
        F.when(F.col("section.confidence") < s.confidence_threshold,
               F.lit("Classification confidence is below the configured threshold"))
         .alias("classification_error"),
        F.lit(True).alias("is_active"), "source_hash",
        F.lit(s.model_endpoint).alias("model_endpoint"),
        F.lit(s.prompt_version).alias("prompt_version"),
        F.current_timestamp().alias("batch_time"),
    )
    if fine_rows.limit(1).count() > 0:
        spark.sql(
            f"UPDATE {s.fine_pages_table} SET is_active=false "
            f"WHERE coalesce(prompt_version, '') <> '{s.prompt_version}'"
        )
        merge_fine_pages(spark, s.fine_pages_table, fine_rows)
    print("Page-window classification completed.")


if __name__ == "__main__":
    main()
