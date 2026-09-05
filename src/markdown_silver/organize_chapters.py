from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import Row, SparkSession, functions as F, types as T

from markdown_silver.config import parse_settings
from markdown_silver.windowing import build_page_windows


WINDOW_SCHEMA = """STRUCT<analysis:STRUCT<
units:ARRAY<STRUCT<
concept_id:STRING,title:STRING,knowledge_markdown:STRING,
examples:ARRAY<STRUCT<title:STRING,markdown:STRING,source_document_ids:ARRAY<STRING>>>,
exercises:ARRAY<STRUCT<exercise_id:STRING,title:STRING,markdown:STRING,
source_document_ids:ARRAY<STRING>,single_exercise_page:BOOLEAN>>,
source_document_ids:ARRAY<STRING>>>,
continuations:ARRAY<STRUCT<source_document_id:STRING,continues_to_next_window:BOOLEAN>>
>>""".replace("\n", "")

FINAL_SCHEMA = """STRUCT<organization:STRUCT<
chapter_title:STRING,
units:ARRAY<STRUCT<
unit_id:STRING,title:STRING,knowledge_markdown:STRING,
examples_markdown:ARRAY<STRING>,
exercises:ARRAY<STRUCT<exercise_id:STRING,title:STRING,markdown:STRING,
source_document_ids:ARRAY<STRING>,preserve_original:BOOLEAN>>,
source_document_ids:ARRAY<STRING>>>
>>""".replace("\n", "")

WINDOWS_TYPE = T.ArrayType(
    T.StructType(
        [
            T.StructField("window_number", T.IntegerType(), False),
            T.StructField("first_page", T.LongType(), True),
            T.StructField("last_page", T.LongType(), True),
            T.StructField("source_document_ids", T.ArrayType(T.StringType()), False),
            T.StructField("pages_json", T.StringType(), False),
            T.StructField("input_characters", T.LongType(), False),
        ]
    )
)


def _make_window_builder(max_pages: int, overlap: int, max_characters: int):
    def build(pages: list[Row] | None) -> list[Row]:
        return [
            Row(**window)
            for window in build_page_windows(
                pages, max_pages=max_pages, overlap=overlap,
                max_characters=max_characters,
            )
        ]

    return F.udf(build, WINDOWS_TYPE)


def _merge_windows(spark: SparkSession, settings, rows) -> None:
    target = DeltaTable.forName(spark, settings.chapter_windows_table)
    values = {
        "window_id": "s.window_id", "chapter_key": "s.chapter_key",
        "book_id": "s.book_id", "chapter_id": "s.chapter_id",
        "window_number": "s.window_number", "first_page": "s.first_page",
        "last_page": "s.last_page", "source_document_ids": "s.source_document_ids",
        "source_hash": "s.source_hash", "input_characters": "s.input_characters",
        "raw_ai_result": "s.raw_ai_result", "model_endpoint": "s.model_endpoint",
        "prompt_version": "s.prompt_version", "processing_status": "s.processing_status",
        "processing_error": "s.processing_error",
    }
    updates = {key: value for key, value in values.items() if key != "window_id"}
    updates["updated_at"] = "s.batch_time"
    inserts = dict(values)
    inserts.update(created_at="s.batch_time", updated_at="s.batch_time")
    (target.alias("t").merge(rows.alias("s"), "t.window_id=s.window_id")
     .whenMatchedUpdate(set=updates).whenNotMatchedInsert(values=inserts).execute())


def _merge_plans(spark: SparkSession, settings, rows) -> None:
    target = DeltaTable.forName(spark, settings.chapter_plans_table)
    (target.alias("t").merge(rows.alias("s"), "t.chapter_key=s.chapter_key")
     .whenMatchedUpdate(set={
         "book_id": "s.book_id", "chapter_id": "s.chapter_id",
         "source_hash": "s.source_hash", "raw_ai_result": "s.raw_ai_result",
         "model_endpoint": "s.model_endpoint", "prompt_version": "s.prompt_version",
         "processing_status": "s.processing_status", "processing_error": "s.processing_error",
         "updated_at": "s.batch_time",
     }).whenNotMatchedInsert(values={
         "chapter_key": "s.chapter_key", "book_id": "s.book_id",
         "chapter_id": "s.chapter_id", "source_hash": "s.source_hash",
         "raw_ai_result": "s.raw_ai_result", "model_endpoint": "s.model_endpoint",
         "prompt_version": "s.prompt_version", "processing_status": "s.processing_status",
         "processing_error": "s.processing_error", "created_at": "s.batch_time",
         "updated_at": "s.batch_time",
     }).execute())


def main() -> None:
    settings = parse_settings()
    if not settings.enable_ai:
        raise ValueError("Window analysis and chapter consolidation require --enable-ai=true")

    spark = SparkSession.builder.getOrCreate()
    page_struct = F.struct(
        "page_number", "document_id", "markdown_content", "image_references"
    )
    chapters = (
        spark.table(settings.bronze_table)
        .filter((F.col("ingestion_status") == "INGESTED") & F.col("markdown_content").isNotNull())
        .groupBy("book_id", "chapter_id")
        .agg(F.sort_array(F.collect_list(page_struct)).alias("pages"))
        .withColumn("chapter_key", F.concat_ws("/", "book_id", "chapter_id"))
        .withColumn(
            "source_hash",
            F.sha2(F.concat_ws("|", F.transform("pages", lambda p: p.markdown_content)), 256),
        )
    )

    existing = spark.table(settings.chapter_plans_table).select(
        "chapter_key", F.col("source_hash").alias("existing_hash"),
        F.col("model_endpoint").alias("existing_model"),
        F.col("prompt_version").alias("existing_prompt"),
    )
    pending = chapters.join(existing, "chapter_key", "left").filter(
        F.col("existing_hash").isNull()
        | (F.col("source_hash") != F.col("existing_hash"))
        | (F.coalesce("existing_model", F.lit("")) != settings.model_endpoint)
        | (F.coalesce("existing_prompt", F.lit("")) != settings.prompt_version)
    )
    if pending.limit(1).count() == 0:
        print("No new or changed chapters require organization.")
        return

    build_windows = _make_window_builder(
        settings.max_pages_per_window,
        settings.window_overlap_pages,
        settings.max_window_characters,
    )
    windows = (
        pending.withColumn("window", F.explode(build_windows("pages")))
        .select("chapter_key", "book_id", "chapter_id", "source_hash", "pages", "window.*")
        .withColumn(
            "window_id",
            F.sha2(F.concat_ws("|", "chapter_key", "source_hash", "window_number", F.lit(settings.prompt_version)), 256),
        )
    )
    window_prompt = F.concat(
        F.lit(
            "Analyze this ordered window from a mathematics-book chapter. Identify knowledge "
            "concepts, worked examples, and every individual exercise. Keep exact source document "
            "IDs and Markdown image links. Mark cross-window continuations. Do not invent content. "
            "This is an intermediate analysis; overlapping pages may occur.\n"
        ),
        F.concat(F.lit("BOOK_ID: "), F.col("book_id"), F.lit("\nCHAPTER_ID: "), F.col("chapter_id")),
        F.lit("\nPAGES_JSON:\n"), F.col("pages_json"),
    )
    analyzed = (
        windows.withColumn("prompt", window_prompt)
        .withColumn(
            "ai",
            F.expr(
                f"ai_query('{settings.model_endpoint}', prompt, "
                f"responseFormat => '{WINDOW_SCHEMA}', failOnError => false)"
            ),
        )
        .withColumn("parsed_window", F.from_json("ai.result", WINDOW_SCHEMA))
        .persist()
    )
    analyzed.count()  # Materialize each external AI call exactly once.
    window_rows = analyzed.select(
        "window_id", "chapter_key", "book_id", "chapter_id", "window_number",
        "first_page", "last_page", "source_document_ids", "source_hash",
        "input_characters", F.col("ai.result").alias("raw_ai_result"),
        F.lit(settings.model_endpoint).alias("model_endpoint"),
        F.lit(settings.prompt_version).alias("prompt_version"),
        F.when(F.col("parsed_window.analysis").isNotNull(), "SUCCEEDED").otherwise("FAILED").alias("processing_status"),
        F.col("ai.errorMessage").alias("processing_error"), F.current_timestamp().alias("batch_time"),
    )
    _merge_windows(spark, settings, window_rows)

    consolidation_inputs = (
        analyzed.groupBy("chapter_key", "book_id", "chapter_id", "source_hash", "pages")
        .agg(
            F.sort_array(F.collect_list(F.struct("window_number", "ai.result"))).alias("window_results"),
            F.sum(F.when(F.col("parsed_window.analysis").isNull(), 1).otherwise(0)).alias("failed_windows"),
        )
    )
    consolidation_prompt = F.concat(
        F.lit(
            "Consolidate the ordered, overlapping window analyses for one mathematics chapter. "
            "Deduplicate overlap and resolve continuations. For each knowledge concept, create one "
            "learning unit containing its knowledge, related worked examples, and related exercises. "
            "Also return every exercise separately. Set preserve_original=true only when exactly one "
            "source page contains one exercise and no other substantive content. Preserve source IDs "
            "and image links, maintain mathematical meaning, and never invent content.\n"
        ),
        F.concat(F.lit("BOOK_ID: "), F.col("book_id"), F.lit("\nCHAPTER_ID: "), F.col("chapter_id")),
        F.lit("\nWINDOW_ANALYSES_JSON:\n"), F.to_json("window_results"),
    )
    consolidated = (
        consolidation_inputs.withColumn("prompt", consolidation_prompt)
        .withColumn(
            "ai",
            F.when(
                F.col("failed_windows") == 0,
                F.expr(
                    f"ai_query('{settings.model_endpoint}', prompt, "
                    f"responseFormat => '{FINAL_SCHEMA}', failOnError => false)"
                ),
            ),
        )
        .withColumn("parsed", F.from_json("ai.result", FINAL_SCHEMA))
        .persist()
    )
    consolidated.count()
    plan_rows = consolidated.select(
        "chapter_key", "book_id", "chapter_id", "source_hash",
        F.col("ai.result").alias("raw_ai_result"),
        F.lit(settings.model_endpoint).alias("model_endpoint"),
        F.lit(settings.prompt_version).alias("prompt_version"),
        F.when(F.col("failed_windows") > 0, "FAILED")
        .when(F.col("parsed.organization").isNull(), "FAILED")
        .otherwise("SUCCEEDED").alias("processing_status"),
        F.when(F.col("failed_windows") > 0, F.concat(F.lit("Failed window count: "), F.col("failed_windows")))
        .otherwise(F.col("ai.errorMessage")).alias("processing_error"),
        F.current_timestamp().alias("batch_time"),
    )
    _merge_plans(spark, settings, plan_rows)

    valid = consolidated.filter(F.col("parsed.organization").isNotNull())
    units = valid.select(
        "book_id", "chapter_id", "source_hash", "pages",
        F.posexplode("parsed.organization.units").alias("unit_pos", "unit"),
    )
    learning = units.select(
        F.sha2(F.concat_ws("|", "book_id", "chapter_id", F.col("unit.unit_id"), F.lit("LEARNING_UNIT")), 256).alias("fine_page_id"),
        "book_id", "chapter_id", F.col("unit.unit_id").alias("unit_id"),
        F.lit("LEARNING_UNIT").alias("page_type"), (F.col("unit_pos") + 1).cast("int").alias("sequence_number"),
        F.col("unit.title").alias("title"),
        F.concat_ws("\n\n", F.col("unit.knowledge_markdown"), F.concat_ws("\n\n", "unit.examples_markdown"),
                    F.concat_ws("\n\n", F.transform("unit.exercises", lambda e: e.markdown))).alias("markdown_content"),
        F.col("unit.source_document_ids").alias("source_document_ids"), F.lit(False).alias("preserved_original"),
        "source_hash", F.lit(settings.model_endpoint).alias("model_endpoint"),
        F.lit(settings.prompt_version).alias("prompt_version"), "pages",
    )
    exercises = units.select(
        "book_id", "chapter_id", "source_hash", "pages", F.col("unit.unit_id").alias("unit_id"),
        F.posexplode("unit.exercises").alias("exercise_pos", "exercise"),
    ).select(
        F.sha2(F.concat_ws("|", "book_id", "chapter_id", "unit_id", F.col("exercise.exercise_id")), 256).alias("fine_page_id"),
        "book_id", "chapter_id", "unit_id", F.lit("EXERCISE").alias("page_type"),
        (F.col("exercise_pos") + 1).cast("int").alias("sequence_number"), F.col("exercise.title").alias("title"),
        F.col("exercise.markdown").alias("markdown_content"), F.col("exercise.source_document_ids").alias("source_document_ids"),
        F.col("exercise.preserve_original").alias("preserved_original"), "source_hash",
        F.lit(settings.model_endpoint).alias("model_endpoint"), F.lit(settings.prompt_version).alias("prompt_version"), "pages",
    )
    generated = learning.unionByName(exercises)
    original_page = F.element_at(
        F.filter("pages", lambda p: F.array_contains(F.col("source_document_ids"), p.document_id)), 1
    )
    generated = generated.withColumn(
        "markdown_content",
        F.when(F.col("preserved_original") & (F.size("source_document_ids") == 1), original_page.markdown_content)
        .otherwise(F.col("markdown_content")),
    ).withColumn(
        "image_references",
        F.flatten(F.transform(
            F.filter("pages", lambda p: F.array_contains(F.col("source_document_ids"), p.document_id)),
            lambda p: p.image_references,
        )),
    ).drop("pages").withColumn("is_active", F.lit(True)).withColumn("batch_time", F.current_timestamp()).persist()
    generated.count()

    target = DeltaTable.forName(spark, settings.fine_pages_table)
    regenerated = valid.select("book_id", "chapter_id").distinct()
    (target.alias("t").merge(
        regenerated.alias("c"),
        "t.book_id=c.book_id AND t.chapter_id=c.chapter_id AND t.is_active=true",
    ).whenMatchedUpdate(set={"is_active": "false", "updated_at": "current_timestamp()"}).execute())

    values = {
        "fine_page_id": "s.fine_page_id", "book_id": "s.book_id", "chapter_id": "s.chapter_id",
        "unit_id": "s.unit_id", "page_type": "s.page_type", "sequence_number": "s.sequence_number",
        "title": "s.title", "markdown_content": "s.markdown_content",
        "source_document_ids": "s.source_document_ids", "image_references": "s.image_references",
        "preserved_original": "s.preserved_original", "is_active": "s.is_active",
        "source_hash": "s.source_hash", "model_endpoint": "s.model_endpoint",
        "prompt_version": "s.prompt_version",
    }
    updates = {key: value for key, value in values.items() if key != "fine_page_id"}
    updates["updated_at"] = "s.batch_time"
    inserts = dict(values)
    inserts.update(created_at="s.batch_time", updated_at="s.batch_time")
    (target.alias("t").merge(generated.alias("s"), "t.fine_page_id=s.fine_page_id")
     .whenMatchedUpdate(set=updates).whenNotMatchedInsert(values=inserts).execute())

    print(
        f"Window analyses={analyzed.count()}; consolidated chapters={valid.count()}; "
        f"active fine-grained pages generated={generated.count()}"
    )


if __name__ == "__main__":
    main()
