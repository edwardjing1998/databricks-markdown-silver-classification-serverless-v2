from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import SparkSession, functions as F, types as T
from markdown_silver.config import parse_settings
from markdown_silver.parser import parse_markdown, section_id

IMAGE_TYPE = T.ArrayType(T.StructType([
    T.StructField("figure_id", T.StringType()), T.StructField("original_link", T.StringType()),
    T.StructField("blob_path", T.StringType()), T.StructField("file_name", T.StringType()),
    T.StructField("content_type", T.StringType()), T.StructField("file_size", T.LongType()),
    T.StructField("content_hash", T.StringType()), T.StructField("source_modified_at", T.TimestampType()),
]))
SCHEMA = T.ArrayType(T.StructType([
    T.StructField("section_number", T.IntegerType(), False),
    T.StructField("heading_level", T.IntegerType()), T.StructField("title", T.StringType()),
    T.StructField("content", T.StringType(), False),
    T.StructField("image_links", T.ArrayType(T.StringType()), False),
    T.StructField("rule_type", T.StringType(), False),
    T.StructField("rule_confidence", T.DoubleType(), False),
]))


@F.udf(SCHEMA)
def parse_udf(value):
    return [tuple(vars(x).values()) for x in parse_markdown(value)]


@F.udf(T.StringType())
def id_udf(document_id, number):
    return section_id(document_id, number)


def main() -> None:
    s = parse_settings()
    spark = SparkSession.builder.getOrCreate()
    source = spark.table(s.bronze_table).filter(F.col("ingestion_status") == "INGESTED")
    sections = (source.withColumn("section", F.explode(parse_udf("markdown_content")))
      .select("document_id", "book_id", "chapter_id", "page_id", "page_number",
        "markdown_content_hash", "image_references", "section.*")
      .withColumn("section_id", id_udf("document_id", "section_number"))
      .withColumn("matched_images", F.expr("filter(image_references, x -> array_contains(image_links, x.original_link))"))
      .withColumn("now", F.current_timestamp()))
    target = DeltaTable.forName(spark, s.sections_table)
    (target.alias("t").merge(sections.alias("s"), "t.section_id=s.section_id")
      .whenMatchedUpdate(condition="t.source_content_hash <> s.markdown_content_hash", set={
        "title":"s.title", "content":"s.content", "heading_level":"s.heading_level",
        "image_links":"s.image_links", "image_references":"s.matched_images",
        "section_type":"s.rule_type", "classification_confidence":"s.rule_confidence",
        "rule_confidence":"s.rule_confidence",
        "subject":"NULL", "topic":"NULL", "difficulty":"NULL", "language":"NULL",
        "classification_method":"'RULE'", "model_endpoint":"NULL", "prompt_version":"NULL",
        "source_content_hash":"s.markdown_content_hash", "classification_status":"'PENDING'",
        "classification_error":"NULL", "updated_at":"s.now"})
      .whenNotMatchedInsert(values={
        "section_id":"s.section_id", "document_id":"s.document_id", "book_id":"s.book_id",
        "chapter_id":"s.chapter_id", "page_id":"s.page_id", "page_number":"s.page_number",
        "section_number":"s.section_number", "heading_level":"s.heading_level", "title":"s.title",
        "content":"s.content", "image_links":"s.image_links", "image_references":"s.matched_images",
        "section_type":"s.rule_type", "classification_confidence":"s.rule_confidence",
        "rule_confidence":"s.rule_confidence",
        "classification_method":"'RULE'", "source_content_hash":"s.markdown_content_hash",
        "classification_status":"'PENDING'", "created_at":"s.now", "updated_at":"s.now"})
      .execute())


if __name__ == "__main__":
    main()
