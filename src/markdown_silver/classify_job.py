from __future__ import annotations

import json
from delta.tables import DeltaTable
from pyspark.sql import SparkSession, functions as F, types as T
from markdown_silver.config import parse_settings

def main() -> None:
    s = parse_settings()
    spark = SparkSession.builder.getOrCreate()
    needs_refresh = (
        F.col("classification_status").isin("PENDING", "RULE_ONLY", "FAILED") |
        ((F.col("classification_method") == "AI") &
         ((F.coalesce(F.col("model_endpoint"), F.lit("")) != F.lit(s.model_endpoint)) |
          (F.coalesce(F.col("prompt_version"), F.lit("")) != F.lit(s.prompt_version))))
    )
    pending = spark.table(s.sections_table).filter(
        needs_refresh & (F.col("rule_confidence") < F.lit(s.confidence_threshold)))
    if pending.limit(1).count() == 0:
        spark.sql(f"UPDATE {s.sections_table} SET classification_status='CLASSIFIED' WHERE classification_status='PENDING'")
        return
    if not s.enable_ai:
        spark.sql(f"UPDATE {s.sections_table} SET classification_status='RULE_ONLY' WHERE classification_status='PENDING'")
        return
    allowed = "TITLE, CHAPTER_HEADING, DEFINITION, THEOREM, EXAMPLE, EXERCISE, EXPLANATION, ANSWER, OTHER"
    prompt = F.concat(F.lit(
        "Classify this school mathematics Markdown section. Return JSON only with keys "
        "section_type, subject, topic, difficulty, language, confidence. section_type must be one of: "
        + allowed + ". confidence must be 0 to 1.\nTITLE: "),
        F.coalesce("title", F.lit("")), F.lit("\nCONTENT:\n"), F.substring("content", 1, 12000))
    output_schema = (
        "STRUCT<classification:STRUCT<section_type:STRING,subject:STRING,"
        "topic:STRING,difficulty:STRING,language:STRING,confidence:DOUBLE>>"
    )
    queried = pending.withColumn("prompt", prompt).withColumn(
        "raw_result", F.expr(
            f"ai_query('{s.model_endpoint}', prompt, "
            f"responseFormat => '{output_schema}', failOnError => false)"
        ))
    classified = (queried
      .withColumn(
          "parsed_result",
          F.from_json(F.col("raw_result.result"), output_schema),
      )
      .select(
          "section_id",
          "parsed_result.classification.*",
          F.col("raw_result.errorMessage").alias("errorMessage"),
      )
      .withColumn("now", F.current_timestamp()))
    target = DeltaTable.forName(spark, s.sections_table)
    (target.alias("t").merge(classified.alias("s"), "t.section_id=s.section_id")
      .whenMatchedUpdate(set={
        "section_type":"coalesce(s.section_type, t.section_type)", "subject":"s.subject",
        "topic":"s.topic", "difficulty":"s.difficulty", "language":"s.language",
        "classification_confidence":"coalesce(s.confidence, t.classification_confidence)",
        "classification_method":"'AI'", "model_endpoint":json.dumps(s.model_endpoint),
        "prompt_version":json.dumps(s.prompt_version),
        "classification_status":"CASE WHEN s.section_type IS NULL THEN 'FAILED' ELSE 'CLASSIFIED' END",
        "classification_error":"s.errorMessage",
        "updated_at":"s.now"}).execute())
    spark.sql(f"UPDATE {s.sections_table} SET classification_status='CLASSIFIED' WHERE classification_status='PENDING'")


if __name__ == "__main__":
    main()
