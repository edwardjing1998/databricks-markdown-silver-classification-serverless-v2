
"""Independent Bible-study Databricks Serverless pipeline.

Run via: setup-bible-tables -> classify-bible-scenes -> verify-bible-scenes.
Reads the existing Bronze schema by configurable catalog; never writes math Silver tables.
"""
from __future__ import annotations
import argparse
import re
from delta.tables import DeltaTable
from pyspark.sql import SparkSession, functions as F, types as T
from markdown_classification.bible_scenes import build_page_input, bible_prompt, validate_scenes
from markdown_classification.source_filters import parse_book_ids

SCENE_STRUCT = T.StructType([
    T.StructField("scene_id",T.StringType()), T.StructField("scene_type",T.StringType()),
    T.StructField("title",T.StringType()), T.StructField("caption",T.StringType()),
    T.StructField("scripture_references",T.ArrayType(T.StringType())),
    T.StructField("block_ids",T.ArrayType(T.StringType())),
    T.StructField("image_links",T.ArrayType(T.StringType())),
    T.StructField("markdown_content",T.StringType()), T.StructField("confidence",T.DoubleType())])
RESULT_SCHEMA=T.StructType([
    T.StructField("status",T.StringType()),T.StructField("error",T.StringType()),
    T.StructField("scenes",T.ArrayType(SCENE_STRUCT)),
    T.StructField("unassigned_image_links",T.ArrayType(T.StringType()))])

def arguments():
    p=argparse.ArgumentParser()
    p.add_argument("--catalog",default="education_rag")
    p.add_argument("--bronze-schema",default="bronze")
    p.add_argument("--silver-schema",default="silver")
    p.add_argument("--model-endpoint",default="databricks-gpt-oss-120b")
    p.add_argument("--prompt-version",default="bible-scenes-v1")
    p.add_argument("--confidence-threshold",type=float,default=0.70)
    p.add_argument("--include-book-ids",default="bible_studies")
    args=p.parse_args()
    for name in (args.catalog,args.bronze_schema,args.silver_schema):
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", name):
            p.error("Catalog and schema names must be SQL identifiers")
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9.-]*",args.model_endpoint):
        p.error("Invalid model endpoint")
    if not 0 <= args.confidence_threshold <= 1:
        p.error("Invalid confidence threshold")
    return args

def names(args):
    return (f"{args.catalog}.{args.bronze_schema}.markdown_documents",
            f"{args.catalog}.{args.silver_schema}.bible_page_analysis",
            f"{args.catalog}.{args.silver_schema}.bible_story_scenes")

def setup_main():
    a=arguments(); spark=SparkSession.builder.getOrCreate(); _,analysis,scenes=names(a)
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {a.catalog}.{a.silver_schema}")
    spark.sql(f"""CREATE TABLE IF NOT EXISTS {analysis} (
        analysis_id STRING NOT NULL, document_id STRING, book_id STRING, chapter_id STRING,
        page_number BIGINT, source_hash STRING, prompt_version STRING, model_endpoint STRING,
        page_json STRING, raw_ai_result STRING, processing_status STRING, processing_error STRING,
        unassigned_image_links ARRAY<STRING>, updated_at TIMESTAMP
        ) USING DELTA""")
    spark.sql(f"""CREATE TABLE IF NOT EXISTS {scenes} (
        scene_id STRING NOT NULL, analysis_id STRING, document_id STRING, book_id STRING,
        chapter_id STRING, page_number BIGINT, sequence_number INT, scene_type STRING,
        title STRING, caption STRING, scripture_references ARRAY<STRING>,
        block_ids ARRAY<STRING>, image_links ARRAY<STRING>, markdown_content STRING,
        classification_confidence DOUBLE, review_required BOOLEAN, source_hash STRING,
        is_active BOOLEAN, updated_at TIMESTAMP
        ) USING DELTA""")
    print(f"Bible tables ready: {analysis} and {scenes}")

def classify_main():
    a=arguments(); spark=SparkSession.builder.getOrCreate(); bronze,analysis,scenes=names(a)
    make_json=F.udf(build_page_input,T.StringType())
    source=(spark.table(bronze).filter((F.col("ingestion_status")=="INGESTED") & F.col("markdown_content").isNotNull())
        .filter(F.col("book_id").isin(*parse_book_ids(a.include_book_ids)))
        .select("document_id","book_id","chapter_id","page_number","markdown_content")
        .withColumn("source_hash",F.sha2("markdown_content",256))
        .withColumn("analysis_id",F.sha2(F.concat_ws("|",F.col("document_id"),F.col("source_hash"),F.lit(a.prompt_version),F.lit(a.model_endpoint)),256))
        .withColumn("page_json",make_json("document_id","page_number","markdown_content")))
    saved=spark.table(analysis).select("analysis_id",F.col("processing_status").alias("old_status"))
    pending=(source.join(saved,"analysis_id","left").filter(F.coalesce(F.col("old_status"),F.lit("FAILED"))!="SUCCEEDED"))
    validator=F.udf(validate_scenes,RESULT_SCHEMA)
    if pending.limit(1).count():
        queried=(pending.withColumn("prompt",F.concat(F.lit(bible_prompt()+"\nINPUT PAGE JSON:\n"),F.col("page_json")))
            .withColumn("ai",F.expr(f"ai_query('{a.model_endpoint}', prompt, failOnError => false)"))
            .withColumn("result",validator(F.col("page_json"),F.col("ai.result"))))
        updates=(queried.select("analysis_id","document_id","book_id","chapter_id","page_number","source_hash","page_json",
                F.col("ai.result").alias("raw_ai_result"),F.col("result.status").alias("processing_status"),
                F.coalesce(F.col("ai.errorMessage"),F.col("result.error")).alias("processing_error"),
                F.col("result.unassigned_image_links").alias("unassigned_image_links"))
            .withColumn("prompt_version",F.lit(a.prompt_version)).withColumn("model_endpoint",F.lit(a.model_endpoint))
            .withColumn("updated_at",F.current_timestamp()))
        cols=updates.columns
        (DeltaTable.forName(spark,analysis).alias("t").merge(updates.alias("s"),"t.analysis_id=s.analysis_id")
         .whenMatchedUpdate(set={x:f"s.{x}" for x in cols if x!="analysis_id"})
         .whenNotMatchedInsert(values={x:f"s.{x}" for x in cols}).execute())
    # Read persisted model responses; avoid re-evaluating ai_query during downstream writes.
    # Join only on unique analysis IDs. The persisted analysis table owns
    # document_id and all other metadata used to materialize scene rows.
    # Including current.document_id would leave two ambiguous columns after
    # the join and fail when scene_rows selects document_id.
    current_ids=source.select("analysis_id").distinct()
    successful=(spark.table(analysis).join(current_ids,"analysis_id","inner")
      .filter(F.col("processing_status")=="SUCCEEDED")
      .withColumn("parsed",validator("page_json","raw_ai_result")))
    scene_rows=(successful.select("analysis_id","document_id","book_id","chapter_id","page_number","source_hash",
       F.posexplode("parsed.scenes").alias("position","scene"))
       .withColumn("scene_id",F.sha2(F.concat_ws("|",F.col("analysis_id"),F.col("scene.scene_id")),256))
       .select("scene_id","analysis_id","document_id","book_id","chapter_id","page_number",
          (F.col("position")+1).cast("int").alias("sequence_number"),
          F.col("scene.scene_type").alias("scene_type"),F.col("scene.title").alias("title"),
          F.col("scene.caption").alias("caption"),F.col("scene.scripture_references").alias("scripture_references"),
          F.col("scene.block_ids").alias("block_ids"),F.col("scene.image_links").alias("image_links"),
          F.col("scene.markdown_content").alias("markdown_content"),
          F.col("scene.confidence").alias("classification_confidence"),
          (F.col("scene.confidence")<F.lit(a.confidence_threshold)).alias("review_required"),
          "source_hash",F.lit(True).alias("is_active"),F.current_timestamp().alias("updated_at")))
    if scene_rows.groupBy("scene_id").count().filter("count > 1").limit(1).count():
        raise RuntimeError("Duplicate scene identifiers")
    # Deactivate superseded rows for current source documents (not unrelated books).
    scope=source.select("document_id","analysis_id").distinct()
    current_ids = scope.select("document_id", "analysis_id").distinct()
    old=(spark.table(scenes).filter("is_active=true")
        .join(scope.select("document_id").distinct(), "document_id", "inner")
        .join(current_ids, ["document_id", "analysis_id"], "left_anti")
        .select("scene_id").distinct())
    if old.limit(1).count():
        (DeltaTable.forName(spark,scenes).alias("t").merge(old.alias("s"),"t.scene_id=s.scene_id")
          .whenMatchedUpdate(set={"is_active":"false","updated_at":"current_timestamp()"}).execute())
    if scene_rows.limit(1).count():
        cols=scene_rows.columns
        (DeltaTable.forName(spark,scenes).alias("t").merge(scene_rows.alias("s"),"t.scene_id=s.scene_id")
         .whenMatchedUpdate(set={x:f"s.{x}" for x in cols if x!="scene_id"})
         .whenNotMatchedInsert(values={x:f"s.{x}" for x in cols}).execute())
    print("Bible scene classification complete")

def verify_main():
    a=arguments(); spark=SparkSession.builder.getOrCreate(); _,analysis,scenes=names(a)
    print("Bible page statuses:")
    spark.table(analysis).groupBy("processing_status").count().show()
    print("Bible scene categories:")
    spark.table(scenes).filter("is_active=true").groupBy("scene_type","review_required").count().show()
    failed=spark.table(analysis).filter("processing_status != 'SUCCEEDED'").count()
    if failed: raise RuntimeError(f"{failed} Bible page analyses failed")
