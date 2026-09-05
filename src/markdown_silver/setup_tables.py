from pyspark.sql import SparkSession
from markdown_silver.config import parse_settings


def main() -> None:
    s = parse_settings()
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {s.catalog}.{s.silver_schema}")
    spark.sql(f"""
      CREATE TABLE IF NOT EXISTS {s.sections_table} (
        section_id STRING NOT NULL, document_id STRING NOT NULL,
        book_id STRING, chapter_id STRING, page_id STRING, page_number BIGINT,
        section_number INT, heading_level INT, title STRING, content STRING,
        image_links ARRAY<STRING>, image_references ARRAY<STRUCT<figure_id: STRING,
          original_link: STRING, blob_path: STRING, file_name: STRING,
          content_type: STRING, file_size: BIGINT, content_hash: STRING,
          source_modified_at: TIMESTAMP>>,
        section_type STRING, subject STRING, topic STRING, difficulty STRING,
        language STRING, rule_confidence DOUBLE, classification_confidence DOUBLE,
        classification_method STRING, model_endpoint STRING,
        prompt_version STRING, source_content_hash STRING,
        classification_status STRING, classification_error STRING,
        created_at TIMESTAMP, updated_at TIMESTAMP,
        CONSTRAINT document_sections_pk PRIMARY KEY (section_id) NOT ENFORCED
      ) USING DELTA TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)
    spark.sql(f"""
      CREATE TABLE IF NOT EXISTS {s.chapter_windows_table} (
        window_id STRING NOT NULL, chapter_key STRING, book_id STRING,
        chapter_id STRING, window_number INT, first_page BIGINT, last_page BIGINT,
        source_document_ids ARRAY<STRING>, source_hash STRING, input_characters BIGINT,
        raw_ai_result STRING, model_endpoint STRING, prompt_version STRING,
        processing_status STRING, processing_error STRING,
        created_at TIMESTAMP, updated_at TIMESTAMP,
        CONSTRAINT chapter_window_analyses_pk PRIMARY KEY (window_id) NOT ENFORCED
      ) USING DELTA TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)
    spark.sql(f"""
      CREATE TABLE IF NOT EXISTS {s.chapter_plans_table} (
        chapter_key STRING NOT NULL, book_id STRING, chapter_id STRING,
        source_hash STRING, raw_ai_result STRING, model_endpoint STRING,
        prompt_version STRING, processing_status STRING, processing_error STRING,
        created_at TIMESTAMP, updated_at TIMESTAMP,
        CONSTRAINT chapter_plans_pk PRIMARY KEY (chapter_key) NOT ENFORCED
      ) USING DELTA TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)
    spark.sql(f"""
      CREATE TABLE IF NOT EXISTS {s.fine_pages_table} (
        fine_page_id STRING NOT NULL, book_id STRING, chapter_id STRING,
        unit_id STRING, page_type STRING, sequence_number INT, title STRING,
        markdown_content STRING, source_document_ids ARRAY<STRING>,
        image_references ARRAY<STRUCT<figure_id: STRING, original_link: STRING,
          blob_path: STRING, file_name: STRING, content_type: STRING,
          file_size: BIGINT, content_hash: STRING, source_modified_at: TIMESTAMP>>,
        preserved_original BOOLEAN, is_active BOOLEAN, source_hash STRING, model_endpoint STRING,
        prompt_version STRING, created_at TIMESTAMP, updated_at TIMESTAMP,
        CONSTRAINT fine_grained_pages_pk PRIMARY KEY (fine_page_id) NOT ENFORCED
      ) USING DELTA TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)
    fine_columns = {column.name for column in spark.catalog.listColumns(s.fine_pages_table)}
    if "is_active" not in fine_columns:
        spark.sql(f"ALTER TABLE {s.fine_pages_table} ADD COLUMNS (is_active BOOLEAN)")
    spark.sql(f"UPDATE {s.fine_pages_table} SET is_active=true WHERE is_active IS NULL")
    print(
        "Silver tables are ready: "
        f"{s.sections_table}, {s.chapter_windows_table}, "
        f"{s.chapter_plans_table}, {s.fine_pages_table}"
    )


if __name__ == "__main__":
    main()
