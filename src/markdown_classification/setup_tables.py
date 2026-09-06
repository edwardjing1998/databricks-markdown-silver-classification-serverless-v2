from pyspark.sql import SparkSession

from markdown_classification.config import parse_settings


def main() -> None:
    s = parse_settings()
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {s.catalog}.{s.silver_schema}")
    spark.sql(f"""
      CREATE TABLE IF NOT EXISTS {s.windows_table} (
        window_id STRING NOT NULL, chapter_key STRING, book_id STRING,
        chapter_id STRING, window_number INT, first_page BIGINT, last_page BIGINT,
        source_document_ids ARRAY<STRING>, source_hash STRING, pages_json STRING,
        input_characters BIGINT,
        raw_ai_result STRING, model_endpoint STRING, prompt_version STRING,
        processing_status STRING, processing_error STRING,
        created_at TIMESTAMP, updated_at TIMESTAMP,
        CONSTRAINT page_classification_windows_pk PRIMARY KEY (window_id) NOT ENFORCED
      ) USING DELTA TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)
    spark.sql(f"""
      CREATE TABLE IF NOT EXISTS {s.fine_pages_table} (
        fine_page_id STRING NOT NULL, window_id STRING, section_id STRING,
        book_id STRING, chapter_id STRING, page_type STRING, sequence_number INT,
        title STRING, knowledge_id STRING, markdown_content STRING,
        block_ids ARRAY<STRING>, source_document_ids ARRAY<STRING>,
        source_page_numbers ARRAY<BIGINT>, image_links ARRAY<STRING>,
        image_references_json STRING, preserved_original BOOLEAN,
        classification_confidence DOUBLE, classification_method STRING,
        classification_status STRING, classification_error STRING,
        is_active BOOLEAN, source_hash STRING, model_endpoint STRING,
        prompt_version STRING, created_at TIMESTAMP, updated_at TIMESTAMP,
        CONSTRAINT fine_grained_pages_pk PRIMARY KEY (fine_page_id) NOT ENFORCED
      ) USING DELTA TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)
    required = {
        "window_id": "STRING", "section_id": "STRING", "knowledge_id": "STRING",
        "block_ids": "ARRAY<STRING>", "source_page_numbers": "ARRAY<BIGINT>",
        "image_links": "ARRAY<STRING>", "image_references_json": "STRING",
        "classification_confidence": "DOUBLE", "classification_method": "STRING",
        "classification_status": "STRING", "classification_error": "STRING",
    }
    existing = {column.name for column in spark.catalog.listColumns(s.fine_pages_table)}
    missing = [f"{name} {data_type}" for name, data_type in required.items() if name not in existing]
    if missing:
        spark.sql(f"ALTER TABLE {s.fine_pages_table} ADD COLUMNS ({', '.join(missing)})")
    print(f"Classification tables are ready: {s.windows_table}, {s.fine_pages_table}")


if __name__ == "__main__":
    main()
