from pyspark.sql import SparkSession, functions as F

from markdown_classification.config import parse_settings


def main() -> None:
    s = parse_settings()
    spark = SparkSession.builder.getOrCreate()
    windows = spark.table(s.windows_table).filter(F.col("prompt_version") == s.prompt_version)
    pages = spark.table(s.fine_pages_table).filter(
        (F.col("is_active") == True) & (F.col("prompt_version") == s.prompt_version)
    )
    failed_windows = windows.filter(F.col("processing_status") == "FAILED").count()
    duplicates = pages.groupBy("fine_page_id").count().filter(F.col("count") > 1).count()
    total = pages.count()
    review_required = pages.filter(F.col("classification_status") == "REVIEW_REQUIRED").count()
    print(
        f"Classification verification: fine_pages={total}, failed_windows={failed_windows}, "
        f"duplicate_ids={duplicates}, review_required={review_required}"
    )
    if failed_windows or duplicates:
        raise RuntimeError("Classification verification failed; inspect page_classification_windows")


if __name__ == "__main__":
    main()
