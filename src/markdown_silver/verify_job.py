from pyspark.sql import SparkSession, functions as F
from markdown_silver.config import parse_settings


def main() -> None:
    s = parse_settings()
    spark = SparkSession.builder.getOrCreate()
    pages = spark.table(s.fine_pages_table).filter(F.col("is_active") == True)
    plans = spark.table(s.chapter_plans_table)
    duplicates = pages.groupBy("fine_page_id").count().filter("count > 1").count()
    failures = plans.filter(F.col("processing_status") == "FAILED").count()
    total = pages.count()
    exercises = pages.filter(F.col("page_type") == "EXERCISE").count()
    units = pages.filter(F.col("page_type") == "LEARNING_UNIT").count()
    print(
        f"Silver verification: fine_pages={total}, learning_units={units}, "
        f"exercises={exercises}, duplicate_ids={duplicates}, failed_chapters={failures}"
    )
    if duplicates or failures:
        raise RuntimeError("Silver verification failed; inspect chapter_plans")


if __name__ == "__main__":
    main()
