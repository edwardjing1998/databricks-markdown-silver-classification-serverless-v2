from pyspark.sql import SparkSession, functions as F

from markdown_classification.classify_windows import build_current_windows
from markdown_classification.config import parse_settings


def main() -> None:
    s = parse_settings()
    spark = SparkSession.builder.getOrCreate()
    current = build_current_windows(spark, s).select("window_id").distinct()

    # Only the source windows presently produced from Bronze are relevant.
    # Failed historical revisions remain in Delta for audit, not verification.
    windows = current.join(
        spark.table(s.windows_table).select("window_id", "processing_status"),
        "window_id", "left",
    )
    failed_windows = windows.filter(
        F.col("processing_status").isNull()
        | (F.col("processing_status") != "SUCCEEDED")
    ).count()

    pages = (
        spark.table(s.fine_pages_table)
        .filter(
            (F.col("is_active") == True)
            & (F.col("prompt_version") == s.prompt_version)
            & (F.col("model_endpoint") == s.model_endpoint)
        )
        .join(current, "window_id", "left_semi")
    )
    duplicates = pages.groupBy("fine_page_id").count().filter(F.col("count") > 1).count()
    total = pages.count()
    review_required = pages.filter(F.col("classification_status") == "REVIEW_REQUIRED").count()
    print(
        f"Classification verification: fine_pages={total}, failed_windows={failed_windows}, "
        f"duplicate_ids={duplicates}, review_required={review_required}"
    )
    if failed_windows or duplicates:
        raise RuntimeError("Classification verification failed; inspect current page_classification_windows")


if __name__ == "__main__":
    main()
