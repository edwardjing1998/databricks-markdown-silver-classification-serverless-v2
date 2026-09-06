from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    catalog: str
    bronze_schema: str
    silver_schema: str
    model_endpoint: str
    prompt_version: str
    max_pages_per_window: int
    confidence_threshold: float

    @property
    def bronze_table(self) -> str:
        return f"{self.catalog}.{self.bronze_schema}.markdown_documents"

    @property
    def windows_table(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.page_classification_windows"

    @property
    def fine_pages_table(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.fine_grained_pages"


def parse_settings() -> Settings:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="education_rag")
    parser.add_argument("--bronze-schema", default="bronze")
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--model-endpoint", default="databricks-gpt-oss-120b")
    parser.add_argument("--prompt-version", default="v3-page-window-block-ids")
    parser.add_argument("--max-pages-per-window", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    args = parser.parse_args()
    if not 1 <= args.max_pages_per_window <= 20:
        parser.error("--max-pages-per-window must be between 1 and 5")
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be between 0 and 1")
    return Settings(
        args.catalog, args.bronze_schema, args.silver_schema,
        args.model_endpoint, args.prompt_version,
        args.max_pages_per_window, args.confidence_threshold,
    )
