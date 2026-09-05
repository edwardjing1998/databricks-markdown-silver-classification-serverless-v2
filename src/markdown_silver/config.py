from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    catalog: str
    bronze_schema: str
    silver_schema: str
    model_endpoint: str
    enable_ai: bool
    confidence_threshold: float
    prompt_version: str
    max_pages_per_window: int
    window_overlap_pages: int
    max_window_characters: int

    @property
    def bronze_table(self) -> str:
        return f"{self.catalog}.{self.bronze_schema}.markdown_documents"

    @property
    def sections_table(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.document_sections"

    @property
    def chapter_plans_table(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.chapter_plans"

    @property
    def chapter_windows_table(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.chapter_window_analyses"

    @property
    def fine_pages_table(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.fine_grained_pages"


def parse_settings() -> Settings:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="education_rag")
    parser.add_argument("--bronze-schema", default="bronze")
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--model-endpoint", default="system.ai.gpt-oss-20b")
    parser.add_argument("--enable-ai", default="true")
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--prompt-version", default="v2-windowed")
    parser.add_argument("--max-pages-per-window", type=int, default=15)
    parser.add_argument("--window-overlap-pages", type=int, default=2)
    parser.add_argument("--max-window-characters", type=int, default=40000)
    args = parser.parse_args()
    return Settings(
        args.catalog, args.bronze_schema, args.silver_schema,
        args.model_endpoint, args.enable_ai.lower() == "true",
        args.confidence_threshold, args.prompt_version,
        args.max_pages_per_window, args.window_overlap_pages,
        args.max_window_characters,
    )
