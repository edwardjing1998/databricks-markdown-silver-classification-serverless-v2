from markdown_classification.source_filters import parse_book_ids, source_allowed
from markdown_classification.config import Settings
from pathlib import Path


def test_book_ids():
    assert parse_book_ids("bible_studies, another ,bible_studies") == ("bible_studies", "another")


def test_isolation():
    assert not source_allowed("bible_studies", exclude="bible_studies")
    assert source_allowed("adam_math_ai", exclude="bible_studies")
    assert source_allowed("bible_studies", include="bible_studies")
    assert not source_allowed("adam_math_ai", include="bible_studies")


def test_math_default():
    s = Settings("education_rag", "bronze", "silver", "model", "v2", 5, .7)
    assert s.exclude_book_ids == "bible_studies"


def test_job_configuration_and_prompt():
    root = Path(__file__).resolve().parents[1]
    assert "exclude-book-ids: ${var.exclude_book_ids}" in (root / "resources/classification-job.yml").read_text()
    assert "include-book-ids: ${var.bible_include_book_ids}" in (root / "resources/bible-classification-job.yml").read_text()
    prompt = (root / "src/markdown_classification/prompts/geometry_classifier_v1.txt").read_text()
    assert "character-for-character" in prompt
