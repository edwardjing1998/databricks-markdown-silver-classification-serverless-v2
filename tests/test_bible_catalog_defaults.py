from pathlib import Path


def test_bible_catalog_defaults_are_education_rag():
    root = Path(__file__).resolve().parents[1]
    bundle = (root / "databricks.yml").read_text()
    code = (root / "src/markdown_classification/bible_job.py").read_text()
    assert "bible_catalog:\n    default: education_rag" in bundle
    assert 'p.add_argument("--catalog",default="education_rag")' in code
    assert "bible_rag" not in bundle
