"""Load the versioned English classification prompt packaged with the application."""
from importlib.resources import files


def classification_prompt(max_pages_per_window: int) -> str:
    if max_pages_per_window < 1:
        raise ValueError("max_pages_per_window must be positive")
    instructions = (files("markdown_classification.prompts") / "geometry_classifier_v1.txt").read_text(encoding="utf-8")
    return f"This request contains at most {max_pages_per_window} pages.\n" + instructions
