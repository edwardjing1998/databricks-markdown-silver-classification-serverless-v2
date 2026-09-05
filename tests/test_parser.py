from markdown_silver.parser import parse_markdown, rule_classify, section_id


SAMPLE = '''---
sourceBlob: "source/book-01/chapter-01/problem-01.png"
---
# Geometry Question 01

Find the missing angle.

![Diagram](figures/figure-1.png)

## Solution

Use the triangle angle sum.
'''


def test_parse_and_preserve_image_link():
    sections = parse_markdown(SAMPLE)
    assert len(sections) == 2
    assert sections[0].title == "Geometry Question 01"
    assert sections[0].image_links == ["figures/figure-1.png"]
    assert sections[1].rule_type == "ANSWER"


def test_stable_section_id():
    assert section_id("book-01/chapter-01/problem-01", 1) == section_id("book-01/chapter-01/problem-01", 1)


def test_ambiguous_text_uses_other():
    assert rule_classify(None, "Triangles have three sides.")[0] == "OTHER"

