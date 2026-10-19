"""Additive guardrails for complex, mixed-layout textbook prompt support."""
from markdown_classification.prompt_loader import classification_prompt


def test_complex_textbook_sections_and_original_schema():
    prompt = classification_prompt(5)
    for term in (
        '知识要点', '例题精讲', '课后作业',
        'KNOWLEDGE', 'EXAMPLE', 'EXERCISE', 'OTHER',
        'block_ids', 'section_id', 'knowledge_id',
        'NOT raw pixels', 'd = 2r',
    ):
        assert term in prompt
    assert prompt.endswith('INPUT PAGES (JSON):\n')
    assert prompt.count('INPUT PAGES (JSON):') == 1


def test_existing_block_assignment_rules_remain():
    prompt = classification_prompt(5)
    assert 'Every input block_id MUST be assigned to exactly one section' in prompt
    assert 'Never invent block IDs' in prompt
    assert 'Never rewrite, translate, or summarize source Markdown' in prompt
