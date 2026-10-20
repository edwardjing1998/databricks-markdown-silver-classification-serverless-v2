import json
from markdown_classification.bible_scenes import build_page_input

def test_bible_layout_included():
    result=json.loads(build_page_input("id",1,"Hello",json.dumps({"pages":[]})))
    assert result["layout"] == {"pages": []}

def test_old_input_still_supported():
    assert json.loads(build_page_input("id",1,"Hello"))["layout"] is None
