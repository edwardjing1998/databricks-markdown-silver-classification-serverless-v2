import json
from markdown_classification.bible_scenes import build_page_input,validate_scenes,bible_prompt

def sample():
    return build_page_input("doc-8",8,"![Light](figures/light.png)\n\nOn the first day, the Lord separated light from darkness.\n\n![Water](figures/water.png)\n\nOn the second day, He divided the waters.")

def test_two_scenes():
    page=json.loads(sample()); blocks=page["blocks"]
    result={"scenes":[{"scene_id":str(i),"scene_type":"SCRIPTURE_STORY","title":"Day", "caption":"Day", "block_ids":[block["block_id"]],"image_links":block["image_links"],"scripture_references":[],"confidence":.9} for i,block in enumerate(blocks)]}
    parsed=validate_scenes(sample(),json.dumps(result))
    assert parsed["status"]=="SUCCEEDED" and len(parsed["scenes"])==4
    assert parsed["unassigned_image_links"]==[]

def test_rejects_fabricated_image():
    page=json.loads(sample()); block=page["blocks"][0]
    response={"scenes":[{"scene_type":"SCRIPTURE_STORY","block_ids":[block["block_id"]],"image_links":["invented.png"]}]}
    assert validate_scenes(sample(),json.dumps(response))["status"]=="FAILED"

def test_rejects_duplicate_blocks():
    page=json.loads(sample()); block=page["blocks"][0]
    response={"scenes":[{"scene_type":"OTHER","block_ids":[block["block_id"]]},{"scene_type":"OTHER","block_ids":[block["block_id"]]}]}
    assert validate_scenes(sample(),json.dumps(response))["status"]=="FAILED"

def test_prompt_and_existing_math_types():
    assert "image-caption" in bible_prompt()
    from markdown_classification.blocks import ALLOWED_TYPES
    assert ALLOWED_TYPES=={"KNOWLEDGE","EXAMPLE","EXERCISE","OTHER"}
