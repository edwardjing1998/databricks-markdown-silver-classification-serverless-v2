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


def test_scene_auto_attaches_existing_image_block():
    page = sample()
    blocks = json.loads(page)["blocks"]
    response = {"scenes": [{
        "scene_type": "SCRIPTURE_STORY", "block_ids": [blocks[1]["block_id"]],
        "image_links": ["figures/light.png"]
    }]}
    parsed = validate_scenes(page, json.dumps(response))
    assert parsed["status"] == "SUCCEEDED", parsed["error"]
    scene = parsed["scenes"][0]
    assert scene["block_ids"] == [blocks[0]["block_id"], blocks[1]["block_id"]]
    assert "![Light]" in scene["markdown_content"]


def test_prevents_stealing_image_block_from_another_scene():
    page = sample()
    blocks = json.loads(page)["blocks"]
    response = {"scenes": [
        {"scene_type": "OTHER", "block_ids": [blocks[0]["block_id"]]},
        {"scene_type": "SCRIPTURE_STORY", "block_ids": [blocks[1]["block_id"]],
         "image_links": ["figures/light.png"]},
    ]}
    parsed = validate_scenes(page, json.dumps(response))
    assert parsed["status"] == "FAILED"
    assert "already assigned" in parsed["error"]


def test_creation_page_image_blocks_recovered():
    document = "bible_studies/old_testament_stories_6/page_20261019191003762"
    markdown = ("# The Creation\n\n![Figure 1](figures/figure-1.png)\n\n"
                "The Lord created the earth.\n\n"
                "![Figure 2](figures/figure-2.png)\n\n"
                "On the first day, light was separated.\n\n"
                "![Figure 3](figures/figure-3.png)\n\n"
                "On the second day, the waters were divided.")
    page = build_page_input(document, 8, markdown)
    blocks = json.loads(page)["blocks"]
    response = {"scenes": [
        {"scene_type": "DOCTRINAL_TEACHING", "block_ids": [blocks[2]["block_id"]], "image_links": ["figures/figure-1.png"]},
        {"scene_type": "SCRIPTURE_STORY", "block_ids": [blocks[4]["block_id"]], "image_links": ["figures/figure-2.png"]},
        {"scene_type": "SCRIPTURE_STORY", "block_ids": [blocks[6]["block_id"]], "image_links": ["figures/figure-3.png"]},
    ]}
    parsed = validate_scenes(page, json.dumps(response))
    assert parsed["status"] == "SUCCEEDED", parsed["error"]
    assert parsed["unassigned_image_links"] == []
    for i, scene in enumerate(parsed["scenes"]):
        assert blocks[2*i+1]["block_id"] in scene["block_ids"]
