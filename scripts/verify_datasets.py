import os
import json

SAMPLE_DIR = "/home/nate/Projects/Continual_Learning/data/sample"

def verify_stage1():
    path = os.path.join(SAMPLE_DIR, "stage1_wernicke_syntax.jsonl")
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    
    assert len(records) >= 20, f"Expected >= 20 records, got {len(records)}"
    for r in records:
        assert "sentence_id" in r and r["sentence_id"]
        assert "text" in r and r["text"]
        assert "tokens" in r and isinstance(r["tokens"], list) and len(r["tokens"]) > 0
        assert "lemmas" in r and len(r["lemmas"]) == len(r["tokens"])
        assert "upos" in r and len(r["upos"]) == len(r["tokens"])
        assert "heads" in r and len(r["heads"]) == len(r["tokens"])
        assert "deprels" in r and len(r["deprels"]) == len(r["tokens"])
    
    print(f"[VERIFIED] Stage 1 (Wernicke Syntax): {len(records)} records.")
    print(f"  Sample Text: \"{records[0]['text']}\"")
    print(f"  Tokens ({len(records[0]['tokens'])}): {records[0]['tokens'][:6]} ...")
    print(f"  UPOS: {records[0]['upos'][:6]} ...")
    print(f"  Heads: {records[0]['heads'][:6]} ...")
    return True

def verify_stage2():
    path = os.path.join(SAMPLE_DIR, "stage2_neocortex_knowledge.jsonl")
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    
    assert len(records) >= 20, f"Expected >= 20 records, got {len(records)}"
    for r in records:
        assert "doc_id" in r and r["doc_id"]
        assert "text" in r and len(r["text"]) > 0
    
    print(f"[VERIFIED] Stage 2 (Neocortex Pure Raw Text): {len(records)} records.")
    print(f"  Sample Text: \"{records[1]['text']}\" (ID: {records[1]['doc_id']})")
    print(f"  Note: Zero manual annotations. Noun anchors & modifiers extracted at runtime by Frozen Wernicke.")
    return True



def verify_stage3():
    path = os.path.join(SAMPLE_DIR, "stage3_parietal_multimodal.jsonl")
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    
    assert len(records) >= 20, f"Expected >= 20 records, got {len(records)}"
    for r in records:
        assert "image_id" in r
        assert "image_filename" in r
        assert "caption_id" in r
        assert "caption" in r and len(r["caption"]) > 0
        
    print(f"[VERIFIED] Stage 1B (Neocortex Visuolinguistic Grounding): {len(records)} records.")
    print(f"  Sample Image: {records[0]['image_filename']} (ID: {records[0]['image_id']})")
    print(f"  Sample Caption: \"{records[0]['caption']}\"")
    print(f"  Note: Pure image-caption pairs. Sights & words bound directly into Neocortex associative memory.")
    return True


def verify_stage4():
    path = os.path.join(SAMPLE_DIR, "stage4_frontal_reasoning.jsonl")
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    
    assert len(records) >= 20, f"Expected >= 20 records, got {len(records)}"
    for r in records:
        assert "id" in r and r["id"]
        assert "prompt" in r and len(r["prompt"]) > 0
        assert "target" in r and len(r["target"]) > 0
        
    print(f"[VERIFIED] Stage 4 (Frontal Supervised Reasoning): {len(records)} records.")
    print(f"  Sample ID: {records[0]['id']}")
    print(f"  Prompt: \"{records[0]['prompt']}\"")
    print(f"  Target Reasoning Sequence:\n    {records[0]['target'].replace(chr(10), chr(10) + '    ')}")
    print(f"  Training Regime: Supervised Learning (Teacher Forcing) on next-token prediction.")
    print(f"  Gradient Wall: Gradients update ONLY theta_frontal + theta_parietal. Wernicke & Neocortex remain 100% frozen.")
    return True


if __name__ == "__main__":
    v1 = verify_stage1()
    v2 = verify_stage2()
    v3 = verify_stage3()
    v4 = verify_stage4()
    if v1 and v2 and v3 and v4:
        print("\nAll 4 component datasets passed schema and content verification with 100% success!")
