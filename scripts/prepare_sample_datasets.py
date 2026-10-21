import os
import io
import json
import zlib
import zipfile
import urllib.request

SAMPLE_DIR = "/home/nate/Projects/Continual_Learning/data/sample"
os.makedirs(SAMPLE_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def fetch_stage1_ud_ewt():
    print("Fetching Stage 1: Universal Dependencies English-EWT...")
    url = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        # Read first ~64KB to extract ~25 complete sentences
        raw_text = resp.read(65536).decode('utf-8', errors='ignore')
    
    sentences = []
    current_sent = {
        'sentence_id': '',
        'text': '',
        'tokens': [],
        'lemmas': [],
        'upos': [],
        'heads': [],
        'deprels': []
    }
    
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            if current_sent['text'] and len(current_sent['tokens']) > 0:
                sentences.append(current_sent)
                if len(sentences) >= 25:
                    break
            current_sent = {
                'sentence_id': '',
                'text': '',
                'tokens': [],
                'lemmas': [],
                'upos': [],
                'heads': [],
                'deprels': []
            }
            continue
        
        if line.startswith("# sent_id ="):
            current_sent['sentence_id'] = line.split("=", 1)[1].strip()
        elif line.startswith("# text ="):
            current_sent['text'] = line.split("=", 1)[1].strip()
        elif not line.startswith("#"):
            parts = line.split("\t")
            if len(parts) >= 8 and parts[0].isdigit():
                current_sent['tokens'].append(parts[1])
                current_sent['lemmas'].append(parts[2])
                current_sent['upos'].append(parts[3])
                current_sent['heads'].append(int(parts[6]))
                current_sent['deprels'].append(parts[7])
                
    out_path = os.path.join(SAMPLE_DIR, "stage1_wernicke_syntax.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(sentences)} Stage 1 samples to {out_path}")

def fetch_stage2_neocortex_knowledge():
    print("Fetching Stage 2: Pure Raw Text Passages...")
    url = "https://s3.amazonaws.com/conceptnet/downloads/2018/omcs-sentences-more.txt"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-150000'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        content = resp.read().decode('utf-8', errors='ignore')
    
    records = []
    lines = content.splitlines()
    for l in lines[1:]:
        parts = l.split('\t')
        if len(parts) >= 7 and parts[4] == 'en':
            text = parts[1].strip()
            if len(text.split()) >= 4:
                records.append({
                    "doc_id": f"passage_{parts[0]}",
                    "text": text if text.endswith('.') else text + '.'
                })
            if len(records) >= 25:
                break
                
    out_path = os.path.join(SAMPLE_DIR, "stage2_neocortex_knowledge.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} Stage 2 pure raw text samples to {out_path}")



def fetch_stage3_parietal_multimodal():
    print("Fetching Stage 1B: MS-COCO Visuolinguistic Grounding Pairs...")
    # Fetch tail of annotations file where annotations objects live
    url = "https://raw.githubusercontent.com/tylin/coco-caption/master/annotations/captions_val2014.json"
    req_head = urllib.request.Request(url, method='HEAD', headers=HEADERS)
    with urllib.request.urlopen(req_head, timeout=10) as resp:
        total_size = int(resp.headers.get('content-length', 29707814))
    
    # Read the last 80KB to cleanly get complete JSON annotation entries
    offset = total_size - 81920
    req_range = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={offset}-{total_size}'})
    with urllib.request.urlopen(req_range, timeout=15) as resp:
        raw_tail = resp.read().decode('utf-8', errors='ignore')
    
    # Parse annotations from substring
    records = []
    import re
    matches = re.findall(r'\{"image_id":\s*(\d+),\s*"id":\s*(\d+),\s*"caption":\s*"([^"]+)"\}', raw_tail)
    
    for img_id, ann_id, caption in matches[:25]:
        img_id_int = int(img_id)
        records.append({
            "image_id": img_id_int,
            "image_filename": f"COCO_val2014_{img_id_int:012d}.jpg",
            "caption_id": int(ann_id),
            "caption": caption.strip()
        })
        
    out_path = os.path.join(SAMPLE_DIR, "stage3_parietal_multimodal.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} pure image-caption grounding pairs to {out_path}")


def fetch_stage4_frontal_reasoning():
    print("Fetching Stage 4: Supervised Learning (SL) Multi-Step Reasoning Dataset (OpenAI GSM8K)...")
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        lines = [resp.readline().decode('utf-8') for _ in range(25)]
    
    import re
    records = []
    for idx, line in enumerate(lines):
        item = json.loads(line)
        # Clean internal calculation brackets for natural reasoning language
        clean_target = re.sub(r'<<.*?>>', '', item['answer']).strip()
        clean_target = clean_target.replace('#### ', 'Therefore, the final answer is ')
        records.append({
            "id": f"reasoning_sl_{idx:03d}",
            "prompt": item["question"].strip(),
            "target": clean_target
        })
        
    out_path = os.path.join(SAMPLE_DIR, "stage4_frontal_reasoning.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} Supervised Learning (SL) reasoning samples to {out_path}")



if __name__ == "__main__":
    fetch_stage1_ud_ewt()
    fetch_stage2_neocortex_knowledge()
    fetch_stage3_parietal_multimodal()
    fetch_stage4_frontal_reasoning()
    print("All fractional dataset samples successfully prepared!")
