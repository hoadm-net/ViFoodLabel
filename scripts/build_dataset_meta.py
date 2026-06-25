#!/usr/bin/env python3
"""
Build the per-image dataset record: KIE ground truth + dataset metadata.

For every annotated image, emit one JSON file combining:
  - auto-derived stats   (token_count, entity_counts per label, num_relations)
  - VLM-proposed meta    (product_category, background_color, material)
  - the Task 3 KIE record

The VLM (Gemini via OpenRouter) only PROPOSES the three visual attributes from a
closed vocabulary; an annotator fixes them only when clearly wrong. These three
attributes are used for dataset description and stratified train/dev/test
splitting — they are NOT KIE evaluation targets, so the proposing model does not
contaminate Tier C scoring.

Usage:
    .venv/bin/python3 scripts/build_dataset_meta.py --n 3      # smoke test
    .venv/bin/python3 scripts/build_dataset_meta.py           # all images
    .venv/bin/python3 scripts/build_dataset_meta.py --overwrite
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

import task3_schema as t3

PROCESSED_DIR = Path("data/processed")
GT_FILE = PROCESSED_DIR / "task3_gt.json"
RAW_IMAGES_DIR = Path("data/raw")
OUT_DIR = PROCESSED_DIR / "dataset"
INDEX_FILE = PROCESSED_DIR / "dataset_meta.json"
VLM_MODEL = "google/gemini-2.5-flash"
MAX_RETRIES = 2

VOCAB = {
    "product_category": ["cake", "candy", "snack", "beverage", "seasoning", "dried_food", "other"],
    "background_color": ["red", "yellow", "green", "blue", "white", "other"],
    "material": ["paper", "plastic", "foil", "can_bottle", "other"],
}
FALLBACK = {"product_category": "other", "background_color": "other", "material": "other"}

PROMPT = f"""Classify this Vietnamese food product label image into a fixed taxonomy.
Choose EXACTLY ONE value per field from the allowed lists. If unsure, use "other".

product_category: {VOCAB['product_category']}
  cake=cakes/cookies/biscuits, candy=candy/jelly, snack=savory snacks,
  beverage=drinks, seasoning=sauces/seasonings, dried_food=instant noodles/dried
  foods/nuts, other=anything else
background_color (dominant packaging background): {VOCAB['background_color']}
  red=red/pink/warm, yellow=yellow/orange, green=green, blue=blue,
  white=white/light, other=dark/black/brown or multicolor with no dominant
material (packaging): {VOCAB['material']}
  paper=paper/cardboard box, plastic=plastic film/bag, foil=metallized foil,
  can_bottle=rigid can/bottle, other=anything else

Return ONLY JSON: {{"product_category": "...", "background_color": "...", "material": "..."}}
"""


def load_processed():
    imgs = {}
    for name in ("train.json", "val.json", "test.json"):
        p = PROCESSED_DIR / name
        if p.exists():
            for img in json.load(open(p, encoding="utf-8")):
                imgs[img["id"]] = img
    return imgs


def auto_meta(img):
    ents = t3.merge_entities(img["tokens"])
    counts = {}
    for e in ents:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    return {
        "token_count": len(img["tokens"]),
        "entity_counts": dict(sorted(counts.items())),
        "num_relations": len(img.get("relations", [])),
        "image_width": img.get("image_width"),
        "image_height": img.get("image_height"),
    }


def parse_json(content):
    if '```json' in content:
        content = content.split('```json')[1].split('```')[0]
    elif '```' in content:
        content = content.split('```')[1].split('```')[0]
    m = re.search(r'\{[\s\S]*\}', content)
    if m:
        content = m.group()
    return json.loads(content)


def vlm_meta(client, image_path):
    with open(image_path, "rb") as f:
        data_url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.send(
                model=VLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
            )
            raw = parse_json(resp.choices[0].message.content)
            # validate against vocab
            out = {}
            for field, allowed in VOCAB.items():
                val = str(raw.get(field, "")).strip().lower()
                out[field] = val if val in allowed else FALLBACK[field]
            return out
        except Exception:
            if attempt == MAX_RETRIES:
                return dict(FALLBACK)
            time.sleep(1.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=0, help="limit to first N images (0 = all)")
    ap.add_argument("--overwrite", action="store_true", help="re-generate existing files")
    args = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set"); sys.exit(1)
    if not GT_FILE.exists():
        print(f"❌ {GT_FILE} not found — run scripts/build_task3_gt.py first"); sys.exit(1)

    imgs = load_processed()
    gt = json.load(open(GT_FILE, encoding="utf-8"))
    ids = sorted(gt.keys())
    if args.n:
        ids = ids[:args.n]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🏷️  Building dataset metadata for {len(ids)} images with {VLM_MODEL}\n")

    from openrouter import OpenRouter
    index = {}

    with OpenRouter(api_key=api_key) as client:
        for i, image_id in enumerate(ids, 1):
            out_path = OUT_DIR / f"{image_id}.json"
            if out_path.exists() and not args.overwrite:
                index[image_id] = json.load(open(out_path, encoding="utf-8"))["meta"]
                print(f"[{i:3d}/{len(ids)}] {image_id} (cached)")
                continue

            image_file = gt[image_id].get("image", f"{image_id}.jpeg")
            image_path = RAW_IMAGES_DIR / image_file
            meta = auto_meta(imgs[image_id]) if image_id in imgs else {}
            if image_path.exists():
                meta.update(vlm_meta(client, image_path))
            else:
                meta.update(FALLBACK)

            record = {
                "id": image_id,
                "image": image_file,
                "meta": meta,
                "kie": {k: gt[image_id][k] for k in t3.ALL_KEYS if k in gt[image_id]},
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            index[image_id] = meta
            print(f"[{i:3d}/{len(ids)}] {image_id} → {meta.get('product_category')}, "
                  f"{meta.get('background_color')}, {meta.get('material')}  "
                  f"({meta.get('token_count')} tok)")

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Wrote {len(index)} records → {OUT_DIR}/  (index: {INDEX_FILE})")

    # distribution snapshot
    for field in VOCAB:
        dist = {}
        for m in index.values():
            dist[m.get(field)] = dist.get(m.get(field), 0) + 1
        print(f"  {field}: {dict(sorted(dist.items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()
