#!/usr/bin/env python3
"""
Tier C zero-shot MLLM evaluation for ViFoodLabel KIE.

Two models, identical sampling and identical evaluation:
  - GPT-5.4-mini                    (OpenAI API)
  - Qwen3-VL-235B-A22B-Instruct     (OpenRouter API)

Run with the project virtualenv:
    .venv/bin/python3 scripts/tier_c_eval.py --model both

Predictions (phrase-level) are matched against de-tokenized BIO ground truth
using token-overlap similarity with a same-label requirement and a 0.5
threshold. Reports recall / precision / F1.
"""

import argparse
import base64
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from detokenize_bio import detokenize_bio_entities, normalize_text

from dotenv import load_dotenv
load_dotenv()

DATA_JSONL = Path("data/label_studio/data.json")
RAW_IMAGES_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/tier_c_results")
SAMPLE_SIZE = 20
SEED = 42
MATCH_THRESHOLD = 0.5

ENTITY_TYPES = [
    "PRODUCT_NAME", "INGREDIENT", "ADDITIVE", "NUTRITION_NAME",
    "NUTRITION_VALUE", "NET_WEIGHT", "MANUFACTURER", "ORIGIN",
    "EXPIRY_DATE", "MFG_DATE", "WARNING",
]

PROMPT = f"""Extract entities from Vietnamese food label.

Entity types: {", ".join(ENTITY_TYPES)}

Return ONLY JSON:
{{"entities": [{{"text": "...", "label": "TYPE"}}, ...]}}
"""

MODELS = {
    "gpt": "gpt-5.4-mini",
    "qwen": "qwen/qwen3-vl-235b-a22b-instruct",
}


# --------------------------------------------------------------------------- #
# Ground truth + evaluation (shared by both models)
# --------------------------------------------------------------------------- #

def extract_ground_truth(item):
    """Pull BIO token-level entities from a Label Studio item.

    Label Studio stores transcribed text (textarea) and BIO label
    (rectanglelabels) as separate results linked only by bounding box, so
    they are matched on rounded bbox coordinates.
    """
    annotations = item.get('annotations', [])
    if not annotations:
        return []
    result_items = annotations[0].get('result', [])
    textareas = []
    labels_by_bbox = {}

    for res in result_items:
        value = res.get('value', {})
        bbox_key = (round(value.get('x', 0), 1), round(value.get('y', 0), 1),
                    round(value.get('width', 0), 1), round(value.get('height', 0), 1))

        if res.get('type') == 'textarea':
            text = value.get('text')
            if isinstance(text, list):
                text = ' '.join(text) if text else None
            if text and text.strip():
                textareas.append({'text': text, 'bbox_key': bbox_key})

        elif res.get('type') == 'rectanglelabels':
            labels = value.get('rectanglelabels', [])
            if labels and labels[0] != 'O':
                labels_by_bbox[bbox_key] = labels[0]

    entities = []
    for ta in textareas:
        label = labels_by_bbox.get(ta['bbox_key'])
        if label:
            entities.append({'text': ta['text'], 'label': label})

    return entities


def token_overlap_similarity(text1, text2):
    tokens1 = set(re.findall(r'\w+', normalize_text(text1).lower()))
    tokens2 = set(re.findall(r'\w+', normalize_text(text2).lower()))
    if not tokens1 or not tokens2:
        return 1.0 if tokens1 == tokens2 else 0.0
    overlap = len(tokens1 & tokens2)
    return overlap / max(len(tokens1), len(tokens2))


def evaluate(gt_bio_entities, pred_entities):
    """Greedy bipartite matching by same-label token overlap >= threshold."""
    gt_phrases = detokenize_bio_entities(gt_bio_entities)
    matched_gt = set()

    for pred in pred_entities:
        best_overlap = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_phrases):
            if gt_idx in matched_gt or gt['label'] != pred.get('label'):
                continue
            overlap = token_overlap_similarity(gt['text'], pred.get('text', ''))
            if overlap > best_overlap:
                best_overlap = overlap
                best_gt_idx = gt_idx
        if best_overlap >= MATCH_THRESHOLD:
            matched_gt.add(best_gt_idx)

    tp = len(matched_gt)
    n_gt = len(gt_phrases)
    n_pred = len(pred_entities)
    recall = tp / n_gt if n_gt else 0.0
    precision = tp / n_pred if n_pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'gt_count': n_gt, 'pred_count': n_pred, 'tp': tp,
        'recall': recall, 'precision': precision, 'f1': f1,
    }


def parse_entities(content):
    """Extract the entities list from a model's raw text response."""
    if '```json' in content:
        content = content.split('```json')[1].split('```')[0]
    elif '```' in content:
        content = content.split('```')[1].split('```')[0]
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        content = json_match.group()
    parsed = json.loads(content)
    return parsed.get('entities', [])


def image_to_data_url(image_path):
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


# --------------------------------------------------------------------------- #
# Model callers
# --------------------------------------------------------------------------- #

def call_gpt(image_path, client):
    """GPT-5.4-mini via OpenAI SDK."""
    response = client.chat.completions.create(
        model=MODELS["gpt"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        temperature=0.2,
        max_completion_tokens=4096,
    )
    return parse_entities(response.choices[0].message.content)


def call_qwen(image_path, client):
    """Qwen3-VL-235B-A22B-Instruct via OpenRouter SDK."""
    response = client.chat.send(
        model=MODELS["qwen"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ],
        }],
    )
    return parse_entities(response.choices[0].message.content)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def load_sample(n=SAMPLE_SIZE):
    with open(DATA_JSONL) as f:
        data = json.load(f)
    items = [item for item in data
             if item.get('annotations') and item['annotations'][0].get('result')]
    random.seed(SEED)
    return random.sample(items, min(n, len(items)))


def run_model(model_key, sample):
    print(f"\n{'='*70}\nTier C — {MODELS[model_key]}\n{'='*70}\n")

    if model_key == "gpt":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not set"); return None
        from openai import OpenAI
        client_ctx = OpenAI(api_key=api_key)
        bind = call_gpt
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("❌ OPENROUTER_API_KEY not set"); return None
        from openrouter import OpenRouter
        client_ctx = OpenRouter(api_key=api_key)
        bind = call_qwen

    per_image = []

    def process(client_caller):
        for idx, item in enumerate(sample, 1):
            image_id = Path(item['data']['image']).stem
            image_path = RAW_IMAGES_DIR / f"{image_id}.jpeg"
            if not image_path.exists():
                continue
            gt_bio = extract_ground_truth(item)
            n_gt = len(detokenize_bio_entities(gt_bio))
            print(f"[{idx:2d}] {image_id} ({n_gt:3d} GT) ", end="", flush=True)
            try:
                preds = client_caller(image_path)
                m = evaluate(gt_bio, preds)
                m['image_id'] = image_id
                per_image.append(m)
                print(f"✓ {m['pred_count']:3d} pred | "
                      f"R {m['recall']:5.1%}  P {m['precision']:5.1%}  F1 {m['f1']:.3f}")
            except Exception as e:
                print(f"❌ {str(e)[:50]}")

    with client_ctx as client:
        process(lambda p: bind(p, client))

    return summarize(model_key, per_image)


def summarize(model_key, per_image):
    if not per_image:
        print("\nNo successful extractions")
        return None
    n = len(per_image)
    avg = lambda k: sum(d[k] for d in per_image) / n
    # micro-averaged (pooled counts)
    tp = sum(d['tp'] for d in per_image)
    n_gt = sum(d['gt_count'] for d in per_image)
    n_pred = sum(d['pred_count'] for d in per_image)
    micro_r = tp / n_gt if n_gt else 0.0
    micro_p = tp / n_pred if n_pred else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    summary = {
        'model': MODELS[model_key],
        'images': n,
        'macro': {'recall': avg('recall'), 'precision': avg('precision'), 'f1': avg('f1')},
        'micro': {'recall': micro_r, 'precision': micro_p, 'f1': micro_f1},
        'per_image': per_image,
    }

    print(f"\n{'-'*70}")
    print(f"{MODELS[model_key]}  ({n} images)")
    print(f"  Macro:  R {avg('recall'):5.1%}  P {avg('precision'):5.1%}  F1 {avg('f1'):.3f}")
    print(f"  Micro:  R {micro_r:5.1%}  P {micro_p:5.1%}  F1 {micro_f1:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"tier_c_{model_key}.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  → saved {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Tier C zero-shot MLLM evaluation")
    ap.add_argument("--model", choices=["gpt", "qwen", "both"], default="both")
    ap.add_argument("--n", type=int, default=SAMPLE_SIZE, help="number of images to sample")
    args = ap.parse_args()

    sample = load_sample(args.n)
    print(f"📂 Loaded sample of {len(sample)} annotated images (seed={SEED})")

    keys = ["gpt", "qwen"] if args.model == "both" else [args.model]
    results = {k: run_model(k, sample) for k in keys}

    done = {k: v for k, v in results.items() if v}
    if len(done) > 1:
        print(f"\n{'='*70}\nCOMPARISON (same {len(sample)} images)\n{'='*70}")
        print(f"{'Model':<38}{'R':>8}{'P':>8}{'F1':>8}")
        for k in keys:
            if results[k]:
                m = results[k]['macro']
                print(f"{MODELS[k]:<38}{m['recall']:>7.1%}{m['precision']:>7.1%}{m['f1']:>8.3f}")


if __name__ == "__main__":
    main()
