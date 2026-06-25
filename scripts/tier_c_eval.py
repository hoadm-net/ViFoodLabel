#!/usr/bin/env python3
"""
Tier C zero-shot MLLM evaluation for ViFoodLabel — Task 3 (End-to-End KIE).

Two models, identical sampling and identical scoring:
  - GPT-5.4-mini                    (OpenAI SDK)
  - Qwen3-VL-235B-A22B-Instruct     (OpenRouter SDK)

Each model receives the label image only and must return the structured JSON
record defined in docs/benchmark-tasks.md. Predictions are scored against
ground truth with field-level / set-based F1 (see scripts/task3_schema.py).

Run with the project virtualenv:
    .venv/bin/python3 scripts/tier_c_eval.py --model both
    .venv/bin/python3 scripts/tier_c_eval.py --model qwen --n 3
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
from dotenv import load_dotenv
load_dotenv()

import task3_schema as t3

DATASET_DIR = Path("data/processed/dataset")
RAW_IMAGES_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/tier_c_results")
SAMPLE_SIZE = 20
SEED = 42
MAX_RETRIES = 2  # retry when a model returns an empty/unparseable record

MODELS = {
    "gpt": "gpt-5.4-mini",
    "qwen": "qwen/qwen3-vl-235b-a22b-instruct",
}


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

def parse_record(content):
    """Extract the structured record dict from a model's raw text response."""
    if '```json' in content:
        content = content.split('```json')[1].split('```')[0]
    elif '```' in content:
        content = content.split('```')[1].split('```')[0]
    match = re.search(r'\{[\s\S]*\}', content)
    if match:
        content = match.group()
    return t3.coerce_record(json.loads(content))


def is_empty(record):
    return (all(not record[k] for k in t3.SINGLE_KEYS)
            and all(not record[k] for k in t3.LIST_KEYS)
            and not record[t3.NUTRITION_KEY])


def image_to_data_url(image_path):
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


# --------------------------------------------------------------------------- #
# Model callers (return a coerced Task 3 record)
# --------------------------------------------------------------------------- #

def call_gpt(image_path, client):
    response = client.chat.completions.create(
        model=MODELS["gpt"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                {"type": "text", "text": t3.PROMPT},
            ],
        }],
        temperature=0.2,
        max_completion_tokens=4096,
    )
    return parse_record(response.choices[0].message.content)


def call_qwen(image_path, client):
    response = client.chat.send(
        model=MODELS["qwen"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": t3.PROMPT},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ],
        }],
    )
    return parse_record(response.choices[0].message.content)


def call_with_retry(caller, image_path):
    """Call a model, retrying when the record is empty or fails to parse."""
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            rec = caller(image_path)
            if not is_empty(rec):
                return rec, attempt
            last_exc = None
        except Exception as e:
            last_exc = e
    if last_exc:
        raise last_exc
    return t3.empty_record(), MAX_RETRIES  # genuinely empty after retries


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def load_sample(n=SAMPLE_SIZE, ids=None):
    """Return [(image_id, gt_record), ...] from the per-image dataset files.

    Ground truth is read from data/processed/dataset/<id>.json (the human-
    reviewed source of truth); the `kie` block is flattened into the record.
    Pass `ids` to evaluate a specific set, else a seeded random sample of `n`.
    """
    files = sorted(DATASET_DIR.glob("*.json"))
    if not files:
        print(f"❌ no GT files in {DATASET_DIR} — run scripts/build_dataset_meta.py first")
        sys.exit(1)
    data = {}
    for fp in files:
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"❌ malformed JSON in {fp.name}: {e}")
            sys.exit(1)
        data[d["id"]] = {"image": d.get("image", f"{d['id']}.jpeg"), **d["kie"]}
    if ids:
        chosen = [i for i in ids if i in data]
        missing = [i for i in ids if i not in data]
        if missing:
            print(f"⚠️  ids not found, skipped: {missing}")
    else:
        all_ids = sorted(data.keys())
        random.seed(SEED)
        chosen = random.sample(all_ids, min(n, len(all_ids)))
    return [(i, data[i]) for i in chosen]


def run_model(model_key, sample):
    print(f"\n{'='*70}\nTier C / Task 3 — {MODELS[model_key]}\n{'='*70}\n")

    if model_key == "gpt":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not set"); return None
        from openai import OpenAI
        client_ctx, bind = OpenAI(api_key=api_key), call_gpt
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("❌ OPENROUTER_API_KEY not set"); return None
        from openrouter import OpenRouter
        client_ctx, bind = OpenRouter(api_key=api_key), call_qwen

    per_image = []

    def process(caller):
        for idx, (image_id, gt) in enumerate(sample, 1):
            image_path = RAW_IMAGES_DIR / gt.get('image', f"{image_id}.jpeg")
            if not image_path.exists():
                continue
            print(f"[{idx:2d}] {image_id} ", end="", flush=True)
            try:
                pred, retries = call_with_retry(caller, image_path)
                s = t3.score(gt, pred)
                s['image_id'] = image_id
                per_image.append(s)
                tag = "∅" if is_empty(pred) else f"R{retries}" if retries else "  "
                print(f"{tag} | P {s['precision']:5.1%}  R {s['recall']:5.1%}  F1 {s['f1']:.3f}")
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

    # micro field-level (pooled tp/fp/fn over all fields and images)
    tp = sum(d['tp'] for d in per_image)
    fp = sum(d['fp'] for d in per_image)
    fn = sum(d['fn'] for d in per_image)
    micro = t3.prf(tp, fp, fn)

    # macro (per-image average F1)
    macro_f1 = sum(d['f1'] for d in per_image) / n

    # per-field-group breakdown (pooled)
    groups = {}
    for d in per_image:
        for k, (a, b, c) in d['fields'].items():
            g = groups.setdefault(k, [0, 0, 0])
            g[0] += a; g[1] += b; g[2] += c

    summary = {
        'model': MODELS[model_key],
        'task': 'task3_end_to_end_kie',
        'images': n,
        'micro': {**micro, 'tp': tp, 'fp': fp, 'fn': fn},
        'macro_f1': macro_f1,
        'per_field': {k: t3.prf(*v) for k, v in groups.items()},
        'per_image': per_image,
    }

    print(f"\n{'-'*70}")
    print(f"{MODELS[model_key]}  ({n} images)  —  Task 3 field-level F1")
    print(f"  Micro:    P {micro['precision']:5.1%}  R {micro['recall']:5.1%}  F1 {micro['f1']:.3f}")
    print(f"  Macro F1: {macro_f1:.3f}")
    print(f"  Per field:")
    for k in t3.ALL_KEYS:
        if k in groups:
            m = t3.prf(*groups[k])
            print(f"    {k:16s} P {m['precision']:5.1%}  R {m['recall']:5.1%}  F1 {m['f1']:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"tier_c_{model_key}.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  → saved {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Tier C / Task 3 zero-shot MLLM evaluation")
    ap.add_argument("--model", choices=["gpt", "qwen", "both"], default="both")
    ap.add_argument("--n", type=int, default=SAMPLE_SIZE, help="number of images to sample")
    ap.add_argument("--ids", default="", help="comma-separated image ids to evaluate (overrides --n)")
    args = ap.parse_args()

    ids = [s.strip() for s in args.ids.split(",") if s.strip()] or None
    sample = load_sample(args.n, ids)
    src = f"{len(sample)} specified images" if ids else f"{len(sample)} images (seed={SEED})"
    print(f"📂 Loaded GT for {src}")

    keys = ["gpt", "qwen"] if args.model == "both" else [args.model]
    results = {k: run_model(k, sample) for k in keys}

    done = {k: v for k, v in results.items() if v}
    if len(done) > 1:
        print(f"\n{'='*70}\nCOMPARISON (same {len(sample)} images, Task 3 micro F1)\n{'='*70}")
        print(f"{'Model':<38}{'P':>8}{'R':>8}{'F1':>8}")
        for k in keys:
            if results[k]:
                m = results[k]['micro']
                print(f"{MODELS[k]:<38}{m['precision']:>7.1%}{m['recall']:>7.1%}{m['f1']:>8.3f}")


if __name__ == "__main__":
    main()
