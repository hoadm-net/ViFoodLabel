#!/usr/bin/env python3
"""
LLM-assisted QC linter for Task 3 ground-truth records.

Reads the assembled GT (data/processed/task3_gt.json) and asks a *neutral*
text-only model (Gemini via OpenRouter — deliberately NOT GPT-5.4-mini or
Qwen3-VL, the two models under evaluation) to flag records whose fields look
fragmented, mis-ordered, column-bled, or internally inconsistent.

The model sees ONLY the assembled text — never the image — so it cannot invent
ground truth; it only triages which images a human should re-open and fix
directly in task3_gt.json. Output is a worklist sorted by severity.

Usage:
    .venv/bin/python3 scripts/task3_qc_lint.py            # all records
    .venv/bin/python3 scripts/task3_qc_lint.py --n 5      # first 5 (smoke test)
"""

import argparse
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

GT_FILE = Path("data/processed/task3_gt.json")
OUT_FILE = Path("data/processed/task3_qc_flags.json")
LINTER_MODEL = "google/gemini-2.5-flash"  # neutral vs. the evaluated models
MAX_RETRIES = 2

SYSTEM = """You are a strict, conservative data-quality linter for a Vietnamese
food-label extraction dataset. A record was auto-assembled from word-level OCR
annotations sorted by reading order; assembly can break on multi-column layouts,
bilingual labels, and line wrapping.

You see ONLY the assembled text, never the image. Do NOT invent the correct
content. Flag a field ONLY when the text is CLEARLY BROKEN, i.e. you are
confident a human would have to fix it:
- a phrase split mid-word or mid-term across two list items
  (e.g. "NƯỚC ÉP NHO CÔ" + "ĐẶC" — "cô đặc" was cut apart)
- obviously duplicated words (e.g. "SỮA SỮA TH")
- a fragment that starts with a stray closing bracket/comma or is a dangling
  clause (e.g. "627), LIỆU NHIÊN,", or a warning split into "CÓ" / "TRẺ EM CẦN")
- two languages clearly interleaved out of order within one item
- a nutrition name that absorbed an adjacent name (e.g. "CARBOHYDRAT TỔNG SỐ ĐẠM")

Do NOT flag (these are acceptable, NOT errors):
- batch/lot codes or unusual date formats (e.g. "31.12.25A1 13:16")
- %DV percentages inside a nutrition value (e.g. "20 g 7%")
- bilingual names joined with "/" (e.g. "Năng lượng / Energy")
- a field being empty, or trailing punctuation
- stylistic/spelling concerns that do not break the phrase

Be conservative: if a field reads as a coherent phrase, do NOT flag it.
Severity: "high" = a core/multiple fields clearly broken; "medium" = one list
field with clear fragments; "low" = a single minor break.

Return ONLY JSON:
{"flag": true|false, "severity": "high"|"medium"|"low",
 "issues": [{"field": "<field>", "problem": "<short reason>"}]}
If everything reads coherently, return {"flag": false, "severity": "low", "issues": []}.
"""


def record_for_prompt(rec):
    """Strip non-field keys (e.g. 'image') for the text-only prompt."""
    return {k: rec[k] for k in t3.ALL_KEYS if k in rec}


def parse_json(content):
    if '```json' in content:
        content = content.split('```json')[1].split('```')[0]
    elif '```' in content:
        content = content.split('```')[1].split('```')[0]
    m = re.search(r'\{[\s\S]*\}', content)
    if m:
        content = m.group()
    return json.loads(content)


def lint_record(client, image_id, rec):
    payload = json.dumps(record_for_prompt(rec), ensure_ascii=False, indent=1)
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.send(
                model=LINTER_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Record id {image_id}:\n{payload}"},
                ],
            )
            return parse_json(resp.choices[0].message.content)
        except Exception as e:
            if attempt == MAX_RETRIES:
                return {"flag": True, "severity": "low",
                        "issues": [{"field": "_error", "problem": f"lint failed: {str(e)[:60]}"}]}
            time.sleep(1.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=0, help="limit to first N records (0 = all)")
    args = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set"); sys.exit(1)
    if not GT_FILE.exists():
        print(f"❌ {GT_FILE} not found — run scripts/build_task3_gt.py first"); sys.exit(1)

    gt = json.load(open(GT_FILE, encoding="utf-8"))
    ids = sorted(gt.keys())
    if args.n:
        ids = ids[:args.n]

    print(f"🔎 Linting {len(ids)} records with {LINTER_MODEL}\n")

    from openrouter import OpenRouter
    rank = {"high": 0, "medium": 1, "low": 2}
    flagged = []

    with OpenRouter(api_key=api_key) as client:
        for i, image_id in enumerate(ids, 1):
            res = lint_record(client, image_id, gt[image_id])
            mark = "  "
            if res.get("flag"):
                sev = res.get("severity", "low")
                mark = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(sev, "⚪")
                flagged.append({
                    "id": image_id,
                    "image": gt[image_id].get("image", f"{image_id}.jpeg"),
                    "severity": sev,
                    "issues": res.get("issues", []),
                })
            fields = ", ".join(i["field"] for i in res.get("issues", []))
            print(f"[{i:3d}/{len(ids)}] {image_id} {mark} {fields}")

    flagged.sort(key=lambda x: rank.get(x["severity"], 3))
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "linter_model": LINTER_MODEL,
            "total_records": len(ids),
            "flagged": len(flagged),
            "by_severity": {s: sum(1 for x in flagged if x["severity"] == s)
                            for s in ("high", "medium", "low")},
            "records": flagged,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(flagged)}/{len(ids)} flagged for human review → {OUT_FILE}")
    print("  Edit the flagged ids directly in data/processed/task3_gt.json, then re-run Tier C.")


if __name__ == "__main__":
    main()
