#!/usr/bin/env python
"""Convert a Label Studio export into HuggingFace NER-ready train/val JSON.

Usage:
    python scripts/preprocessing/convert_ls_to_ner.py \\
        --input  data/label_studio/data.json \\
        --output data/processed/ \\
        --split  0.8 \\
        --autofix-bio

Steps (see docs/annotation/annotation-pipeline.md #4):
  1. Coordinate normalization: Label Studio % -> [0, 1000]
  2. Reading-order sort (row-bucket by y, then sort by x)
  3. BIO sequence validation (+ optional autofix of dangling I- tags)
  4. Train/val split
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from urllib.parse import urlparse


def normalize_bbox(x_pct: float, y_pct: float, w_pct: float, h_pct: float) -> tuple[int, int, int, int]:
    x0 = int(x_pct / 100 * 1000)
    y0 = int(y_pct / 100 * 1000)
    x1 = int((x_pct + w_pct) / 100 * 1000)
    y1 = int((y_pct + h_pct) / 100 * 1000)
    return x0, y0, x1, y1


def sort_reading_order(tokens: list[dict]) -> list[dict]:
    """Reading-order sort via adaptive line clustering, then left-to-right.

    Tokens are grouped into lines by vertical center with a tolerance derived
    from the median token height (robust to per-token y jitter and to lines a
    fixed-width bin would wrongly merge or split). Within a line, tokens are
    sorted by x; lines are emitted top to bottom.
    """
    if not tokens:
        return tokens

    import statistics

    heights = [t["bbox"][3] - t["bbox"][1] for t in tokens]
    tol = max(1.0, statistics.median(heights) * 0.6)

    lines: list[dict] = []  # each: {"y": running_center, "items": [...]}
    for t in sorted(tokens, key=lambda t: (t["bbox"][1] + t["bbox"][3]) / 2):
        yc = (t["bbox"][1] + t["bbox"][3]) / 2
        if lines and abs(yc - lines[-1]["y"]) <= tol:
            line = lines[-1]
            line["items"].append(t)
            line["y"] = (line["y"] * (len(line["items"]) - 1) + yc) / len(line["items"])
        else:
            lines.append({"y": yc, "items": [t]})

    ordered: list[dict] = []
    for line in lines:
        ordered.extend(sorted(line["items"], key=lambda t: t["bbox"][0]))
    return ordered


def autofix_bio(tags: list[str]) -> list[str]:
    """Convert a dangling I-<TYPE> (no preceding B-<TYPE>/I-<TYPE> of the same type) into B-<TYPE>."""
    fixed = []
    open_type = None
    for tag in tags:
        if tag == "O":
            fixed.append(tag)
            open_type = None
            continue
        prefix, _, etype = tag.partition("-")
        if prefix == "B" or etype == open_type:
            fixed.append(tag)
        else:
            fixed.append(f"B-{etype}")
        open_type = etype
    return fixed


def load_label_studio_export(path: Path) -> list[dict]:
    """Parse a Label Studio JSON export into per-image token/relation records.

    Each token is reconstructed by pairing the `textarea` (transcription)
    and `rectanglelabels` (BIO tag) result entries that share the same
    Label Studio annotation `id`. A small fraction of real exports have
    one half missing (e.g. a leftover OCR box never reviewed) -- these are
    kept with a defaulted label/text rather than dropped, and counted in
    the warning printed at the end.
    """
    with open(path, encoding="utf-8") as f:
        raw_tasks = json.load(f)

    missing_label = 0
    missing_text = 0
    skipped_cancelled = 0
    images = []

    for task in raw_tasks:
        annotations = task.get("annotations") or []
        if not annotations:
            continue
        annotation = annotations[0]
        if annotation.get("was_cancelled"):
            skipped_cancelled += 1
            continue

        by_id: dict[str, dict] = {}
        image_width = image_height = None
        for r in annotation["result"]:
            rtype = r.get("type")
            if rtype in ("textarea", "rectanglelabels"):
                by_id.setdefault(r["id"], {})[rtype] = r
                image_width = image_width or r.get("original_width")
                image_height = image_height or r.get("original_height")

        tokens_raw = []
        for ls_id, parts in by_id.items():
            textarea = parts.get("textarea")
            rect = parts.get("rectanglelabels")
            rect_labels = rect["value"].get("rectanglelabels") if rect else None

            if not rect_labels:
                missing_label += 1
            if textarea is None:
                missing_text += 1

            value = (rect or textarea)["value"]
            text = textarea["value"]["text"][0] if textarea else ""
            label = rect_labels[0] if rect_labels else "O"
            bbox = normalize_bbox(value["x"], value["y"], value["width"], value["height"])
            tokens_raw.append({"ls_id": ls_id, "text": text, "label": label, "bbox": bbox})

        tokens_sorted = sort_reading_order(tokens_raw)
        id_map = {t["ls_id"]: f"token_{i:03d}" for i, t in enumerate(tokens_sorted)}

        relations = []
        for r in annotation["result"]:
            if r.get("type") != "relation":
                continue
            from_id = id_map.get(r["from_id"])
            to_id = id_map.get(r["to_id"])
            if from_id is not None and to_id is not None:
                relations.append({"from_id": from_id, "to_id": to_id, "type": "HAS_VALUE"})

        image_url = task["data"].get("image", "")
        image_name = Path(urlparse(image_url).path).name or image_url

        images.append({
            "id": Path(image_name).stem,
            "image": image_name,
            "image_width": image_width,
            "image_height": image_height,
            "tokens": [
                {"text": t["text"], "bbox": list(t["bbox"]), "label": t["label"]}
                for t in tokens_sorted
            ],
            "relations": relations,
        })

    if missing_label or missing_text or skipped_cancelled:
        print(
            f"[load_label_studio_export] {missing_label} tokens missing a label (defaulted to O), "
            f"{missing_text} tokens missing transcription (defaulted to ''), "
            f"{skipped_cancelled} cancelled annotations skipped -- "
            f"run check_data.py for a per-image breakdown"
        )

    return images


def split_train_val(images: list[dict], split: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * split)
    return shuffled[:cut], shuffled[cut:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Label Studio export JSON")
    parser.add_argument("--output", required=True, help="output directory for train/val JSON")
    parser.add_argument("--split", type=float, default=0.8, help="train split ratio")
    parser.add_argument("--autofix-bio", action="store_true", help="autofix dangling I- tags")
    args = parser.parse_args()

    images = load_label_studio_export(Path(args.input))

    if args.autofix_bio:
        for img in images:
            tags = [t["label"] for t in img["tokens"]]
            fixed = autofix_bio(tags)
            n_fixed = sum(1 for old, new in zip(tags, fixed) if old != new)
            if n_fixed:
                print(f"  autofixed {n_fixed} dangling I- tag(s) in {img['image']}")
            for token, new_tag in zip(img["tokens"], fixed):
                token["label"] = new_tag

    train, val = split_train_val(images, args.split)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(train)} train / {len(val)} val images to {output_dir}")


if __name__ == "__main__":
    main()
