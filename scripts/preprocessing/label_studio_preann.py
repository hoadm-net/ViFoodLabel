#!/usr/bin/env python
"""Label Studio pre-annotation pipeline: doctr detection + VietOCR recognition.

Generates a tasks.json file for Label Studio with OCR pre-annotations:
  - Step 1: doctr detects word-level bounding boxes from images
  - Step 2: VietOCR recognizes text in each bbox crop (Vietnamese-optimized)
  - Step 3: Split multi-word bboxes by character-count ratio
  - Step 4: Export as Label Studio predictions format

Usage:
    python label_studio_preann.py \\
        --folder data/raw \\
        --start 1 \\
        --end 10 \\
        --output data/label_studio/tasks.json

The script writes checkpoints after every image, so it can be resumed if interrupted.

Requirements:
    pip install python-doctr vietocr torch torchvision
"""

from __future__ import annotations

import json
import os
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Any

# Cap thread pools before heavy imports to avoid resource exhaustion on multi-core systems
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

cv2.setNumThreads(4)
torch.set_num_threads(4)

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.floating):
            return round(float(obj), 4)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_models() -> tuple:
    """Load doctr (detection) and VietOCR (recognition) models."""
    print("Loading doctr model (text detection)...")
    doctr_model = ocr_predictor(pretrained=True, assume_straight_pages=False)
    if torch.cuda.is_available():
        doctr_model = doctr_model.cuda()

    print("Loading VietOCR model (text recognition)...")
    cfg = Cfg.load_config_from_name("vgg_transformer")
    cfg["cnn"]["pretrained"] = False
    cfg["predictor"]["beamsearch"] = False
    vietocr_model = Predictor(cfg)

    print("Models loaded.\n")
    return doctr_model, vietocr_model


def split_multiword_bbox(bbox_entry: dict) -> list[dict]:
    """Split a multi-word bbox by character-count ratio into per-word bboxes.

    If bbox contains multiple words (space-separated), proportionally divide
    the bbox width based on each word's character count.
    """
    words = bbox_entry["text"].split()
    if len(words) <= 1:
        return [bbox_entry]

    x, y, w, h = bbox_entry["x"], bbox_entry["y"], bbox_entry["width"], bbox_entry["height"]
    total_chars = sum(len(word) for word in words)
    if total_chars == 0:
        return [bbox_entry]

    splits = []
    cursor = x
    for word in words:
        ratio = len(word) / total_chars
        word_width = w * ratio
        splits.append({
            "x": cursor,
            "y": y,
            "width": word_width,
            "height": h,
            "text": word,
            "confidence": bbox_entry["confidence"],
        })
        cursor += word_width

    # Snap last bbox right edge to original boundary (avoid float drift)
    splits[-1]["width"] = (x + w) - splits[-1]["x"]
    return splits


def process_image(
    image_path: str,
    doctr_model: Any,
    vietocr_model: Any,
    image_base_url: str,
) -> dict:
    """Process one image: detect bboxes, recognize text, format for Label Studio."""
    filename = Path(image_path).name
    image_url = f"{image_base_url.rstrip('/')}/{filename}"
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError(f"Could not read image: {image_path}")

    H_img, W_img = img_cv.shape[:2]

    # Text detection
    doc = DocumentFile.from_images(image_path)
    result = doctr_model(doc)

    ls_results = []

    for page in result.pages:
        H_page, W_page = page.dimensions
        sx, sy = W_img / W_page, H_img / H_page

        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    pts = np.array(word.geometry, dtype=np.float32)
                    xs, ys = pts[:, 0], pts[:, 1]

                    # Pixel coordinates
                    x1, y1 = int(xs.min() * W_page * sx), int(ys.min() * H_page * sy)
                    x2, y2 = int(xs.max() * W_page * sx), int(ys.max() * H_page * sy)

                    if (x2 - x1) < 8 or (y2 - y1) < 8:
                        continue

                    # Text recognition on cropped region
                    crop = img_cv[y1:y2, x1:x2]
                    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    try:
                        text, conf = vietocr_model.predict(pil_crop, return_prob=True)
                        conf = float(conf)
                    except Exception:
                        text, conf = word.value, 0.0

                    # Label Studio percentage coordinates
                    bbox = {
                        "x": float(xs.min() * 100),
                        "y": float(ys.min() * 100),
                        "width": float((xs.max() - xs.min()) * 100),
                        "height": float((ys.max() - ys.min()) * 100),
                        "text": text,
                        "confidence": round(conf, 4),
                    }

                    # Split multi-word bboxes
                    for part in split_multiword_bbox(bbox):
                        ls_results.append({
                            "value": {
                                "x": part["x"],
                                "y": part["y"],
                                "width": part["width"],
                                "height": part["height"],
                                "rotation": 0,
                                "text": [part["text"]],
                                "rectanglelabels": ["O"],
                            },
                            "from_name": "transcription",
                            "to_name": "image",
                            "type": "textarea",
                        })

    print(f"  → {len(ls_results)} words detected and recognized")

    return {
        "data": {"image": image_url},
        "predictions": [{"result": ls_results}],
    }


def build_image_list(folder: str, start: int, end: int | None, ext: str = "jpeg") -> list[str]:
    """Build list of image paths from numbered range."""
    if end is None:
        # Auto-detect max index
        max_idx = 0
        for fname in os.listdir(folder):
            stem, _, fext = fname.rpartition(".")
            if fext == ext and stem.isdigit():
                max_idx = max(max_idx, int(stem))
        end = max_idx
        if end == 0:
            print(f"Warning: No images found in {folder}")
            return []
        print(f"Auto-detected highest image index: {end}")

    paths = []
    for i in range(start, end + 1):
        name = f"{i:04d}.{ext}"
        p = os.path.join(folder, name)
        if os.path.exists(p):
            paths.append(p)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-annotate images with OCR for Label Studio.",
    )
    parser.add_argument("--folder", default="data/raw", help="Image directory")
    parser.add_argument("--start", type=int, default=1, help="Start image index")
    parser.add_argument("--end", type=int, default=None, help="End image index (auto-detects if omitted)")
    parser.add_argument("--ext", default="jpeg", help="Image extension")
    parser.add_argument("--output", default="tasks.json", help="Output tasks.json path")
    parser.add_argument(
        "--image-base-url",
        default="http://103.159.52.8/images",
        help="Base URL the images are served from (Label Studio reads 'data.image' as a URL, not a local path)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Label Studio Pre-annotation Pipeline")
    print("=" * 60)

    # Load models once
    doctr_model, vietocr_model = load_models()

    # Build image list
    image_list = build_image_list(args.folder, args.start, args.end, args.ext)
    if not image_list:
        print("No images found.")
        return

    print(f"Processing {len(image_list)} images...\n")

    tasks = []
    for i, img_path in enumerate(image_list, 1):
        print(f"[{i}/{len(image_list)}] {os.path.basename(img_path)}")
        try:
            task = process_image(img_path, doctr_model, vietocr_model, args.image_base_url)
            tasks.append(task)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        # Checkpoint after every image
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 60)
    print(f"✓ Saved {args.output} ({len(tasks)} images)")
    print("=" * 60)


if __name__ == "__main__":
    main()
