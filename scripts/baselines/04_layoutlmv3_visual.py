#!/usr/bin/env python
"""LayoutLMv3 (microsoft/layoutlmv3-base) full multimodal: text + layout + image.

Usage:
    python scripts/baselines/04_layoutlmv3_visual.py \\
        --train  data/processed/train.json \\
        --val    data/processed/val.json \\
        --images data/raw/ \\
        --epochs 30

Same flow as 03_layoutlmv3_no_visual.py, plus the actual cropped/resized
image patches via `LayoutLMv3Processor` (visual segment enabled). The
processor's image side always returns a batch dimension even for one
image, so `pixel_values` needs an explicit `[0]` unwrap that the text
side (input_ids/bbox/labels) doesn't. `pixel_values` is also a fixed
(3, 224, 224) tensor per example, not a variable-length sequence, so it
needs a dedicated collator that stacks it directly instead of going
through the tokenizer's sequence-padding path (which expects every field
to be paddable along the sequence dimension).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.metrics import entity_f1, token_f1  # noqa: E402

MODEL_NAME = "microsoft/layoutlmv3-base"
MAX_LENGTH = 512

ENTITY_TYPES = [
    "PRODUCT_NAME", "INGREDIENT", "ADDITIVE", "NUTRITION_NAME", "NUTRITION_VALUE",
    "MANUFACTURER", "ORIGIN", "NET_WEIGHT", "MFG_DATE", "EXPIRY_DATE", "WARNING",
]
LABELS = ["O"] + [f"{prefix}-{etype}" for etype in ENTITY_TYPES for prefix in ("B", "I")]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


def load_images(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_examples(images: list[dict], processor, images_dir: str) -> list[dict]:
    examples = []
    for img in images:
        words = [t["text"] for t in img["tokens"]]
        boxes = [t["bbox"] for t in img["tokens"]]
        word_label_ids = [LABEL2ID.get(t["label"], LABEL2ID["O"]) for t in img["tokens"]]
        pil_image = Image.open(Path(images_dir) / img["image"]).convert("RGB")
        encoding = processor(
            pil_image, text=words, boxes=boxes, word_labels=word_label_ids,
            truncation=True, max_length=MAX_LENGTH,
        )
        examples.append({
            "id": img["id"],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": encoding["bbox"],
            "labels": encoding["labels"],
            "pixel_values": encoding["pixel_values"][0],
        })
    return examples


class NERDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "bbox": ex["bbox"],
            "labels": ex["labels"],
            "pixel_values": ex["pixel_values"],
        }


class VisualDataCollator:
    """Pads the text-side sequence fields normally, then stacks the fixed-size
    `pixel_values` tensor separately (it isn't a sequence, so it can't go
    through the tokenizer's sequence-padding path)."""

    def __init__(self, tokenizer):
        self.base_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

    def __call__(self, features: list[dict]) -> dict:
        pixel_values = torch.stack([torch.as_tensor(f["pixel_values"]) for f in features])
        text_features = [{k: v for k, v in f.items() if k != "pixel_values"} for f in features]
        batch = self.base_collator(text_features)
        batch["pixel_values"] = pixel_values
        return batch


def compute_metrics(eval_pred) -> dict:
    """Lightweight live per-epoch signal (flat token accuracy over real, non-padded labels)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    correct = total = 0
    for pred_seq, label_seq in zip(preds, labels):
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            total += 1
            correct += int(p == l)
    return {"token_accuracy": correct / total if total else 0.0}


def predictions_to_images(logits: np.ndarray, labels: np.ndarray, ids: list[str], gold_images: list[dict]) -> list[dict]:
    """Re-attach predicted word-level labels onto a copy of each image's gold tokens
    (the benchmark metrics in src/metrics.py expect the same per-image record shape)."""
    pred_ids = np.argmax(logits, axis=-1)
    gold_by_id = {g["id"]: g for g in gold_images}
    pred_images = []
    for img_id, label_seq, pred_seq in zip(ids, labels, pred_ids):
        gold = gold_by_id[img_id]
        word_preds = [ID2LABEL[p] for p, l in zip(pred_seq, label_seq) if l != -100]
        tokens = []
        for i, t in enumerate(gold["tokens"]):
            label = word_preds[i] if i < len(word_preds) else "O"  # truncated-away words default to O
            tokens.append({**t, "label": label})
        pred_images.append({"id": img_id, "tokens": tokens, "relations": gold.get("relations", [])})
    return pred_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--images", required=True, help="dir with source images for cropped visual features")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="checkpoints/layoutlmv3_visual")
    args = parser.parse_args()

    train_images = load_images(args.train)
    val_images = load_images(args.val)

    processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_examples = build_examples(train_images, processor, args.images)
    val_examples = build_examples(val_images, processor, args.images)
    train_dataset = NERDataset(train_examples)
    val_dataset = NERDataset(val_examples)
    val_ids = [ex["id"] for ex in val_examples]

    data_collator = VisualDataCollator(processor.tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predict_output = trainer.predict(val_dataset)
    pred_images = predictions_to_images(predict_output.predictions, predict_output.label_ids, val_ids, val_images)

    report = {**token_f1(pred_images, val_images), **entity_f1(pred_images, val_images)}
    print(json.dumps(
        {k: v for k, v in report.items() if not k.endswith("_per_label")},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
