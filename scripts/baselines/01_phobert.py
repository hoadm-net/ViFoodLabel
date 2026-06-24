#!/usr/bin/env python
"""PhoBERT (vinai/phobert-base) text-only token classification baseline.

Usage:
    python scripts/baselines/01_phobert.py \\
        --train data/processed/train.json \\
        --val   data/processed/val.json \\
        --epochs 20

Note (docs/notes/technical-runtime-notes.md #1-2): PhoBERT's effective max
position embedding is ~258 -> cap content at max_length=256 with truncation;
PhoBERT's tokenizer needs explicit word-to-subword label alignment, with
continuation subwords set to the ignored label index (-100). The slow
PhobertTokenizer also has no `word_ids()` (fast-tokenizer-only API), so
each word is BPE-encoded individually to find its subword span.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.metrics import entity_f1, token_f1  # noqa: E402

MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 256

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


def tokenize_and_align(
    words: list[str], word_label_ids: list[int], tokenizer, max_length: int
) -> tuple[list[int], list[int], list[int]]:
    """BPE-encode each word individually and assign its label to the first
    subword, -100 to continuation subwords and special tokens."""
    input_ids = [tokenizer.bos_token_id]
    labels = [-100]
    for word, label_id in zip(words, word_label_ids):
        piece_ids = tokenizer.encode(word, add_special_tokens=False)
        if not piece_ids:
            continue
        if len(input_ids) + len(piece_ids) + 1 > max_length:  # +1 reserves room for eos
            break
        input_ids.extend(piece_ids)
        labels.append(label_id)
        labels.extend([-100] * (len(piece_ids) - 1))
    input_ids.append(tokenizer.eos_token_id)
    labels.append(-100)
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, labels


def build_examples(images: list[dict], tokenizer) -> list[dict]:
    examples = []
    for img in images:
        words = [t["text"] for t in img["tokens"]]
        word_label_ids = [LABEL2ID.get(t["label"], LABEL2ID["O"]) for t in img["tokens"]]
        input_ids, attention_mask, labels = tokenize_and_align(words, word_label_ids, tokenizer, MAX_LENGTH)
        examples.append({"id": img["id"], "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
    return examples


class NERDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"], "labels": ex["labels"]}


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="checkpoints/phobert")
    args = parser.parse_args()

    train_images = load_images(args.train)
    val_images = load_images(args.val)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_examples = build_examples(train_images, tokenizer)
    val_examples = build_examples(val_images, tokenizer)
    train_dataset = NERDataset(train_examples)
    val_dataset = NERDataset(val_examples)
    val_ids = [ex["id"] for ex in val_examples]

    data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

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
