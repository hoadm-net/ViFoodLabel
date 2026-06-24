#!/usr/bin/env python
"""XLM-RoBERTa (xlm-roberta-base) multilingual text-only token classification baseline.

Usage:
    python scripts/baselines/02_xlmr.py \\
        --train data/processed/train.json \\
        --val   data/processed/val.json \\
        --epochs 20

Same flow as 01_phobert.py, but XLM-R's tokenizer is a fast (Rust-backed)
SentencePiece tokenizer that supports `word_ids()` directly, so word-to-
subword label alignment doesn't need the manual per-word BPE-encode
workaround PhoBERT's slow tokenizer requires.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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

MODEL_NAME = "xlm-roberta-base"
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


def build_examples(images: list[dict], tokenizer) -> list[dict]:
    examples = []
    for img in images:
        words = [t["text"] for t in img["tokens"]]
        word_label_ids = [LABEL2ID.get(t["label"], LABEL2ID["O"]) for t in img["tokens"]]
        encoding = tokenizer(words, is_split_into_words=True, truncation=True, max_length=MAX_LENGTH)
        labels = []
        prev_word_id = None
        for word_id in encoding.word_ids():
            if word_id is None or word_id == prev_word_id:
                labels.append(-100)  # special token or continuation subword
            else:
                labels.append(word_label_ids[word_id])
            prev_word_id = word_id
        examples.append({
            "id": img["id"],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        })
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
    parser.add_argument("--output-dir", default="checkpoints/xlmr")
    args = parser.parse_args()

    train_images = load_images(args.train)
    val_images = load_images(args.val)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
