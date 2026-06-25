#!/usr/bin/env python
"""PhoBERT (vinai/phobert-base) text-only token classification baseline.

Usage:
    python scripts/baselines/01_phobert.py \\
        --train data/processed/train.json \\
        --val   data/processed/val.json \\
        --epochs 20

Note (docs/notes/technical-runtime-notes.md #1-2): PhoBERT's effective max
position embedding is ~258 -> content is capped at max_length=256. Labels here
routinely exceed that, so instead of truncating (which would drop ~40% of words
and unfairly depress PhoBERT vs the 512-cap models) each image is split into
contiguous word chunks that each fit the budget (baseline_common.chunk_ranges).
PhoBERT's tokenizer needs explicit word-to-subword label alignment, with
continuation subwords set to the ignored label index (-100). The slow
PhobertTokenizer also has no `word_ids()` (fast-tokenizer-only API), so each
word is BPE-encoded individually to find its subword span; the gold word index
of every kept label is recorded for prediction scatter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_common import (  # noqa: E402
    ID2LABEL, LABEL2ID, LABELS, NERDataset, chunk_ranges, compute_metrics,
    example_metas, load_images, scatter_predictions,
)
from src.metrics import entity_f1, token_f1  # noqa: E402

MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 256


def build_examples(images: list[dict], tokenizer) -> list[dict]:
    """One example per word chunk; each chunk fits MAX_LENGTH subwords.

    BPE-encode each word individually, assign its label to the first subword and
    -100 to continuation subwords; record the gold word index of each kept label.
    """
    examples = []
    for img in images:
        words = [t["text"] for t in img["tokens"]]
        word_label_ids = [LABEL2ID.get(t["label"], LABEL2ID["O"]) for t in img["tokens"]]
        sub_lens = [len(tokenizer.tokenize(w)) for w in words]
        for start, end in chunk_ranges(sub_lens, MAX_LENGTH - 2):
            input_ids = [tokenizer.bos_token_id]
            labels = [-100]
            word_index: list[int] = []
            for wi in range(start, end):
                piece_ids = tokenizer.encode(words[wi], add_special_tokens=False)
                if not piece_ids:
                    continue
                if len(input_ids) + len(piece_ids) + 1 > MAX_LENGTH:  # +1 reserves room for eos
                    break
                input_ids.extend(piece_ids)
                labels.append(word_label_ids[wi])
                labels.extend([-100] * (len(piece_ids) - 1))
                word_index.append(wi)
            input_ids.append(tokenizer.eos_token_id)
            labels.append(-100)
            examples.append({
                "id": img["id"],
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "word_index": word_index,
            })
    return examples


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

    train_dataset = NERDataset(build_examples(train_images, tokenizer))
    val_examples = build_examples(val_images, tokenizer)
    val_dataset = NERDataset(val_examples)
    val_metas = example_metas(val_examples)

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
    pred_images = scatter_predictions(predict_output.predictions, predict_output.label_ids, val_metas, val_images)

    report = {**token_f1(pred_images, val_images), **entity_f1(pred_images, val_images)}
    print(json.dumps(
        {k: v for k, v in report.items() if not k.endswith("_per_label")},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
