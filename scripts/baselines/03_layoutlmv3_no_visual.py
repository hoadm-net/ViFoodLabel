#!/usr/bin/env python
"""LayoutLMv3 (microsoft/layoutlmv3-base) text + layout, visual segment disabled.

Usage:
    python scripts/baselines/03_layoutlmv3_no_visual.py \\
        --train  data/processed/train.json \\
        --val    data/processed/val.json \\
        --epochs 30

Boxes in data/processed/*.json are already normalized to [0, 1000] by
convert_ls_to_ner.py, so no image loading/scaling is needed here (unlike
04_layoutlmv3_visual.py, which needs the actual pixel patches). The
tokenizer's `boxes=`/`word_labels=` API handles BPE-splitting, per-token
box broadcast, and word-to-subword label alignment (-100 for continuations)
in one call. Long labels are split into word chunks
(baseline_common.chunk_ranges) so nothing is truncated; the gold word index
of each kept label is taken from `word_ids()` for prediction scatter.

The visual segment is disabled via `visual_embed=False` on the config,
which also makes `pixel_values` unnecessary in the forward pass.
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

MODEL_NAME = "microsoft/layoutlmv3-base"
MAX_LENGTH = 512


def build_examples(images: list[dict], tokenizer) -> list[dict]:
    examples = []
    for img in images:
        words = [t["text"] for t in img["tokens"]]
        boxes = [t["bbox"] for t in img["tokens"]]
        word_label_ids = [LABEL2ID.get(t["label"], LABEL2ID["O"]) for t in img["tokens"]]
        sub_lens = [len(tokenizer.tokenize(w)) for w in words]
        for start, end in chunk_ranges(sub_lens, MAX_LENGTH - 2):
            encoding = tokenizer(
                text=words[start:end], boxes=boxes[start:end], word_labels=word_label_ids[start:end],
                truncation=True, max_length=MAX_LENGTH,
            )
            word_index = []
            prev_word_id = None
            for word_id in encoding.word_ids():
                if word_id is not None and word_id != prev_word_id:
                    word_index.append(start + word_id)
                prev_word_id = word_id
            examples.append({
                "id": img["id"],
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "bbox": encoding["bbox"],
                "labels": encoding["labels"],
                "word_index": word_index,
            })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="checkpoints/layoutlmv3_no_visual")
    args = parser.parse_args()

    train_images = load_images(args.train)
    val_images = load_images(args.val)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID, visual_embed=False
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
