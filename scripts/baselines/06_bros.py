#!/usr/bin/env python
"""BROS (naver-clova-ocr/bros-base-uncased) text + layout token classification baseline.

Usage:
    python scripts/baselines/06_bros.py \\
        --train data/processed/train.json \\
        --val   data/processed/val.json \\
        --epochs 30

No Vietnamese or multilingual BROS checkpoint exists — it was pretrained on
English IE benchmarks (FUNSD/CORD/SROIE) with a BERT-base-uncased English
tokenizer, which lowercases and strips diacritics (Vietnamese text survives
as recognizable but accent-stripped Latin subwords, not `[UNK]`-soup). This
is a known, expected literature-baseline mismatch — we run the released
checkpoint as-is rather than skip the model, consistent with how Tier C
reports MLLM degradation patterns rather than omitting a weaker model.

BROS expects `bbox` as floats normalized to [0, 1] (not [0, 1000] ints like
LayoutLM/LiLT) and accepts 4-point `[x0, y0, x1, y1]` boxes (`BrosModel`
expands them to quads internally). Its tokenizer has no `boxes=` convenience
API, so word-to-subword label *and* bbox alignment is done manually via
`word_ids()`, same approach as 02_xlmr.py. Long labels are split into word
chunks (baseline_common.chunk_ranges) so nothing is truncated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
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

MODEL_NAME = "naver-clova-ocr/bros-base-uncased"
MAX_LENGTH = 512

PAD_BOX = [0.0, 0.0, 0.0, 0.0]


def restore_pretrained_bbox_projection(model) -> None:
    """Copy the pretrained spatial-projection weight the released checkpoint
    stores under `embeddings.bbox_projection.weight` into the slot current
    `transformers` expects (`bros.bbox_embeddings.bbox_projection.weight`).

    The two naming schemes diverged, so `from_pretrained` silently leaves this
    one learned layout parameter randomly initialized (the sinusoidal
    `inv_freq` buffers are recomputed identically and need no loading). Without
    this, BROS would learn its bbox projection from scratch during fine-tuning
    instead of from pretraining — an unfair handicap vs the other layout models.
    """
    import torch
    from huggingface_hub import hf_hub_download

    key = "embeddings.bbox_projection.weight"
    try:
        from safetensors.torch import load_file
        state = load_file(hf_hub_download(MODEL_NAME, "model.safetensors"))
    except Exception:
        state = torch.load(hf_hub_download(MODEL_NAME, "pytorch_model.bin"), map_location="cpu", weights_only=True)

    weight = state.get(key)
    target = model.bros.bbox_embeddings.bbox_projection.weight
    if weight is None or tuple(weight.shape) != tuple(target.shape):
        print(f"warning: could not restore pretrained {key}; leaving it as initialized")
        return
    with torch.no_grad():
        target.copy_(weight)


def build_examples(images: list[dict], tokenizer) -> list[dict]:
    examples = []
    for img in images:
        words = [t["text"] for t in img["tokens"]]
        boxes = [[c / 1000.0 for c in t["bbox"]] for t in img["tokens"]]  # normalize [0,1000] -> [0,1]
        word_label_ids = [LABEL2ID.get(t["label"], LABEL2ID["O"]) for t in img["tokens"]]
        sub_lens = [len(tokenizer.tokenize(w)) for w in words]
        for start, end in chunk_ranges(sub_lens, MAX_LENGTH - 2):
            encoding = tokenizer(words[start:end], is_split_into_words=True, truncation=True, max_length=MAX_LENGTH)
            labels, bbox, word_index = [], [], []
            prev_word_id = None
            for word_id in encoding.word_ids():
                if word_id is None:
                    labels.append(-100)
                    bbox.append(PAD_BOX)
                elif word_id == prev_word_id:
                    labels.append(-100)  # continuation subword
                    bbox.append(boxes[start + word_id])
                else:
                    labels.append(word_label_ids[start + word_id])
                    bbox.append(boxes[start + word_id])
                    word_index.append(start + word_id)
                prev_word_id = word_id
            examples.append({
                "id": img["id"],
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "bbox": bbox,
                "labels": labels,
                "word_index": word_index,
            })
    return examples


class BrosDataCollator:
    """Pads input_ids/attention_mask/labels via the base text collator, then
    pads the parallel float `bbox` sequence to the same length on the same side."""

    def __init__(self, tokenizer):
        self.base_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

    def __call__(self, features: list[dict]) -> dict:
        bboxes = [f["bbox"] for f in features]
        text_features = [{k: v for k, v in f.items() if k != "bbox"} for f in features]
        batch = self.base_collator(text_features)
        max_len = batch["input_ids"].shape[1]
        padded = [b + [PAD_BOX] * (max_len - len(b)) for b in bboxes]
        batch["bbox"] = torch.tensor(padded, dtype=torch.float32)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="checkpoints/bros")
    args = parser.parse_args()

    train_images = load_images(args.train)
    val_images = load_images(args.val)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    restore_pretrained_bbox_projection(model)

    train_dataset = NERDataset(build_examples(train_images, tokenizer))
    val_examples = build_examples(val_images, tokenizer)
    val_dataset = NERDataset(val_examples)
    val_metas = example_metas(val_examples)

    data_collator = BrosDataCollator(tokenizer)

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
