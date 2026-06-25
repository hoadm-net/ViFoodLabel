"""Shared utilities for the SER baseline scripts (01-06).

Centralizes the parts that were duplicated verbatim across every baseline
(label set, data loading, dataset wrapper, live metric, prediction
re-attachment) plus two correctness fixes that must behave identically for all
models to keep the Tier A/B comparison fair:

1. **Sliding-window chunking** (`chunk_ranges`): long labels exceed every
   model's position budget — PhoBERT's is only ~256, so it would otherwise
   *see* ~58% of words while the 512-cap models see ~93%, making the
   comparison measure the cap, not the model. Each image is split into
   contiguous, non-overlapping word chunks that each fit the model's subword
   budget, so no document is truncated for any model.

2. **Word-index scatter** (`scatter_predictions`): predictions are mapped back
   to gold tokens by the explicit word index each kept label came from, not by
   positional order. Tokenizers silently drop empty-transcription words (no
   subword emitted), which otherwise shifts every later predicted label by one
   on the 59/419 images that contain such a token.
"""

from __future__ import annotations

import json

import numpy as np
from torch.utils.data import Dataset

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


def chunk_ranges(word_subword_lens: list[int], budget: int) -> list[tuple[int, int]]:
    """Greedy contiguous, non-overlapping word ranges that each fit `budget`
    subwords. Tiles the full word list (every index lands in exactly one range,
    including zero-length/empty words), so concatenating per-chunk predictions
    covers the whole document with no truncation and no overlap to deduplicate.

    `budget` should be the model's max_length minus its special tokens (2).
    """
    if not word_subword_lens:
        return [(0, 0)]
    ranges: list[tuple[int, int]] = []
    start = 0
    cur = 0
    for i, length in enumerate(word_subword_lens):
        if length <= 0:
            continue  # empty word: no subwords, stays in the current chunk's index span
        if cur > 0 and cur + length > budget:
            ranges.append((start, i))
            start = i
            cur = 0
        cur += length
    ranges.append((start, len(word_subword_lens)))
    return ranges


class NERDataset(Dataset):
    """Wraps example dicts; returns model tensors only (drops bookkeeping keys
    `id`/`word_index`, which are carried separately for scatter)."""

    _META_KEYS = {"id", "word_index"}

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return {k: v for k, v in self.examples[idx].items() if k not in self._META_KEYS}


def example_metas(examples: list[dict]) -> list[tuple[str, list[int]]]:
    """(image_id, word_index) per example, in dataset order — the key for scatter."""
    return [(ex["id"], ex["word_index"]) for ex in examples]


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


def scatter_predictions(
    logits: np.ndarray, labels: np.ndarray, metas: list[tuple[str, list[int]]], gold_images: list[dict]
) -> list[dict]:
    """Re-attach predicted labels to gold tokens by explicit word index.

    `metas[k]` is `(image_id, word_index)` for prediction row `k`; `word_index`
    lists the gold word position of each kept (first-subword) label, in order.
    Multiple chunks of the same image accumulate into one prediction record.
    Words never covered (truncated within a chunk, or unlabeled) default to "O".
    """
    pred_ids = np.argmax(logits, axis=-1)
    pred_labels_by_id = {g["id"]: ["O"] * len(g["tokens"]) for g in gold_images}
    for (img_id, word_index), label_seq, pred_seq in zip(metas, labels, pred_ids):
        kept = [ID2LABEL[p] for p, l in zip(pred_seq, label_seq) if l != -100]
        arr = pred_labels_by_id[img_id]
        for wi, lab in zip(word_index, kept):
            if 0 <= wi < len(arr):
                arr[wi] = lab
    pred_images = []
    for gold in gold_images:
        labs = pred_labels_by_id[gold["id"]]
        tokens = [{**t, "label": labs[i]} for i, t in enumerate(gold["tokens"])]
        pred_images.append({"id": gold["id"], "tokens": tokens, "relations": gold.get("relations", [])})
    return pred_images
