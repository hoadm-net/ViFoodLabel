"""Proposed model-based HAS_VALUE relation extraction (Tier D contribution).

Replaces the geometric heuristic in src/relation_extractor.py with a
lightweight learned typed-link predictor (GLiNER-Relex / Parallel-Pointer-
Network style): score candidate (NUTRITION_NAME, NUTRITION_VALUE) pairs
from their text+layout embeddings instead of relying on coordinates alone.

This is the dataset paper's proposed Vietnamese-specific contribution — see
docs/baseline-models.md Tier D. Benchmark against
`relation_extractor.extract_relations` (heuristic baseline) using
`src/metrics.py::relation_f1`.

The encoder (PhoBERT by default) is frozen -- only the small pairwise MLP
head is trained. This keeps the model "lightweight" as the module name
promises, and is cheap enough to smoke-test on a handful of images.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.relation_extractor import Relation, split_nutrition_entities

GEOMETRY_FEATURE_DIM = 10  # name bbox(4) + value bbox(4) + dx + dy


@dataclass
class RelationModelConfig:
    encoder_name: str = "vinai/phobert-base"
    hidden_dim: int = 128
    max_text_length: int = 32
    pair_batch_size: int = 512  # candidate pairs scored per forward at predict time


class RelationModel:
    """Scores candidate (NUTRITION_NAME, NUTRITION_VALUE) entity pairs and
    predicts HAS_VALUE links above a probability threshold.
    """

    def __init__(self, config: RelationModelConfig):
        self.config = config
        self._encoder = None
        self._tokenizer = None
        self._scorer = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self) -> None:
        if self._encoder is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_name, use_fast=False)
        self._encoder = AutoModel.from_pretrained(self.config.encoder_name).to(self._device).eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)
        feat_dim = self._encoder.config.hidden_size * 2 + GEOMETRY_FEATURE_DIM
        self._scorer = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, self.config.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.hidden_dim, 1),
        ).to(self._device)

    def _embed_text(self, texts: list[str]) -> torch.Tensor:
        """Mean-pooled encoder embedding per text (encoder is frozen)."""
        encoding = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self.config.max_text_length, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            hidden = self._encoder(**encoding).last_hidden_state
        mask = encoding["attention_mask"].unsqueeze(-1)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    @staticmethod
    def _geometry(name: dict, value: dict) -> list[float]:
        nb, vb = name["bbox"], value["bbox"]
        dx = (vb[0] - nb[2]) / 1000.0
        dy = ((vb[1] + vb[3]) / 2 - (nb[1] + nb[3]) / 2) / 1000.0
        return [c / 1000.0 for c in nb] + [c / 1000.0 for c in vb] + [dx, dy]

    def _pair_features(self, pairs: list[tuple[dict, dict]]) -> torch.Tensor:
        name_emb = self._embed_text([name["text"] for name, _ in pairs])
        value_emb = self._embed_text([value["text"] for _, value in pairs])
        geometry_t = torch.tensor(
            [self._geometry(name, value) for name, value in pairs], dtype=torch.float32, device=self._device
        )
        return torch.cat([name_emb, value_emb, geometry_t], dim=-1)

    def _score_examples(self, examples: list[dict]) -> torch.Tensor:
        pairs = [(ex["name"], ex["value"]) for ex in examples]
        return self._scorer(self._pair_features(pairs)).squeeze(-1)

    def _embed_entities(self, entities: list[dict]) -> torch.Tensor:
        """Encode each entity's text once (batched), so a name/value appearing in
        many candidate pairs isn't re-encoded per pair."""
        embs = []
        bs = 64
        for start in range(0, len(entities), bs):
            embs.append(self._embed_text([e["text"] for e in entities[start : start + bs]]))
        return torch.cat(embs, dim=0)

    def train(
        self,
        train_examples: list[dict],
        val_examples: list[dict],
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 2e-4,
    ) -> None:
        """`train_examples`/`val_examples`: pair dicts with keys "name", "value"
        (entity dicts from src.relation_extractor.entities_from_tokens) and
        "label" (1 for an annotated HAS_VALUE relation, 0 otherwise) — built by
        scripts/baselines/07_relation_model.py: build_pairs from data/processed/.
        """
        self._load()
        optimizer = torch.optim.AdamW(self._scorer.parameters(), lr=lr)
        examples = list(train_examples)

        # Candidate pairs are every NUTRITION_NAME x NUTRITION_VALUE combination per
        # image, so true relations (positives) are a small minority -- without
        # reweighting, BCE drives the model to just predict "no relation" everywhere
        # (high pair-accuracy, but a degenerate, all-zero-recall classifier).
        n_pos = sum(ex["label"] for ex in examples)
        n_neg = len(examples) - n_pos
        pos_weight = torch.tensor(n_neg / n_pos if n_pos else 1.0, device=self._device)

        for epoch in range(1, epochs + 1):
            random.shuffle(examples)
            total_loss = 0.0
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                if not batch:
                    continue
                logits = self._score_examples(batch)
                labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.float32, device=self._device)
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            train_loss = total_loss / len(examples) if examples else 0.0
            val_acc = self._pair_accuracy(val_examples, batch_size=batch_size)
            print(f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_pair_acc={val_acc:.4f}")

    def _pair_accuracy(self, examples: list[dict], batch_size: int = 16) -> float:
        if not examples:
            return 0.0
        correct = 0
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                logits = self._score_examples(batch)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.float32, device=self._device)
                correct += (preds == labels).sum().item()
        return correct / len(examples)

    def predict(self, entities: list, threshold: float = 0.5) -> list[Relation]:
        """Score all candidate (NUTRITION_NAME, NUTRITION_VALUE) pairs in
        `entities` and return those scoring above `threshold` as Relations,
        greedily assigned one-to-one (highest score first), mirroring the
        heuristic's exclusivity so the two are directly comparable.

        All name x value pairs are scored (no per-image cap): entity texts are
        embedded once each and the pairs are scored in batches, so even an image
        with dozens of nutrition rows (thousands of pairs) is handled without
        either truncating candidates or running out of memory. Capping pairs
        would unfairly handicap the model relative to the uncapped heuristic.
        """
        names, values = split_nutrition_entities(entities)
        if not names or not values:
            return []
        self._load()
        name_emb = self._embed_entities(names)
        value_emb = self._embed_entities(values)
        pairs = [(hi, vi) for hi in range(len(names)) for vi in range(len(values))]

        scores: list[float] = []
        bs = self.config.pair_batch_size
        with torch.no_grad():
            for start in range(0, len(pairs), bs):
                chunk = pairs[start : start + bs]
                ne = name_emb[[hi for hi, _ in chunk]]
                ve = value_emb[[vi for _, vi in chunk]]
                geo = torch.tensor(
                    [self._geometry(names[hi], values[vi]) for hi, vi in chunk],
                    dtype=torch.float32, device=self._device,
                )
                feats = torch.cat([ne, ve, geo], dim=-1)
                scores.extend(torch.sigmoid(self._scorer(feats).squeeze(-1)).tolist())

        ranked = sorted(zip(pairs, scores), key=lambda item: -item[1])
        used_names: set[int] = set()
        used_values: set[int] = set()
        relations: list[Relation] = []
        for (hi, vi), score in ranked:
            if score < threshold or hi in used_names or vi in used_values:
                continue
            used_names.add(hi)
            used_values.add(vi)
            relations.append(Relation(head_entity_idx=hi, tail_entity_idx=vi))
        return relations
