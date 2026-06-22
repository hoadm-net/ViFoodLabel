# Benchmark Tasks

ViFoodLabel supports three tasks of increasing difficulty.

---

## Task 1: Semantic Entity Recognition (SER)

**Goal**: Assign the correct BIO NER label to each word token given the product label image and its word-level bounding boxes.

| Property | Detail |
|---|---|
| Input | Image + word tokens + normalized bounding boxes |
| Output | BIO label sequence |
| Evaluation | Token-level F1, Entity-level F1 (span-exact match) |

**Token-level F1** measures per-token classification accuracy.  
**Entity-level F1** counts an entity as correct only if both the span boundaries and the entity type match exactly (no partial credit).

---

## Task 2: Relation Extraction (RE)

**Goal**: Given ground-truth entity spans, predict all `HAS_VALUE` relation pairs between `NUTRITION_NAME` and `NUTRITION_VALUE` entities.

| Property | Detail |
|---|---|
| Input | Entity spans + layout (gold NER labels given) |
| Output | Set of directed `(head, tail)` entity pairs |
| Evaluation | Relation F1 (exact head/tail span match) |

---

## Task 3: End-to-End Key Information Extraction (KIE)

**Goal**: From a raw product label image alone, produce a structured JSON record of all key food label fields — without intermediate ground-truth supervision.

| Property | Detail |
|---|---|
| Input | Image only |
| Output | Structured JSON (see format below) |
| Evaluation | Field-level F1, exact match per product |

**Output format:**

```json
{
  "product_name": "KẸO DẺO BOOM VỊ NHO",
  "net_weight": "52.5 g",
  "manufacturer": "Công ty TNHH Thực phẩm Orion Vina",
  "origin": "Việt Nam",
  "mfg_date": "31.12.25A1",
  "expiry_date": "6 tháng kể từ NSX",
  "ingredients": ["đường", "gelatin", "nước cốt nho", "..."],
  "additives": ["Chất làm dày (1422)", "..."],
  "nutrition_value": {
    "Năng lượng": "90 kcal",
    "Chất béo": "0 g",
    "Carbohydrate": "20 g",
    "Natri": "0 mg"
  },
  "warnings": ["Bảo quản nơi khô ráo, thoáng mát"]
}
```

**Evaluation protocol**: Each field is evaluated independently. String normalization (lowercase, whitespace) is applied before comparison. List fields (ingredients, additives, etc.) use set-based F1.

> **Baseline status**: Task 3 currently has no model-based results — only a heuristic cascade chaining `src/ocr_engine.py` -> `src/ner_engine.py` -> `src/relation_extractor.py` -> `src/json_parser.py` (run ad hoc for benchmarking, not exposed as a service). Planned baselines: zero-shot MLLM prompting (Tier C) and a fine-tuned generative seq2seq model (Tier D) — see [baseline-models.md](baseline-models.md).

---

## Evaluation Script

```bash
python scripts/evaluate.py \
    --predictions results/predictions.json \
    --ground-truth data/processed/test.json \
    --task ser          # or: re, kie
```
