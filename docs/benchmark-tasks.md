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
| Evaluation | Field-level F1 with token-overlap matching |

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

**Evaluation protocol**: Each field group is scored independently, then pooled
into a micro-averaged precision / recall / F1 across all fields and all images
(a per-image macro-F1 is also reported).

- **Single-value fields** (`product_name`, `net_weight`, `manufacturer`,
  `origin`, `mfg_date`, `expiry_date`) contribute one unit each; a unit is a
  true positive when the prediction matches ground truth. Fields empty in both
  ground truth and prediction are skipped (not counted).
- **List fields** (`ingredients`, `additives`, `warnings`) are scored by greedy
  one-to-one matching between predicted and ground-truth items.
- **`nutrition_value`** is scored as a set of `"name: value"` pairs, matched the
  same way (this couples each nutrient name to its value).

**Matching is intentionally tolerant**, not exact-string. Two strings match when
their *token-overlap* (Jaccard over normalized word tokens) is at least **0.6**.
Normalization lowercases, collapses whitespace, and trims punctuation/brackets at
word edges only — internal punctuation is preserved, so Vietnamese decimal commas
(`6,3`) and additive code groups (`330,334`) keep their meaning. Ground truth and
predictions are stored verbatim (with original punctuation) and normalized only at
comparison time, so no information is lost for other tasks.

This tolerance is deliberate: the ground truth is a faithful transcription that
may contain on-label spelling/OCR artifacts (e.g. `Carbonhydrate`) and includes
the printed `%DV` in nutrition values; a model that reads the label correctly
should not be penalized for cosmetic differences. See
[task3-kie-record.md](task3-kie-record.md) for the record schema, ground-truth
construction, bilingual handling, and the exact normalization/matching rules.

> **Baseline status**: Task 3 is evaluated via the Tier C zero-shot MLLMs (image
> -> JSON directly) — see [baseline-models.md](baseline-models.md). A heuristic
> cascade chaining `src/ocr_engine.py` -> `src/ner_engine.py` ->
> `src/relation_extractor.py` -> `src/json_parser.py` exists for local
> reproduction only (not exposed as a service). No generative seq2seq
> (Donut-style) baseline is planned.

---

## Evaluation Script

```bash
python scripts/evaluate.py \
    --predictions results/predictions.json \
    --ground-truth data/processed/test.json \
    --task ser          # or: re, kie
```
