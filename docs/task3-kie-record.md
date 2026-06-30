# KIE Record: Schema, Ground Truth, and Quality Control

This document specifies the structured **per-image KIE record** — the published
`data/processed/dataset/<id>.json` `kie` field — how it is derived from the
word-level annotation and how its quality is controlled. See
[dataset-overview.md](dataset-overview.md) for the surrounding dataset record and
[benchmark-tasks.md](benchmark-tasks.md) for the tasks it supports.

## 1. Why a derived record is needed

Label Studio stores annotations at the **word level**: each region carries a
transcription, a BIO label (one of 11 entity types), a bounding box, and — for
nutrition rows — a `HAS_VALUE` relation. The KIE record is a clean, field-grouped
**JSON record per product**:

```json
{
  "product_name": "KẸO DẺO BOOM VỊ NHO",
  "net_weight": "52,5g",
  "manufacturer": "CÔNG TY TNHH THỰC PHẨM ORION VINA.",
  "origin": "VIỆT NAM",
  "mfg_date": "31.12.25A1 13:16",
  "expiry_date": "",
  "ingredients": ["ĐƯỜNG", "MẠCH NHA", "MALTOSE,", "GELATIN"],
  "additives": ["Chất làm dày (1442)", "Chất điều chỉnh độ acid (330, 334, 296)"],
  "warnings": ["Bảo quản nơi khô ráo, tránh ánh nắng trực tiếp."],
  "nutrition_value": {"Năng lượng": "90 kcal", "Carbohydrate": "20 g 7%"}
}
```

No such record is annotated directly — it is **assembled** from the word-level
labels and relations, and is part of the published dataset.

### Field types

| Group | Fields | Type |
|---|---|---|
| Single-value | `product_name`, `net_weight`, `manufacturer`, `origin`, `mfg_date`, `expiry_date` | string |
| List | `ingredients`, `additives`, `warnings` | list of strings |
| Paired | `nutrition_value` | object `{name: value}` |

## 2. Ground-truth construction

Implemented by `scripts/preprocessing/convert_ls_to_ner.py` (token preparation)
and `scripts/task3_schema.py` + `scripts/build_task3_gt.py` (record assembly).

1. **Coordinate normalization** — Label Studio percentages → `[0, 1000]`.
2. **Reading-order sort** — tokens are grouped into lines by vertical center
   using an adaptive tolerance (≈ 0.6 × median token height), then ordered
   left-to-right within a line and top-to-bottom across lines. This is robust to
   per-token vertical jitter and reduces fragmentation of long list fields
   (ingredients/additives/warnings).
3. **BIO span merging** — consecutive `B-/I-` tokens of the same type are merged
   into one phrase entity (`scripts/task3_schema.py: merge_entities`).
4. **Field routing** — single-value entity types fill their string field; list
   entity types accumulate into their list.
5. **Nutrition pairing** — each `HAS_VALUE` relation links a `NUTRITION_NAME`
   entity to its `NUTRITION_VALUE` entity, producing one `nutrition_value`
   object entry. Pairing comes from the relation graph, not from text order.

### Bilingual labels (VI + EN)

Per the annotation guideline ([annotation/annotation.md §4](annotation/annotation.md)):

| Entity | Rule | Effect on the record |
|---|---|---|
| `NUTRITION_NAME` | merge VI+EN into one span | one key, e.g. `"Calories/ Năng lượng"` |
| `WARNING` (allergen) | merge if consecutive | one list item |
| `INGREDIENT` | two separate lists → two entities | both languages appear as list items |
| `ORIGIN` | two separate entities | annotator keeps one canonical string |
| `MANUFACTURER` | merge if consecutive; split if interrupted | usually one string |

Because bilingual ingredient lists yield items in **both** languages, a faithful
record transcribes both — this is a deliberate "read what is printed" design.

## 3. Quality control (LLM-flag → human fix)

The geometric reading-order sort resolves most fragmentation but cannot recover
inherently ambiguous cases (overlapping boxes, multi-column nutrition tables,
wrapped bilingual warnings). These are caught by a **text-only LLM linter**
(`scripts/task3_qc_lint.py`):

1. The linter reads the **assembled text only** (never the image) and flags
   records whose fields look clearly broken (mid-phrase splits, duplicated text,
   stray leading punctuation, column bleed), with a severity rank.
2. An annotator opens the flagged images, looks at the **actual image**, and
   edits the affected record (`data/processed/dataset/<id>.json`) directly.
3. The corrected record is the frozen ground truth.

The linter only *triages* which images a human should re-open; the human authors
the correction from the image. The same independence applies to the `meta` visual
attributes (see [dataset-overview.md](dataset-overview.md)).

## 4. Verbatim storage and normalization

**Storage is verbatim; normalization happens only when records are compared.**
Ground truth keeps its original punctuation so the data stays useful for other
tasks (e.g. SER). The recommended normalization
(`scripts/detokenize_bio.py: normalize_text`):

- lowercases and collapses whitespace,
- trims punctuation/brackets at **word edges only**, preserving internal
  punctuation so Vietnamese decimal commas (`6,3`) and additive code groups
  (`330,334`) are not corrupted.

For downstream comparison of two records, a **token-overlap** match (Jaccard over
normalized word tokens, `scripts/task3_schema.py: token_overlap`) is recommended
rather than exact-string match, because faithful transcriptions may include
on-label spelling/OCR artifacts (e.g. `Carbonhydrate`) and the printed `%DV`
(e.g. `20 g 7%`). `scripts/task3_schema.py: score` implements one such
field-level comparison (single-value units, greedy list matching, and
`name: value` pairing for `nutrition_value`).

## 5. Reproduction

```bash
# 1. Token prep with reading-order sort
.venv/bin/python3 scripts/preprocessing/convert_ls_to_ner.py \
    --input data/label_studio/data.json --output data/processed/ --autofix-bio

# 2. Assemble KIE records + per-image dataset files (meta + kie)
.venv/bin/python3 scripts/build_task3_gt.py
.venv/bin/python3 scripts/build_dataset_meta.py

# 3. QC worklist for human review
.venv/bin/python3 scripts/task3_qc_lint.py
```
