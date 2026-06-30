# Intended Uses and Supported Tasks

The annotations in ViFoodLabel support three information-extraction tasks of
increasing difficulty. They are described here as **intended uses of the data** —
to document what each label layer enables, not to define a benchmark or report
results.

---

## Task 1: Semantic Entity Recognition (SER)

Assign the correct BIO entity label to each word token, given the label image and
its word-level bounding boxes.

| Property | Detail |
|---|---|
| Inputs available | Image + word tokens + normalized bounding boxes |
| Target | BIO label sequence (11 entity types) |
| Data layer used | Token-level `tokens[].label` in `data/processed/{train,dev,test}.json` |

---

## Task 2: Relation Extraction (RE)

Given entity spans, recover all `HAS_VALUE` relation pairs between
`NUTRITION_NAME` and `NUTRITION_VALUE` entities.

| Property | Detail |
|---|---|
| Inputs available | Entity spans + layout |
| Target | Set of directed `(head, tail)` entity pairs |
| Data layer used | Token-level `relations[]` |

---

## Task 3: End-to-End Key Information Extraction (KIE)

From a raw product label image alone, produce a structured JSON record of all key
food-label fields.

| Property | Detail |
|---|---|
| Inputs available | Image only |
| Target | Structured JSON record |
| Data layer used | Per-image KIE record `data/processed/dataset/<id>.json` (`kie`) |

**Record format:**

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
  "nutrition_value": {"Năng lượng": "90 kcal", "Carbohydrate": "20 g 7%"},
  "warnings": ["Bảo quản nơi khô ráo, tránh ánh nắng trực tiếp."]
}
```

The three field groups (single-value strings, lists, and the paired
`nutrition_value` object) and how the record is assembled from the token
annotation are documented in [task3-kie-record.md](task3-kie-record.md).

---

Records are stored **verbatim** (original casing, diacritics, on-label OCR/spelling
artifacts, printed `%DV`), so any downstream comparison should normalize before
matching rather than relying on exact strings. The recommended normalization and
token-overlap comparison are described in
[task3-kie-record.md](task3-kie-record.md).
