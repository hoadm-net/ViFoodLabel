# Data Dictionary

Every released file and every field, with types and allowed values. Coordinates
are normalized to `[0, 1000]` on both axes unless stated otherwise.

## Released files

| Path | Description |
|---|---|
| `images/<id>.jpeg` | Original product label image |
| `label_studio/data.json` | Anonymized Label Studio annotation export (one task per image) |
| `processed/dataset/<id>.json` | Per-image KIE record (descriptive metadata + structured fields) |
| `processed/dataset_meta.json` | All per-image metadata in one object, keyed by image id |
| `processed/splits.json` | Frozen train/dev/test image-id lists |
| `processed/statistics.json` | Computed dataset statistics |

`<id>` is a zero-padded 4-digit image identifier (e.g. `0001`).

---

## 1. `label_studio/data.json`

A JSON list of task objects (anonymized). Per task:

| Field | Type | Description |
|---|---|---|
| `id` | int | Task identifier |
| `data.image` | string | Relative path to the image (e.g. `images/0001.jpeg`) |
| `data.thumbnail` | string | Relative path to the thumbnail |
| `annotations` | list | One annotation object per task |

Each `annotations[]` object:

| Field | Type | Description |
|---|---|---|
| `completed_by` | string | Anonymized annotator id (`annotator_1` … `annotator_4`) |
| `result` | list | Region/relation objects (below) |

`result[]` has three object kinds, distinguished by `type`:

| `type` | `from_name` | Key fields | Meaning |
|---|---|---|---|
| `rectanglelabels` | `bbox_tool` | `value.x`, `value.y`, `value.width`, `value.height` (percent 0–100), `value.rectanglelabels` (`[BIO label]`), `id` | One word bounding box + its BIO tag |
| `textarea` | `transcription` | `value.text` (`[string]`), `id` | The transcription of the box sharing the same `id` |
| `relation` | — | `from_id`, `to_id`, `direction` | A `HAS_VALUE` link from a `NUTRITION_NAME` box to a `NUTRITION_VALUE` box |

All relations are of type `HAS_VALUE` (the only relation in the schema), so they
carry no separate label.

---

## 2. `processed/dataset/<id>.json` — KIE record

| Field | Type | Description |
|---|---|---|
| `id` | string | Image id |
| `image` | string | Image filename |
| `meta` | object | Descriptive metadata (below) |
| `kie` | object | Structured key-information fields (below) |

### `meta`

| Field | Type | Values |
|---|---|---|
| `token_count` | int | Number of word tokens in the image |
| `entity_counts` | object | `{ENTITY_TYPE: int}` over the 11 types |
| `num_relations` | int | Number of `HAS_VALUE` relations |
| `image_width`, `image_height` | int | Original pixel dimensions |
| `product_category` | enum | `cake`, `candy`, `snack`, `beverage`, `seasoning`, `dried_food`, `other` |
| `background_color` | enum | `red`, `yellow`, `green`, `blue`, `white`, `other` |
| `material` | enum | `paper`, `plastic`, `foil`, `can_bottle`, `other` |

`product_category`, `background_color`, and `material` are auxiliary descriptive
attributes (vision-model proposed, annotator-corrected); they support stratified
splitting and are **not** extraction targets.

### `kie`

| Field | Type | Description |
|---|---|---|
| `product_name` | string | Commercial product name |
| `net_weight` | string | Net weight/volume declaration |
| `manufacturer` | string | Producer/packer name and address |
| `origin` | string | Country/region of manufacture |
| `mfg_date` | string | Manufacturing date / batch-lot code (may be empty) |
| `expiry_date` | string | Best-before/use-by date (may be empty) |
| `ingredients` | list of strings | Individual ingredients |
| `additives` | list of strings | Additives (functional name + Codex code) |
| `warnings` | list of strings | Safety/allergen/storage statements |
| `nutrition_value` | object | `{nutrient_name: value}` pairs from the nutrition table |

Values are stored **verbatim** (original casing, diacritics, on-label artifacts,
printed `%DV`). Empty single-value fields are the empty string `""`.

---

## 3. `processed/dataset_meta.json`

A single JSON object keyed by image id, each value being the `meta` object from
the corresponding KIE record (Section 2).

```json
{ "0001": { "token_count": 287, "entity_counts": {"ADDITIVE": 7, "...": 0}, "...": "..." } }
```

---

## 4. `processed/splits.json`

| Field | Type | Description |
|---|---|---|
| `seed` | int | Random seed used for the split |
| `ratios` | object | `{train, dev, test}` target fractions (0.8/0.1/0.1) |
| `strata_key` | list | Fields used for stratification (`product_category`, `background_color`, `material`) |
| `n_images` | int | Total images split |
| `counts` | object | Image count per split |
| `train`, `dev`, `test` | list of strings | Image ids in each split |

---

## 5. Token-level layer (derivable)

`scripts/preprocessing/convert_ls_to_ner.py` produces a token-level view of each
image (used for SER/RE). Per image:

| Field | Type | Description |
|---|---|---|
| `id`, `image` | string | Image id / filename |
| `image_width`, `image_height` | int | Original pixel dimensions |
| `tokens` | list | `{text: string, bbox: [x0,y0,x1,y1] in [0,1000], label: BIO string}` |
| `relations` | list | `{from_id: "token_NNN", to_id: "token_NNN", type: "HAS_VALUE"}` |

`from_id`/`to_id` index the first (`B-`) token of each entity span.

### Entity types and BIO labels

The 11 entity types: `PRODUCT_NAME`, `INGREDIENT`, `ADDITIVE`, `NUTRITION_NAME`,
`NUTRITION_VALUE`, `MANUFACTURER`, `ORIGIN`, `NET_WEIGHT`, `MFG_DATE`,
`EXPIRY_DATE`, `WARNING`. Each appears as `B-<TYPE>` (span start) and `I-<TYPE>`
(continuation); tokens outside any entity are `O`. See
[annotation/entity-schema.md](annotation/entity-schema.md).

---

## 6. `processed/statistics.json`

Computed by `scripts/dataset/compute_statistics.py`: dataset totals, per-split
sizes, token-length summary, entity distribution (with long-tail flags),
relation-count summary, and `product_category`/`background_color`/`material`
distributions. See [dataset-statistics.md](dataset-statistics.md) for the current
values.
