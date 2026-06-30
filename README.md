# ViFoodLabel: A Vietnamese Food Product Label Dataset for Key Information Extraction

![Type: Data Article](https://img.shields.io/badge/Type-Data%20Article-blue)
![Language: Vietnamese](https://img.shields.io/badge/Language-Vietnamese-red)
![Domain: Food Product Labels](https://img.shields.io/badge/Domain-Food%20Product%20Labels-green)
![Annotation: BIO NER + Layout + Relations](https://img.shields.io/badge/Annotation-BIO%20NER%20%2B%20Layout%20%2B%20Relations-orange)
![Dataset License: CC BY-NC 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY--NC%204.0-lightgrey)
![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow)

---

## Overview

**ViFoodLabel** is a layout-aware dataset for **Key Information Extraction (KIE)** from Vietnamese food product labels. Each label image is annotated at the word level with a tight bounding box, a transcription, a BIO entity tag (one of **11 semantic types**), and — for nutrition rows — a `HAS_VALUE` relation linking each nutrient name to its value. From these word-level annotations a clean, field-grouped **per-image KIE record** is assembled (product name, ingredients, additives, nutrition table, manufacturer, origin, net weight, dates, warnings).

Vietnamese food labels are a demanding source for visually-rich document understanding: dense multi-column ingredient and nutrition layouts, diacritics and tone marks that OCR frequently misreads, non-standard typography, and domain vocabulary (Codex additive codes, regulatory terminology). The dataset is released to support research on document AI for Vietnamese and other under-resourced languages, and on food-safety information systems (allergen, additive, and age-warning extraction).

This repository is a **data article companion**: it holds the dataset, the annotation specifications, and the open-source tooling used to build, validate, and describe it. It does **not** contain model training, benchmarking, or a serving layer.

> **Status (2026-06-30):** 560 raw label images collected; **416 images annotated** and processed so far. Statistics, splits, and the data dictionary below describe the current annotated set and will be refreshed when annotation reaches the full target. See [docs/dataset-statistics.md](docs/dataset-statistics.md).

---

## Specifications

| | |
|---|---|
| **Subject** | Computer Science — Artificial Intelligence; Document AI / NLP |
| **Specific subject area** | Layout-aware Key Information Extraction from Vietnamese food product label images |
| **Data type** | Images (JPEG); annotations (JSON); derived structured records (JSON) |
| **How data were acquired** | Self-photographed from physical products in Vietnamese supermarkets and convenience stores; annotated in [Label Studio](https://labelstud.io/) |
| **Data format** | Raw and processed (token-level BIO + bounding boxes + relations; per-image KIE records) |
| **Annotation** | Word-level bounding box + transcription + BIO tag (11 entity types) + `HAS_VALUE` relation |
| **Coordinate system** | Normalized to `[0, 1000]` on both axes (LayoutLM-compatible) |
| **Data source location** | Vietnam |
| **Data accessibility** | Images + annotations released on Mendeley Data (link added on publication); tooling in this repository |
| **License** | Dataset: CC BY-NC 4.0 · Code: MIT |

---

## Value of the Data

- **First layout-aware KIE dataset for Vietnamese food labels.** Existing KIE datasets (FUNSD, CORD, SROIE, DocVQA) are English and cover forms or receipts; ViFoodLabel fills a language and domain gap with pixel-accurate boxes, BIO spans, and nutrition relations.
- **Multiple levels of supervision in one resource.** Word-level tokens/boxes/tags, entity spans, `HAS_VALUE` relations, and assembled per-image field records support semantic entity recognition, relation extraction, and end-to-end information extraction without further annotation.
- **Faithful, real-world transcriptions.** Text is stored verbatim (original casing, diacritics, on-label OCR/spelling artifacts, printed `%DV`), so the data is reusable for OCR, normalization, and robustness studies, not only clean extraction.
- **Direct relevance to food safety.** Structured ingredient, additive, allergen, and warning fields enable compliance checking and child-safety screening of product categories heavily consumed by children (confectionery, dairy, snacks, beverages).
- **Reusable beyond Vietnamese.** Normalized `[0,1000]` coordinates and a documented schema make it a drop-in benchmark for multilingual, layout-aware document models.

---

## Documentation

Full documentation is in [`docs/`](docs/):

| Document | Description |
|---|---|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/motivation.md](docs/motivation.md) | Background, regulatory context, and the gap the dataset fills |
| [docs/dataset-overview.md](docs/dataset-overview.md) | Dataset scope, format, per-image record, and metadata |
| [docs/dataset-statistics.md](docs/dataset-statistics.md) | Split sizes, token/entity/relation distributions, category breakdown |
| [docs/data-dictionary.md](docs/data-dictionary.md) | Every file and every field, with values and types |
| [docs/benchmark-tasks.md](docs/benchmark-tasks.md) | Intended uses: the tasks the data supports (SER / RE / KIE) |
| [docs/task3-kie-record.md](docs/task3-kie-record.md) | KIE record schema, ground-truth construction, and QC |
| [docs/annotation/](docs/annotation/) | Annotation guidelines (EN/VI), entity & relation schemas, pipeline |

---

## File Structure

```text
ViFoodLabel/
├── data/                       # (released separately on Mendeley Data)
│   ├── images/                 # Original product label images (JPEG)
│   ├── label_studio/           # Anonymized Label Studio annotation export
│   └── processed/
│       ├── dataset/            # Per-image KIE records (meta + structured fields)
│       ├── dataset_meta.json   # Per-image descriptive metadata
│       ├── splits.json         # Frozen train/dev/test ID lists (80/10/10)
│       └── statistics.json     # Computed dataset statistics
├── docs/                       # Dataset specifications, schemas, statistics
├── scripts/
│   ├── preprocessing/          # Label Studio → token-level conversion & validation
│   └── dataset/                # Splitting, statistics, anonymization, packaging
├── LICENSE                     # MIT (code)
└── requirements.txt
```

---

## Reproducing the Derived Data

```bash
pip install -r requirements.txt

# 1. Label Studio export -> token-level BIO + bounding boxes (coords -> [0,1000])
python scripts/preprocessing/convert_ls_to_ner.py \
    --input data/label_studio/data.json --output data/processed/ --autofix-bio

# 2. Validate BIO consistency, bounding boxes, and relations
python scripts/preprocessing/check_data.py \
    --input data/processed/train.json data/processed/val.json

# 3. Assemble per-image KIE records + descriptive metadata
python scripts/build_task3_gt.py
python scripts/build_dataset_meta.py

# 4. Freeze the stratified 80/10/10 split and compute statistics
python scripts/dataset/split_dataset.py
python scripts/dataset/compute_statistics.py
```

---

## License

- **Code** in `src/` and `scripts/` is licensed under [MIT](LICENSE).
- The **ViFoodLabel dataset** (images + annotations) is licensed under [CC BY-NC 4.0](docs/DATA_LICENSE.md) — non-commercial use, attribution required.

**Citation**: A BibTeX entry will be added here once the data article is published.

---

_Released for Vietnamese NLP and food-safety research._
