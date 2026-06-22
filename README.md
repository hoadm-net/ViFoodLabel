# ViFoodLabel: A Vietnamese Food Product Label Dataset for Key Information Extraction

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Task-Key%20Information%20Extraction-blue"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Language-Vietnamese-red"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Domain-Food%20Product%20Labels-green"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Annotation-BIO%20NER%20%2B%20Layout-orange"/></a>
  <a href="docs/DATA_LICENSE.md"><img src="https://img.shields.io/badge/Dataset%20License-CC%20BY--NC%204.0-lightgrey"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code%20License-MIT-yellow"/></a>
</p>

---

## Abstract

We introduce **ViFoodLabel**, the first large-scale dataset for **Key Information Extraction (KIE)** from Vietnamese food product labels. Food product labels contain heterogeneous, multi-region text with complex layout, non-standard typography, and domain-specific vocabulary — making them a challenging testbed for visually-rich document understanding (VrDU) models — with particular relevance to food categories (confectionery, dairy, snacks, beverages) heavily consumed by children, where accurate allergen and additive extraction has direct child-safety value. ViFoodLabel provides pixel-accurate bounding box annotations, word-level BIO entity tags, and cross-entity relational links across **11 semantic entity types** including product name, ingredients, additives, nutritional information, manufacturer, origin, net weight, dates, and warnings. Each annotated token is paired with its transcription and spatial coordinates, enabling layout-aware model training. We benchmark a tiered set of baselines — from text-only and layout-aware transformers (PhoBERT, XLM-R, LayoutLMv3, LiLT, BROS) to zero-shot multimodal LLMs (closed and open-weight) — and propose a lightweight Vietnamese-specific relation-extraction module, reporting results under standard KIE evaluation protocols. ViFoodLabel aims to serve as a foundational resource for food safety information systems, regulatory compliance automation, and multilingual document AI research.

---

## Dataset & Baseline Status

**550 Vietnamese food product label images** have been collected. **134 images** are annotated and ready for baseline training. Implementation of preprocessing, training, and evaluation scripts is underway (see `src/` and `scripts/` for current scaffold status and `TODO` items).

---

## Repository Scope

This repository provides:

1. **The dataset**: raw images + Label Studio annotation exports + HuggingFace-ready processed splits.
2. **Baseline models**: training scripts for supervised text/layout baselines (PhoBERT, XLM-R, LayoutLMv3) and zero-shot MLLM evaluation.
3. **Our proposed model**: a learned relation-extraction module (`src/relation_model.py`) for the `HAS_VALUE` link, benchmarked against the geometric heuristic baseline.

There is **no deployment/serving layer** here (no API, no packaged inference
service) — `src/` modules other than the baselines/proposed model exist only
to reproduce the Task 3 (End-to-End KIE) benchmark locally for evaluation.

---

## Documentation

Full documentation is in the [`docs/`](docs/) folder:

| Document | Description |
|---|---|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/motivation.md](docs/motivation.md) | Research gaps and regulatory context |
| [docs/dataset-overview.md](docs/dataset-overview.md) | Dataset properties, format, product categories |
| [docs/annotation/entity-schema.md](docs/annotation/entity-schema.md) | 11 entity types, BIO rules |
| [docs/annotation/relation-schema.md](docs/annotation/relation-schema.md) | `HAS_VALUE` relation annotation guide |
| [docs/annotation/annotation-pipeline.md](docs/annotation/annotation-pipeline.md) | Annotation workflow, coordinate normalization |
| [docs/annotation/annotation.md](docs/annotation/annotation.md) | Annotation guideline (English) |
| [docs/annotation/annotation-vi.md](docs/annotation/annotation-vi.md) | Annotation guideline (Vietnamese) |
| [docs/notes/technical-runtime-notes.md](docs/notes/technical-runtime-notes.md) | Runtime implementation notes |
| [docs/dataset-statistics.md](docs/dataset-statistics.md) | Split sizes, entity distribution, IAA |
| [docs/benchmark-tasks.md](docs/benchmark-tasks.md) | SER, RE, and End-to-End KIE task definitions |
| [docs/baseline-models.md](docs/baseline-models.md) | Baseline tiers (A–D) and result tables |

---

## Quick Start

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Download Dataset Images

Images are hosted on Google Drive (to keep repo lightweight).

```bash
# Automatic download and extract
python scripts/download_dataset.py

# Or manual: see data/DOWNLOAD.md
```

### 3. Preprocessing Pipeline

```bash
# Step 1: Generate Label Studio pre-annotations (OCR)
python scripts/preprocessing/label_studio_preann.py \
    --folder data/raw \
    --output data/label_studio/tasks.json

# Step 2: (Upload tasks.json to Label Studio, annotate, export as data.json)

# Step 3: Convert Label Studio → HuggingFace NER format
python scripts/preprocessing/convert_ls_to_ner.py \
    --input  data/label_studio/data.json \
    --output data/processed/ \
    --split  0.8 \
    --autofix-bio

# Step 4: Validate data quality
python scripts/preprocessing/check_data.py \
    --input data/processed/train.json data/processed/val.json
```

### 4. Train Baselines (Once Data is Ready)

```bash
# PhoBERT (text-only)
python scripts/baselines/01_phobert.py \
    --train data/processed/train.json \
    --val data/processed/val.json \
    --epochs 20

# LayoutLMv3 (text + layout)
python scripts/baselines/03_layoutlmv3_no_visual.py \
    --train data/processed/train.json \
    --val data/processed/val.json \
    --images data/raw/ \
    --epochs 30
```

### 5. Evaluate

```bash
python scripts/evaluate.py \
    --predictions results/predictions.json \
    --ground-truth data/processed/val.json \
    --task ser  # or: re, kie
```

> **Note**: All scripts are currently **scaffolds** — CLI args and signatures match the spec in `docs/`, but core logic raises `NotImplementedError` pending implementation.

**Full workflow guide**: [scripts/preprocessing/README.md](scripts/preprocessing/README.md)  
**Dataset download**: [data/DOWNLOAD.md](data/DOWNLOAD.md)

---

## Key Locations

### Dataset Tooling (main objective: dataset delivery)

- Convert Label Studio annotations to model-ready format.
- Validate label quality and BIO consistency.
- Publish docs/specs for annotation and evaluation.

Key locations:
- [data/](data/)
- [scripts/preprocessing/](scripts/preprocessing/)
- [scripts/dataset/](scripts/dataset/)
- [docs/](docs/)

### Baselines & Proposed Model

- Tier A–B supervised baselines: text-only (PhoBERT, XLM-R) and layout-aware (LayoutLMv3).
- Tier C zero-shot MLLM evaluation (planned).
- Tier D proposed model: learned `HAS_VALUE` relation extraction.
- Evaluation: token/entity/relation/field F1.

Key locations:
- [scripts/baselines/](scripts/baselines/)
- [src/relation_extractor.py](src/relation_extractor.py) — heuristic baseline
- [src/relation_model.py](src/relation_model.py) — proposed model
- [src/metrics.py](src/metrics.py)
- [scripts/evaluate.py](scripts/evaluate.py)

---

## File Structure

```
ViFoodLabel/
├── data/
│   ├── label_studio/   # Label Studio JSON exports
│   ├── thumbnail/      # Resized preview images
│   ├── raw/            # Original product label images
│   └── processed/      # HuggingFace-ready splits (train/val/test)
├── docs/                       # Full documentation
│   ├── annotation/             # Annotation guides and schemas
│   ├── plan/                   # Planning and milestone docs
│   ├── notes/                  # Runtime implementation notes
│   └── DATA_LICENSE.md         # CC BY-NC 4.0 text (dataset license)
├── scripts/
│   ├── dataset/        # Dataset publishing utilities (HF upload)
│   ├── baselines/      # Baseline training scripts (Tier A)
│   ├── preprocessing/  # Data conversion/validation utilities
│   └── evaluate.py     # SER/RE/KIE evaluation entry point
├── src/                 # OCR/NER/RE/JSON modules (benchmark reproduction)
│   ├── relation_extractor.py  # heuristic HAS_VALUE baseline
│   ├── relation_model.py      # proposed learned relation model
│   └── metrics.py
├── LICENSE              # MIT (code)
├── requirements.txt
└── README.md
```

---

## License

- **Code** in `src/` and `scripts/` is licensed under [MIT](LICENSE).
- The **ViFoodLabel dataset** (images + annotations, once released) is licensed under [CC BY-NC 4.0](docs/DATA_LICENSE.md) — non-commercial use, attribution required.

**Citation**: Once the dataset paper is published, a BibTeX entry will be provided here.

---

<p align="center">Made with ❤️ for Vietnamese NLP and Food Safety Research</p>
