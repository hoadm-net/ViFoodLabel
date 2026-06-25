# Documentation Overview

This documentation is organized into focused sections to avoid duplicated content and make navigation easier.

## 1) Dataset Description

Use this section to understand what ViFoodLabel contains and how it is evaluated.

| File | Description |
|---|---|
| [motivation.md](motivation.md) | Why ViFoodLabel exists: research and practical motivation |
| [dataset-overview.md](dataset-overview.md) | Dataset scope, format, per-image record, and metadata |
| [dataset-statistics.md](dataset-statistics.md) | Split sizes and distribution summaries |
| [benchmark-tasks.md](benchmark-tasks.md) | SER/RE/End-to-End task definitions |
| [task3-kie-record.md](task3-kie-record.md) | Task 3 record schema, ground-truth construction, and evaluation protocol |

## 2) Annotation

Use this section to annotate and quality-check labels consistently.

| Folder/File | Description |
|---|---|
| [annotation/README.md](annotation/README.md) | Index for annotation docs |
| [annotation/annotation.md](annotation/annotation.md) | Annotation guide (English) |
| [annotation/annotation-vi.md](annotation/annotation-vi.md) | Annotation guide (Vietnamese) |
| [annotation/entity-schema.md](annotation/entity-schema.md) | Entity schema and BIO rules |
| [annotation/relation-schema.md](annotation/relation-schema.md) | Relation schema |
| [annotation/annotation-pipeline.md](annotation/annotation-pipeline.md) | Annotation workflow |

## 3) Baselines and Proposed Model

| File | Description |
|---|---|
| [baseline-models.md](baseline-models.md) | Baseline tiers (A–D) and result summaries, incl. the proposed relation-extraction model |

## 4) Technical Notes

| File | Description |
|---|---|
| [notes/technical-runtime-notes.md](notes/technical-runtime-notes.md) | Runtime implementation notes (PhoBERT/Pillow/OCR profiling) |

## Out of Scope

This repository covers the **dataset** (images + Label Studio annotations),
**baseline models**, and the **proposed relation-extraction model**. It does
not include a deployment/serving layer (no API, no packaged inference
service) — see [benchmark-tasks.md](benchmark-tasks.md) for how models are
expected to be run and evaluated instead.
