# Documentation Overview

This documentation describes the ViFoodLabel dataset: what it contains, how it
was built, and how every file and field is structured. It is written for a
**data article** — it covers the data, not model training or deployment.

## 1) Dataset Description

| File | Description |
|---|---|
| [motivation.md](motivation.md) | Background, regulatory context, and the gap the dataset fills |
| [dataset-overview.md](dataset-overview.md) | Scope, format, per-image record, and metadata |
| [dataset-statistics.md](dataset-statistics.md) | Split sizes, token/entity/relation distributions, category breakdown |
| [data-dictionary.md](data-dictionary.md) | Every file and every field, with values and types |
| [benchmark-tasks.md](benchmark-tasks.md) | Intended uses: the tasks the data supports (SER / RE / KIE) |
| [task3-kie-record.md](task3-kie-record.md) | KIE record schema, ground-truth construction, and QC |

## 2) Annotation (Methods)

How labels were produced and quality-checked.

| Folder/File | Description |
|---|---|
| [annotation/README.md](annotation/README.md) | Index for annotation docs |
| [annotation/annotation.md](annotation/annotation.md) | Annotation guide (English) |
| [annotation/annotation-vi.md](annotation/annotation-vi.md) | Annotation guide (Vietnamese) |
| [annotation/entity-schema.md](annotation/entity-schema.md) | Entity schema and BIO rules |
| [annotation/relation-schema.md](annotation/relation-schema.md) | Relation schema |
| [annotation/annotation-pipeline.md](annotation/annotation-pipeline.md) | Annotation workflow |

## 3) License

| File | Description |
|---|---|
| [DATA_LICENSE.md](DATA_LICENSE.md) | CC BY-NC 4.0 text (dataset license) |

## Scope

This repository covers the **dataset** (images + annotations + derived records)
and the tooling to build, validate, and describe it. It does not include
baseline models, benchmark experiments, or a serving layer.
