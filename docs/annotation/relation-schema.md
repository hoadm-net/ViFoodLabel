# Relation Schema

## Overview

ViFoodLabel defines one directed relation type between entity spans:

| Relation | Direction | Description |
|---|---|---|
| `HAS_VALUE` | `NUTRITION_NAME` → `NUTRITION_VALUE` | Links a nutrition attribute name to its corresponding quantitative value |

## Purpose

Nutrition fact tables on food labels present information in key-value pairs:

```
Năng lượng       90 kcal
Chất béo          0 g
  - Chất béo bão hòa  0 g
Carbohydrate     20 g
  - Đường         14 g
Chất đạm          2 g
Natri             0 mg
```

The `HAS_VALUE` relation captures this pairing explicitly, enabling models to reconstruct the full nutrition table as structured data rather than a flat token sequence.

## Annotation Rules

1. **From `B-NUTRITION_NAME`** — the relation arrow must originate from the **first token** (`B-`) of the nutrition name span.
2. **To `B-NUTRITION_VALUE`** — the relation arrow must point to the **first token** (`B-`) of the nutrition value span.
3. **One-to-one**: Each `NUTRITION_NAME` span links to exactly one `NUTRITION_VALUE` span.
4. **Hierarchical entries** (e.g., *Chất béo bão hòa* as a sub-row of *Chất béo*) are annotated the same way — each row gets its own `HAS_VALUE` relation regardless of indentation level.

## Example

Given the row: **Năng lượng** `90 kcal`

```
[B-NUTRITION_NAME: "Năng"] --HAS_VALUE--> [B-NUTRITION_VALUE: "90"]
      |                                         |
[I-NUTRITION_NAME: "lượng"]             [I-NUTRITION_VALUE: "kcal"]
```
