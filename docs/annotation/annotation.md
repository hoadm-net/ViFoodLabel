# ANNOTATION GUIDE — ViFoodLabel

Read this guide in full before you start annotating.

---

## 1. Core Principles

### BIO Tagging

All entities use the **BIO** tagging scheme:

| Tag | Meaning | When to use |
|---|---|---|
| `B-<TYPE>` | Beginning | The **first** token of an entity |
| `I-<TYPE>` | Inside | Every **subsequent** token of the same entity |
| `O` | Outside | Any token that does **not** belong to an entity |

**Example**:

```
Hạnh       → B-INGREDIENT
nhân,      → I-INGREDIENT
điều,      → B-INGREDIENT
óc         → B-INGREDIENT
chó,       → I-INGREDIENT
NSX:       → O
18/11/2025 → B-MFG_DATE
```

> **Hard rule**: Every entity span **must start** with `B-`. Never use `I-` without a preceding `B-` of the same type.

---

### One word = one bounding box

- Draw **one bounding box** tightly around **one word** (token)
- Enter the **transcription** — the exact text inside the box, preserving case and punctuation
- Assign **one** BIO label to that bbox
- **Do not skip any token** — including punctuation, numbers, special characters, and barcode text

### Handling unclear tokens

Some tokens on real product labels may be blurry, glare-affected, rotated, or partially obscured. Apply these rules:

| Case | Action |
|---|---|
| Token is an **entity** (any type), even if hard to read | **Must annotate** — never skip |
| Token is clearly `O` but **unreadable** (decorative, barcode, etc.) | May skip |
| Token is **uncertain** — not sure if `O` or entity | Must annotate; enter `???` as transcription |

> When in doubt, always annotate. Skipping a potential entity is a worse mistake than keeping a noisy `O` token.

---

### Working order

```
Step 1: Scan the image top-to-bottom, left-to-right
Step 2: Draw bbox + assign label for each token in that order
Step 3: After ALL tokens are labeled → draw HAS_VALUE relations
```

> **Important**: Work line by line. If an entity is interrupted by an `O` token in the middle, the entity has ended — the next token of the same type must start a new `B-` span.

---

### Section headers and field prefixes → always `O`

The following tokens are **never** assigned an entity label:

| Group | Examples |
|---|---|
| Section headers | "Thành phần:", "INGREDIENT:", "Thông tin dinh dưỡng", "Nutrition Facts" |
| Field prefixes | "NSX:", "HSD:", "Xuất xứ:", "Xứ:", "Khối lượng tịnh:", "Net weight:" |
| Manufacturer prefixes | "Sản phẩm của:", "Sản xuất tại:", "Product of:", "Manufactured by:" |
| Explanatory text | "Phần trăm giá trị dinh dưỡng hàng ngày dựa trên..." |
| Usage instructions | "Uống ngay sau khi mở", "Cách dùng:", "Dùng trực tiếp" |
| Barcode, website, phone | "8936036027983", "www.kash.vn", "+84 888 670 588" |
| Certifications, standards | "Số TCCS 47:2019/THM", "Theo bản quyền thương hiệu..." |

---

## 2. Label Reference

---

### `PRODUCT_NAME` — Product name

The full commercial name: brand + product name + flavor + variant.

**Examples**:

| Token | Label |
|---|---|
| KẸO | `B-PRODUCT_NAME` |
| DẺO | `I-PRODUCT_NAME` |
| BOOM | `I-PRODUCT_NAME` |
| VỊ | `I-PRODUCT_NAME` |
| NHO | `I-PRODUCT_NAME` |

| Token | Label |
|---|---|
| Trail | `B-PRODUCT_NAME` |
| Mix | `I-PRODUCT_NAME` |
| Việt | `I-PRODUCT_NAME` |
| quất | `I-PRODUCT_NAME` |

> Taglines, slogans, and marketing copy are not part of the product name → `O`

---

### `INGREDIENT` — Ingredient

Each distinct ingredient in the list = **one entity** (starting with `B-`). This means the ingredient list is **not** one long span — every new ingredient restarts with `B-`.

> **Design rationale**: The goal is to extract individual ingredients, not just locate the ingredient field. Labeling each item separately allows the model to output a structured list like `["Hạnh nhân", "Điều", "Óc chó"]`.

**Example — simple list** (`mạch nha, lúa mì, đường thốt nốt`):

| Token | Label |
|---|---|
| mạch | `B-INGREDIENT` |
| nha, | `I-INGREDIENT` |
| lúa | `B-INGREDIENT` |
| mì, | `I-INGREDIENT` |
| đường | `B-INGREDIENT` |
| thốt | `I-INGREDIENT` |
| nốt | `I-INGREDIENT` |

**Example — with group label and percentage** (`Hỗn hợp hạt (76,3%): Hạnh nhân lát, điều, óc chó,`):

| Token | Label |
|---|---|
| Hỗn | `O` |
| hợp | `O` |
| hạt | `O` |
| (76,3%) | `O` |
| Hạnh | `B-INGREDIENT` |
| nhân | `I-INGREDIENT` |
| lát, | `I-INGREDIENT` |
| điều, | `B-INGREDIENT` |
| óc | `B-INGREDIENT` |
| chó, | `I-INGREDIENT` |

> "Hỗn hợp hạt" is a group name, not a specific ingredient. "(76,3%)" is the group's percentage — both are `O`.
> Section headers "Thành phần:", "INGREDIENT:" → `O`.

---

### `ADDITIVE` — Food additive

The full span of **functional name + Codex number(s)** = one continuous entity.

**Examples**:

| Token | Label |
|---|---|
| CHẤT | `B-ADDITIVE` |
| LÀM | `I-ADDITIVE` |
| DÀY | `I-ADDITIVE` |
| (1200, | `I-ADDITIVE` |
| 1442, | `I-ADDITIVE` |
| 440), | `I-ADDITIVE` |

| Token | Label |
|---|---|
| CHẤT | `B-ADDITIVE` |
| ĐIỀU | `I-ADDITIVE` |
| ĐỘ | `I-ADDITIVE` |
| ACID | `I-ADDITIVE` |
| (330, | `I-ADDITIVE` |
| 334), | `I-ADDITIVE` |

> Each different additive type → start a new `B-ADDITIVE` span.

---

### `NUTRITION_NAME` — Nutrition attribute name

Label only the **name** of the nutrient. Do not include numbers or units.

**Monolingual example**:

| Token | Label |
|---|---|
| Năng | `B-NUTRITION_NAME` |
| lượng | `I-NUTRITION_NAME` |

| Token | Label |
|---|---|
| Natri | `B-NUTRITION_NAME` |

**Bilingual VI + EN example**:

| Token | Label |
|---|---|
| Năng | `B-NUTRITION_NAME` |
| lượng / | `I-NUTRITION_NAME` |
| Energy | `I-NUTRITION_NAME` |

| Token | Label |
|---|---|
| Chất | `B-NUTRITION_NAME` |
| béo/ | `I-NUTRITION_NAME` |
| Fat | `I-NUTRITION_NAME` |

> "Thông tin dinh dưỡng", "Nutrition Facts", "Giá Trị Dinh Dưỡng Trung Bình Trong 100g", "Per serving" → `O`

---

### `NUTRITION_VALUE` — Nutrition value

Label the **number + unit**. A daily value percentage on the same row → `I-NUTRITION_VALUE`.

**If number and unit are in the same bbox**:

| Token | Label |
|---|---|
| 90 kcal | `B-NUTRITION_VALUE` |
| 60,1 kcal | `B-NUTRITION_VALUE` |

**If number and unit are in separate bboxes**:

| Token | Label |
|---|---|
| 20 | `B-NUTRITION_VALUE` |
| g | `I-NUTRITION_VALUE` |
| 7% | `I-NUTRITION_VALUE` |

> A daily value percentage on the same row → `I-NUTRITION_VALUE`.  
> "Khẩu phần:", "Per serving:", "trong 25g" → `O`.

---

### `MANUFACTURER` — Manufacturer

Company name and address. If they appear consecutively without interruption → one entity.

**Examples**:

| Token | Label |
|---|---|
| CÔNG | `B-MANUFACTURER` |
| TY | `I-MANUFACTURER` |
| CỔ | `I-MANUFACTURER` |
| PHẦN | `I-MANUFACTURER` |
| SỮA | `I-MANUFACTURER` |

| Token | Label |
|---|---|
| TH MILK | `B-MANUFACTURER` |
| JOINT | `I-MANUFACTURER` |
| STOCK | `I-MANUFACTURER` |

> "Sản phẩm của:", "Sản xuất tại:", "Product of:", "Manufactured by:" → `O`

---

### `ORIGIN` — Country of origin

Label only the **country or region name**. Vietnamese and English versions → **two separate entities**.

**Examples**:

| Token | Label |
|---|---|
| Việt | `B-ORIGIN` |
| Nam | `I-ORIGIN` |

| Token | Label |
|---|---|
| Vietnam. | `B-ORIGIN` |

> "Xuất xứ:", "Xứ:", "Made in", "Sản xuất tại", "Xuất" → `O`

---

### `NET_WEIGHT` — Net weight

Label only the **number + unit**.

| Token | Label |
|---|---|
| 550 | `B-NET_WEIGHT` |
| g | `I-NET_WEIGHT` |

Or if they are in one bbox:

| Token | Label |
|---|---|
| 550 g | `B-NET_WEIGHT` |

> "Khối lượng tịnh:", "Net weight:", "Lượng tịnh:" → `O`

---

### `MFG_DATE` — Manufacturing date

Label only the **date value**. A lot code or time printed on the same line → `I-MFG_DATE`.

| Token | Label |
|---|---|
| 31.12.25A1 | `B-MFG_DATE` |
| 13:16 | `I-MFG_DATE` |

| Token | Label |
|---|---|
| 18/11/2025 | `B-MFG_DATE` |

> "NSX:", "Ngày sản xuất:", "Production date:", "Sản xuất:" → `O`

---

### `EXPIRY_DATE` — Expiry date

Label only the **date value**.

| Token | Label |
|---|---|
| 18/11/2026 | `B-EXPIRY_DATE` |

> "HSD:", "Hạn sử dụng:", "Best before:", "Expiry date:", "Use by:" → `O`

---

### `WARNING` — Warning / Allergen Declaration / Storage instructions

Safety warnings, allergen declarations, and storage instructions. Start from the **first word of the content**.

**Safety warning example**:

| Token | Label |
|---|---|
| Không | `B-WARNING` |
| dùng | `I-WARNING` |
| sản | `I-WARNING` |
| phẩm | `I-WARNING` |
| đã | `I-WARNING` |
| hết | `I-WARNING` |
| hạn | `I-WARNING` |
| sử | `I-WARNING` |
| dụng | `I-WARNING` |

**Allergen declaration example**:

| Token | Label |
|---|---|
| Sản | `B-WARNING` |
| phẩm | `I-WARNING` |
| có | `I-WARNING` |
| chứa | `I-WARNING` |
| sữa | `I-WARNING` |

> If VI and EN allergen warnings appear consecutively → merge into one `WARNING` entity span.  
> If the allergen warning appears at two separate locations → two separate `WARNING` entities.  
> "Thông tin cảnh báo:", "Bảo quản:" → `O`  
> General usage tips ("Uống lạnh ngon hơn", "Cách dùng:") → `O`

---

### `O` — Not an entity

Assign `O` to every token that does not belong to any of the 11 entity types, including:

- Section headers and field prefixes (see table in Section 1)
- Marketing and decorative text
- General usage instructions that are not safety warnings
- Phone numbers, websites, barcodes, QR codes
- Daily value percentage explanatory text

---

## 3. HAS_VALUE Relation

Draw relations only **after** all bounding boxes have been labeled.

### How to draw

For each row in the nutrition facts table:

```
[B-NUTRITION_NAME] ──HAS_VALUE──► [B-NUTRITION_VALUE]
```

**Always draw from the `B-` token** (first token) of NUTRITION_NAME to the `B-` token of NUTRITION_VALUE.

### Example

| From | | To |
|---|---|---|
| `B-NUTRITION_NAME` "Năng" | → HAS_VALUE → | `B-NUTRITION_VALUE` "90 kcal" |
| `B-NUTRITION_NAME` "Chất" | → HAS_VALUE → | `B-NUTRITION_VALUE` "0 g" |
| `B-NUTRITION_NAME` "Natri" | → HAS_VALUE → | `B-NUTRITION_VALUE` "0 mg" |

> Each row in the nutrition table = exactly one relation.  
> The table header row and "Khẩu phần / Per serving" row do not get a relation.

---

## 4. Bilingual Labels (Vietnamese + English)

| Entity | Rule |
|---|---|
| `NUTRITION_NAME` | Merge VI + EN into **one entity span** |
| `WARNING` *(allergen)* | If VI and EN allergen warnings appear consecutively → merge into **one entity span** |
| `INGREDIENT` | If VI and EN are two separate lists → **two separate entities** per item |
| `ORIGIN` | VI and EN → **two separate entities** |
| `MANUFACTURER` | Merge if consecutive; split if interrupted by other content |

---

## 5. Pre-submission Checklist

Before marking an image as complete, verify:

- [ ] Every token has exactly one bbox + one transcription + one label
- [ ] No `I-X` tag appears without a preceding `B-X` of the same type immediately before it
- [ ] Every HAS_VALUE relation goes from `B-NUTRITION_NAME` → `B-NUTRITION_VALUE`
- [ ] Number of relations = number of rows in the nutrition facts table
- [ ] All section headers and field prefixes are labeled `O`
