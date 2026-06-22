# Entity Schema

## BIO Tagging Scheme

All entities use the **BIO (Beginning-Inside-Outside)** convention:

- **`B-<TYPE>`** — first token of an entity span
- **`I-<TYPE>`** — any subsequent token within the same span
- **`O`** — token that does not belong to any entity

> **Rule**: Every entity span, regardless of length, **must** begin with a `B-` tag. An `I-` tag appearing without a preceding `B-` of the same type is an annotation error.

**Example** — multi-token entity:

| Token | Tag |
|---|---|
| CHẤT | `B-ADDITIVE` |
| LÀM | `I-ADDITIVE` |
| DÀY | `I-ADDITIVE` |
| (1422) | `I-ADDITIVE` |

## Entity Types

| Entity Type | Tag Prefix | Description | Example |
|---|---|---|---|
| Product Name | `PRODUCT_NAME` | Commercial name of the product | *KẸO DẺO BOOM VỊ NHO* |
| Ingredient | `INGREDIENT` | Raw ingredient in the ingredient list | *đường, gelatin, nước cốt nho* |
| Additive | `ADDITIVE` | Food additive, preservative, or colorant | *Chất làm dày (1422)* |
| Nutrition Name | `NUTRITION_NAME` | Name of a nutritional attribute | *Năng lượng* |
| Nutrition Value | `NUTRITION_VALUE` | Quantitative value of a nutritional attribute | *90 kcal* |
| Manufacturer | `MANUFACTURER` | Producer or packer name and address | *Công ty TNHH Thực phẩm Orion Vina* |
| Origin | `ORIGIN` | Country or region of manufacture | *Việt Nam* |
| Net Weight | `NET_WEIGHT` | Net weight or volume declaration | *52.5 g* |
| Manufacturing Date | `MFG_DATE` | Date of manufacture or batch/lot code | *31.12.25A1* |
| Expiry Date | `EXPIRY_DATE` | Best-before or use-by date | *6 tháng kể từ NSX* |
| Warning | `WARNING` | Safety warning, allergen declaration, or storage instructions | *Bảo quản nơi khô ráo, thoáng mát* |
| Other | `O` | Token not belonging to any entity class | — |


## Annotation Boundary Rules

- **INGREDIENT vs ADDITIVE**: Additives are explicitly labeled ingredients with a functional class name and/or E-number (e.g., *Chất tạo màu (102)*). Pure culinary ingredients (e.g., *đường trắng*) are tagged `INGREDIENT`.
- **NUTRITION_NAME vs NUTRITION_VALUE**: `NUTRITION_NAME` covers only the attribute name (e.g., *Natri*); `NUTRITION_VALUE` covers only its value and unit (e.g., *0 mg*). These two are linked via a `HAS_VALUE` relation — see [relation-schema.md](relation-schema.md).
- **MFG_DATE vs EXPIRY_DATE**: If only one date is present and its type is ambiguous from context, default to `MFG_DATE` and note the ambiguity in the annotation review log.
- **WARNING**: Includes safety warnings (e.g., *không dùng cho trẻ em dưới 3 tuổi*), allergen declarations (e.g., *Sản phẩm có chứa sữa*), and storage instructions (e.g., *bảo quản ở nhiệt độ dưới 25°C*).
