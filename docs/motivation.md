# Motivation

Food labeling regulations in Vietnam mandate that product labels include critical consumer information: ingredients, additives, allergens, nutritional values, manufacturer details, and expiry dates.

**Key regulations:**
- **TCVN 7087:2013** — Labeling of prepackaged foods
- **Decree 43/2017/NĐ-CP** — Labeling of goods

## The Problem

Despite regulatory requirements, automated verification of food label compliance at scale remains an open challenge:

- **Manual verification** is labor-intensive and error-prone for inspectors and food safety agencies.
- **Retailers and e-commerce platforms** need to automatically extract and index structured product information from label images.
- **Consumers** increasingly rely on digital tools to check allergens, nutritional content, and product origin.

### Why Children's Food Products

Children are a disproportionately exposed consumer segment: confectionery, flavored dairy, snacks, and beverages — categories already well represented in ViFoodLabel (see [dataset-overview.md](dataset-overview.md)) — make up a large share of what Vietnamese children consume directly, often without an adult reading the label first. Accurate, automated extraction of allergens, additives, and age-appropriate warnings (e.g., *"không dùng cho trẻ dưới 3 tuổi"*) from these labels has direct child-safety value, beyond the general compliance-checking motivation above. This framing motivates the dataset's narrative without changing its collection scope, which remains general-purpose across all food categories.

## The Gap

Existing Key Information Extraction (KIE) datasets do not address this need:

| Dataset | Language | Domain | Layout-aware |
|---|---|---|---|
| FUNSD | English | Noisy forms | ✅ |
| CORD | English | Receipts | ✅ |
| SROIE | English | Receipts | ✅ |
| DocVQA | English | Documents | ✅ |
| **ViFoodLabel** | **Vietnamese** | **Food labels** | ✅ |

The closest prior work is a bilingual (English-Arabic) food-label nutrition extraction study (294 images, GPT-4V/4o/Gemini, Jaccard-Index evaluation) — it is neither Vietnamese nor layout-annotated (no bounding boxes or BIO spans), and is an order of magnitude smaller than ViFoodLabel's 550 images. More broadly, recent multimodal-LLM benchmarks for document KIE (e.g., UniKIE-Bench, 15 SOTA LMMs) report "substantial performance degradation under diverse schema definitions, long-tail key fields, and complex layouts" — exactly the conditions food labels present (dense multi-column ingredient/nutrition layout, long-tail additive codes). This suggests off-the-shelf MLLMs are not yet a substitute for a purpose-built, layout-annotated benchmark in this domain.

Vietnamese food product labels pose unique challenges:
- **Diacritics and tone marks** (e.g., *đường*, *nước cốt nho*) that OCR systems frequently misread
- **Dense, non-linear layout** with multi-column ingredient lists, stacked nutrition tables, and rotated text
- **Non-standard typography** — mixed fonts, sizes, colors, and background textures
- **Domain vocabulary** — Codex Alimentarius additive codes (e.g., 1422, 472c), Vietnamese regulatory terminology

## Goal

ViFoodLabel aims to be the **first large-scale, layout-aware KIE benchmark for Vietnamese food product labels**, enabling:
- Automated food label compliance checking
- Structured product data extraction for e-commerce
- Child-safety-oriented label screening (allergens, additives, age warnings) for product categories heavily consumed by children
- Foundation for multilingual document AI research in Southeast Asian languages
