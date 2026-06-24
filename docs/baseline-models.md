# Baseline Models and Proposed Model

Since the primary goal of this research is to provide a high-quality, diverse Vietnamese food label dataset, the baseline models mainly serve to (1) validate dataset difficulty and (2) benchmark across different modalities (text-only vs. layout-aware vs. multimodal vs. zero-shot MLLM).

> **Update 2026-06-21** after literature review: baselines are organized into 4 tiers (A–D) to be competitive for a Q1/Scopus 2026 target journal. References are listed in Section 6.

## Tier A — Supervised Text/Layout Baselines

1. **PhoBERT (Text-only)**: Vietnamese-specialized language model. Input is text-only (assuming perfect OCR / ground-truth transcriptions).
2. **XLM-RoBERTa (Text-only)**: Multilingual language model, handles Vietnamese well.
3. **LayoutLMv3 — No Visual (Text + Layout)**: Microsoft's multimodal model with the visual segment disabled. Uses Vietnamese text + bounding boxes (2D spatial coordinates) to learn label layout structure.
4. **LayoutLMv3 — Visual (Text + Layout + Image)**: Complete multimodal version with text, bounding boxes, and cropped image features. Allows evaluation of whether label background/visual cues provide signal or noise.

## Tier B — Next-Generation Layout-Aware Models

5. **LiLT (Language-Independent Layout Transformer)**: Designed to pair with any RoBERTa-style backbone. Pairs naturally with PhoBERT for Vietnamese without requiring a new layout pretraining from scratch.
6. **BROS**: Pretrained on text + layout without requiring images. Complements Tier A for layout-only evaluation.

## Tier C — Multimodal LLM Zero-Shot (No Fine-tuning)

According to **UniKIE-Bench** (2026, benchmarking 15 SOTA MLLMs on document KIE), current MLLMs still suffer *"substantial performance degradation under diverse schema definitions, long-tail key fields, and complex layouts"* — exactly the conditions food labels present (multi-row nutrition tables, long-tail additive codes, dense layouts). Tier C is mandatory to demonstrate that the dataset remains challenging for SOTA 2026 — not solvable by a single prompt.

7. **One closed-source MLLM** (GPT-5) — zero-shot prompt evaluation using entity schema.
8. **One open-weight MLLM** (Qwen3-VL) — required for reproducibility (Q1 reviewers typically mandate this, not just closed APIs).

**Related observation**: A bilingual (English–Arabic) nutrition-extraction study from food labels using GPT-4V/4o/Gemini showed strong English performance but significant degradation on non-Latin-script languages before post-processing — a similar pattern is expected for Vietnamese diacritics and will be validated experimentally here.

## Tier D — Proposed Contribution (Vietnamese-Specific Model)

9. **Model-based Relation Extraction**: A lightweight typed-link predictor (`src/relation_model.py`, inspired by GLiNER-Relex / Parallel Pointer Networks) that predicts `HAS_VALUE` link probability from entity-pair embeddings, replacing the geometric heuristic baseline in `src/relation_extractor.py`. This is the paper's methodological contribution — measuring Relation-F1 delta between the two approaches.

> **Decision (2026-06-24)**: dropped the generative seq2seq (Donut) baseline for Task 3. End-to-End KIE is evaluated via the Tier C zero-shot MLLMs (image → JSON directly) instead — matches the actual paper outline (Section 5.2 lists only text-only / layout-aware / zero-shot MLLM baseline categories) and avoids committing to an extra heavyweight training pipeline while Tier A–D are still scaffolds. Donut/UDOP/OmniParser remain cited in Related Work as motivation for the task design only, not as a baseline to train.

## Module Reference for Task 3 Reproduction (Benchmark Use Only)

The modules below exist to **locally reproduce and evaluate** Task 3 (End-to-End KIE) — this repository provides **no inference API or service**, only dataset + baselines + proposed model.

**Step 1: Text Detection**
- **Model**: `doctr` (in `src/ocr_engine.py`).
- **Purpose**: Detect text regions on product label images, including complex/small text, and return bounding box coordinates.

**Step 2: Text Recognition (OCR)**
- **Model**: VietOCR (Transformer/CNN-based for Vietnamese, in `src/ocr_engine.py`).
- **Purpose**: Crop detected regions and recognize text strings. VietOCR handles Vietnamese diacritics better than generic OCR, especially on low-quality label fonts.

**Step 3: Information Extraction (BIO NER Tagging)**
- **Model**: Tier A/B baselines listed above (PhoBERT, XLM-R, LayoutLMv3, LiLT, BROS), wrapped by `src/ner_engine.py`.
- **Purpose**: Classify each token (B-PRODUCT_NAME, I-NUTRITION_VALUE, etc.) given text and spatial coordinates, aggregating tokens into entity spans.

**Note**: Results from Tier A–D baselines help the research community understand the true difficulty of Information Extraction (with and without ground-truth boxes/text), establishing a measurement baseline when chaining all four modules together for Task 3 end-to-end evaluation.

## Related Work

- UniKIE-Bench: Benchmarking Large Multimodal Models for Key Information Extraction in Visual Documents — https://arxiv.org/pdf/2602.07038
- A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends — https://arxiv.org/html/2507.09861v2
- Extract Nutritional Information from Bilingual Food Labels Using Large Language Models — https://pmc.ncbi.nlm.nih.gov/articles/PMC12387780/
- LiGT: Layout-infused Generative Transformer for Visual Question Answering on Vietnamese Receipts — https://arxiv.org/html/2502.19202
- GLiNER-Relex: A Unified Framework for Joint Named Entity Recognition and Relation Extraction — https://arxiv.org/html/2605.10108v1
- Deep Learning based Key Information Extraction from Business Documents: Systematic Literature Review — https://arxiv.org/pdf/2408.06345
- VAREX: A Benchmark for Multi-Modal Structured Extraction from Documents — https://arxiv.org/pdf/2603.15118
- MC-OCR Challenge 2021: End-to-end system to extract key information from Vietnamese Receipts — https://www.researchgate.net/publication/357234375
