# Technical Runtime Notes

This page keeps only implementation-level notes discovered while running ViFoodLabel.

## 1) PhoBERT Sequence Length Constraint

Model family: `vinai/phobert-base`.

- PhoBERT max position embeddings are effectively bounded around 258.
- After reserving special tokens (`<s>`, `</s>`), the practical content length should be capped at 256 subword tokens.
- In training/inference preprocessing, truncate to 256 for safety.

Practical rule:
- Use `max_length=256` with truncation enabled for PhoBERT pipelines.
- Do not assume 512-token behavior like some BERT/XLM-R settings.

## 2) PhoBERT Tokenizer Detail

- PhoBERT commonly relies on slow tokenizer behavior in this project flow.
- Word-to-subword label alignment should be explicitly handled in dataset code.
- Continuation subwords should use ignored label index (`-100`) when training token classification.

## 3) Pillow 10 Compatibility for VietOCR

Observed issue:
- `PIL.Image.ANTIALIAS` was removed in newer Pillow versions.
- VietOCR or related image-resize paths may still reference `ANTIALIAS`.

Symptom:
- OCR path can fail or fall back to slower behavior.

Applied compatibility patch (in OCR engine init path):

```python
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
```

Result:
- Restores expected resize behavior and avoids legacy constant errors in runtime.

## 4) Runtime Profiling Baseline

Current pipeline bottleneck pattern:
- OCR is the dominant phase.
- NER is secondary.
- Relation extraction and JSON formatting are negligible.

When profiling performance regressions, verify OCR dependency compatibility first (Pillow/VietOCR/doctr stack).
