# Image Preprocessing Workflow

This directory contains tools for preparing raw product label images for annotation in Label Studio.

## Workflow Overview

```
iPhone Photos (HEIC)
    ↓ [prepare_images.py]
data/raw/ (JPEG) + data/thumbnail/ (250px preview)
    ↓ [label_studio_preann.py]
tasks.json (OCR pre-annotations)
    ↓ [Upload to Label Studio]
Annotate & correct predictions
    ↓ [Export + convert_ls_to_ner.py]
HuggingFace NER format (train/val/test splits)
```

---

## Step 1: Prepare Images

Convert iPhone HEIC files to JPEG and generate thumbnails.

```bash
python prepare_images.py \
    --input ~/Desktop/airdrop/heic \
    --output data/raw \
    --thumbnail data/thumbnail \
    --start-index 200
```

**Options:**
- `--input`: Directory containing HEIC files from airdrop
- `--output`: Save location for JPEG files (default: `data/raw`)
- `--thumbnail`: Thumbnail output directory (default: `data/thumbnail`)
- `--start-index`: Starting number for sequential naming (default: 1)
- `--no-shuffle`: Preserve file order (default: shuffle)
- `--no-thumbnail`: Skip thumbnail generation

**Output:**
- `data/raw/0200.jpeg`, `data/raw/0201.jpeg`, etc. (sequential, zero-padded)
- `data/thumbnail/0200.jpeg`, etc. (250px max dimension)

---

## Step 2: Generate Label Studio Pre-annotations

Run OCR to detect text and create pre-filled annotations for Label Studio.

```bash
python label_studio_preann.py \
    --folder data/raw \
    --start 200 \
    --end 250 \
    --output data/label_studio/tasks.json
```

**Options:**
- `--folder`: Image directory (default: `data/raw`)
- `--start`: First image index (default: 1)
- `--end`: Last image index (auto-detects if omitted)
- `--output`: Save tasks.json path (default: `tasks.json`)
- `--image-base-url`: Base URL the images are served from (default: `http://103.159.52.8/images`). Label Studio's `data.image` field must be a URL it can fetch, not a local filesystem path — this script builds it as `{base_url}/{filename}`.

**Process:**
1. Detects text regions using **doctr** (scene text detection)
2. Recognizes Vietnamese text using **VietOCR** (transformer-based OCR)
3. Splits multi-word regions by character-count ratio
4. Exports as `tasks.json` (Label Studio predictions format)
5. **Checkpoints after every image** — safe to resume if interrupted

**Requirements:**
```bash
pip install python-doctr vietocr torch torchvision
```

**Output:**
- `data/label_studio/tasks.json` — Import into Label Studio as predictions
- Annotators review and correct pre-filled text + bboxes
- Field names: `transcription` (text field), `image` (image)

---

## Step 3: Export and Convert

After annotation and QC in Label Studio:

1. **Export** from Label Studio as JSON
2. **Convert** to HuggingFace NER format:

```bash
python convert_ls_to_ner.py \
    --input data/label_studio/data.json \
    --output data/processed/ \
    --split 0.8 \
    --autofix-bio
```

See `convert_ls_to_ner.py --help` for all options.

---

## Resuming Interrupted Runs

Both scripts checkpoint results:
- `prepare_images.py`: Skips already-converted files
- `label_studio_preann.py`: Writes `tasks.json` after every image

So if a run is interrupted, just re-run the command — it will pick up where it left off (or skip completed work).

---

## Notes

- **Platform**: `prepare_images.py` works on Mac/Linux/Windows (PIL-based). Original `convert.sh`/`resize_thumb.sh` bash scripts (macOS-only) are deprecated.
- **GPU**: `label_studio_preann.py` auto-detects CUDA and will use GPU if available
- **Thread pools**: Both scripts cap OMP/BLAS threads to 4 to avoid resource exhaustion on multi-core systems
- **Quality**: VietOCR is optimized for Vietnamese diacritics; OCR quality depends on image sharpness and font consistency
