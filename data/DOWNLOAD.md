# Download ViFoodLabel Images

The dataset images and thumbnails are hosted on **Google Drive** to avoid bloating the GitHub repository.

## Quick Start

```bash
# From repo root, download and extract images
python scripts/download_dataset.py

# This will:
# 1. Download the archive from Google Drive
# 2. Extract to data/raw/ and data/thumbnail/
# 3. Verify integrity
```

## Manual Download

If the script doesn't work:

1. **Download** the archive from Google Drive:
   - Link: https://drive.google.com/file/d/1MjrffVfmH4gfyftjV54IT0fpq4RDWPqL/view?usp=sharing
   - Click "Download" (top-right)

2. **Extract** the archive:
   ```bash
   unzip vifoodlabel_images.zip
   # Creates data/raw/*.jpeg and data/thumbnail/*.jpeg
   ```

3. **Verify**:
   ```bash
   ls data/raw/ | wc -l  # Should show 550 (image count)
   ```

## Next: Preprocessing Pipeline

Once images are downloaded:

```bash
# 1. Generate Label Studio pre-annotations
python scripts/preprocessing/label_studio_preann.py \
    --folder data/raw \
    --output data/label_studio/tasks.json

# 2. Upload tasks.json to Label Studio
# 3. Annotate / correct predictions
# 4. Export and convert to NER format
python scripts/preprocessing/convert_ls_to_ner.py \
    --input data/label_studio/data.json \
    --output data/processed/
```

See [scripts/preprocessing/README.md](../scripts/preprocessing/README.md) for full workflow.

## Dataset Contents

- **data/raw/** — Original JPEG images (550 total, sequentially numbered: 0001.jpeg → 0550.jpeg)
- **data/thumbnail/** — Preview thumbnails (250px max, for quick QA)

## Storage Notes

- Total size: ~400–500 MB (raw) + ~50 MB (thumbnail)
- Hosted on Google Drive to keep repo lightweight
- Images are **NOT** committed to Git (see `.gitignore`)
- Label Studio exports are committed (see `data/label_studio/.gitkeep`)

## Troubleshooting

**Q: "gdown is not installed"**
```bash
pip install gdown
```

**Q: "BadZipFile" error**
- Delete the partial download: `rm vifoodlabel_images.zip`
- Re-run: `python scripts/download_dataset.py`
- Check if the Google Drive link is still accessible

**Q: Slow download?**
- Google Drive throttles large downloads. Be patient or try:
  ```bash
  python scripts/download_dataset.py --quiet
  ```

**Q: Want to upload a new version?**
- Update `GDRIVE_FILE_ID` in `scripts/download_dataset.py`
- Update this doc
- Announce to team
