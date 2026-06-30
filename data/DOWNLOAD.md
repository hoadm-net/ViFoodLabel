# Download ViFoodLabel

The ViFoodLabel dataset (images + annotations + derived records) is published on
**Mendeley Data**. The download link / DOI will be added here on publication.

## Released contents

| Item | Description |
|---|---|
| `images/` | Original product label images (JPEG, sequentially numbered `0001.jpeg` …) |
| `label_studio/data.json` | Anonymized Label Studio annotation export |
| `processed/dataset/<id>.json` | Per-image KIE records (metadata + structured fields) |
| `processed/splits.json` | Frozen train/dev/test image-id lists |

See [../docs/data-dictionary.md](../docs/data-dictionary.md) for the full field
reference and [../docs/dataset-overview.md](../docs/dataset-overview.md) for the
record schema.

## Regenerating the derived data

If you have the Label Studio export and images, the processed files can be rebuilt:

```bash
# token-level NER + bounding boxes
python scripts/preprocessing/convert_ls_to_ner.py \
    --input data/label_studio/data.json --output data/processed/ --autofix-bio

# per-image KIE records, metadata, split, and statistics
python scripts/build_task3_gt.py
python scripts/build_dataset_meta.py
python scripts/dataset/split_dataset.py
python scripts/dataset/compute_statistics.py
```

## Notes

- Images and annotation exports are **not** committed to git (see `.gitignore`);
  only the `data/` folder structure is tracked via `.gitkeep`.
- The dataset is licensed CC BY-NC 4.0 (see [../docs/DATA_LICENSE.md](../docs/DATA_LICENSE.md)).
