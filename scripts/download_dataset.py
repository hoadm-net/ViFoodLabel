#!/usr/bin/env python
"""Download ViFoodLabel images from Google Drive.

The dataset (raw images + thumbnails) is hosted on Google Drive.
This script downloads and extracts the archive to data/ folder.

Usage:
    python scripts/download_dataset.py

The script will:
1. Download the archive from Google Drive
2. Extract images to data/raw/ and data/thumbnail/
3. Verify integrity

Requirements:
    pip install gdown
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
import argparse

try:
    import gdown
except ImportError:
    print("Error: gdown is not installed.")
    print("Install it with: pip install gdown")
    sys.exit(1)


# Google Drive file ID for ViFoodLabel dataset
# Link: https://drive.google.com/file/d/1MjrffVfmH4gfyftjV54IT0fpq4RDWPqL/view?usp=sharing
GDRIVE_FILE_ID = "1MjrffVfmH4gfyftjV54IT0fpq4RDWPqL"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"


def download_dataset(
    output_dir: str = ".",
    archive_name: str = "vifoodlabel_images.zip",
    quiet: bool = False,
) -> bool:
    """Download and extract ViFoodLabel images.

    Args:
        output_dir: Directory to save archive and extract to
        archive_name: Name of the downloaded zip file
        quiet: Suppress progress output

    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    archive_path = output_path / archive_name

    print("=" * 60)
    print("ViFoodLabel Dataset Download")
    print("=" * 60)
    print(f"\nDownloading from Google Drive...")
    print(f"File ID: {GDRIVE_FILE_ID}")
    print(f"Destination: {archive_path}\n")

    try:
        # Download from Google Drive
        gdown.download(GDRIVE_URL, str(archive_path), quiet=quiet, use_cookies=False)

        if not archive_path.exists():
            print(f"❌ Download failed: File not found at {archive_path}")
            return False

        print(f"✓ Downloaded: {archive_path}")
        print(f"  Size: {archive_path.stat().st_size / (1024 * 1024):.1f} MB\n")

        # Extract archive
        print("Extracting archive...")
        try:
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(output_path)
            print(f"✓ Extracted to: {output_path}")
        except zipfile.BadZipFile:
            print(f"❌ Error: Invalid zip file. Download may have been corrupted.")
            print(f"   Try deleting {archive_path} and re-running this script.")
            return False

        # Verify structure
        data_path = output_path / "data"
        raw_path = data_path / "raw"
        thumb_path = data_path / "thumbnail"

        if raw_path.exists():
            num_raw = len(list(raw_path.glob("*.jpeg"))) + len(list(raw_path.glob("*.jpg")))
            print(f"  Found {num_raw} images in data/raw/")
        else:
            print(f"⚠ Warning: data/raw/ not found after extraction")

        if thumb_path.exists():
            num_thumb = len(list(thumb_path.glob("*.jpeg"))) + len(list(thumb_path.glob("*.jpg")))
            print(f"  Found {num_thumb} thumbnails in data/thumbnail/")
        else:
            print(f"⚠ Warning: data/thumbnail/ not found after extraction")

        print("\n" + "=" * 60)
        print("✓ Download complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run pre-annotation: python scripts/preprocessing/label_studio_preann.py")
        print("2. Upload to Label Studio")
        print("3. Annotate and export")
        print("4. Convert: python scripts/preprocessing/convert_ls_to_ner.py")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ViFoodLabel images from Google Drive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to download to (default: current directory)",
    )
    parser.add_argument(
        "--archive-name",
        default="vifoodlabel_images.zip",
        help="Name of the zip file to download",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    success = download_dataset(
        output_dir=args.output_dir,
        archive_name=args.archive_name,
        quiet=args.quiet,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
