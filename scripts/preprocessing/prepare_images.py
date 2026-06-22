#!/usr/bin/env python
"""Prepare images for annotation: convert HEIC→JPEG and create thumbnails.

This script automates the image preparation workflow from iPhone photos:
1. Convert HEIC/heic → JPEG (from airdrop or camera roll)
2. Randomly shuffle and sequentially number files
3. Generate thumbnails (max 250px) for quick preview

Usage:
    python prepare_images.py \\
        --input /path/to/heic/folder \\
        --output data/raw \\
        --thumbnail data/thumbnail \\
        --start-index 200

Requirements:
    pip install Pillow
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from PIL import Image


def convert_heic_to_jpeg(
    input_dir: str,
    output_dir: str,
    start_index: int = 1,
    shuffle: bool = True,
) -> int:
    """Convert all HEIC files in input_dir to JPEG in output_dir.

    Args:
        input_dir: Directory containing HEIC files
        output_dir: Directory to save JPEG files
        start_index: Starting number for sequential naming (e.g. 0200.jpeg)
        shuffle: Randomize order before numbering

    Returns:
        Number of files converted
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all HEIC files (case-insensitive)
    heic_files = sorted(
        input_path.glob("*.[Hh][Ee][Ii][Cc]"),
        key=lambda p: p.name.lower(),
    )

    if not heic_files:
        print(f"❌ No HEIC files found in {input_dir}")
        return 0

    print(f"Found {len(heic_files)} HEIC files")

    if shuffle:
        random.shuffle(heic_files)
        print("Shuffled order randomly")

    converted = 0
    for idx, heic_path in enumerate(heic_files, start=start_index):
        try:
            # Convert HEIC → JPEG
            img = Image.open(heic_path)
            if img.mode in ("RGBA", "LA", "P"):
                # Convert to RGB if necessary
                img = img.convert("RGB")

            jpeg_name = f"{idx:04d}.jpeg"
            jpeg_path = output_path / jpeg_name

            img.save(jpeg_path, "JPEG", quality=95)
            print(f"✓ {heic_path.name} → {jpeg_name}")
            converted += 1

        except Exception as e:
            print(f"✗ Error converting {heic_path.name}: {e}")

    return converted


def create_thumbnails(
    input_dir: str,
    output_dir: str,
    max_size: int = 250,
    skip_existing: bool = True,
) -> int:
    """Create thumbnails from JPEG files.

    Args:
        input_dir: Directory containing JPEG files
        output_dir: Directory to save thumbnails
        max_size: Max width/height in pixels
        skip_existing: Skip if thumbnail already exists

    Returns:
        Number of thumbnails created
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JPEG files
    jpeg_files = sorted(
        list(input_path.glob("*.jpeg")) + list(input_path.glob("*.jpg")),
        key=lambda p: p.name,
    )

    if not jpeg_files:
        print(f"❌ No JPEG files found in {input_dir}")
        return 0

    print(f"\nFound {len(jpeg_files)} JPEG files")

    created = 0
    for jpeg_path in jpeg_files:
        thumb_path = output_path / jpeg_path.name

        if skip_existing and thumb_path.exists():
            print(f"⏭ Skip: {jpeg_path.name}")
            continue

        try:
            img = Image.open(jpeg_path)
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            img.save(thumb_path, "JPEG", quality=90)
            print(f"✓ Thumbnail: {jpeg_path.name}")
            created += 1

        except Exception as e:
            print(f"✗ Error creating thumbnail for {jpeg_path.name}: {e}")

    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare images: convert HEIC→JPEG and create thumbnails.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing HEIC files (from airdrop/camera roll)",
    )
    parser.add_argument(
        "--output",
        default="data/raw",
        help="Output directory for JPEG files (default: data/raw)",
    )
    parser.add_argument(
        "--thumbnail",
        default="data/thumbnail",
        help="Output directory for thumbnails (default: data/thumbnail)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index for sequential naming (default: 1)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle file order before numbering",
    )
    parser.add_argument(
        "--no-thumbnail",
        action="store_true",
        help="Skip thumbnail generation",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Image Preparation Workflow")
    print("=" * 60)

    # Step 1: Convert HEIC → JPEG
    print("\n[Step 1] Converting HEIC → JPEG...")
    converted = convert_heic_to_jpeg(
        args.input,
        args.output,
        start_index=args.start_index,
        shuffle=not args.no_shuffle,
    )
    print(f"→ Converted {converted} files")

    # Step 2: Create thumbnails
    if not args.no_thumbnail:
        print("\n[Step 2] Creating thumbnails...")
        created = create_thumbnails(args.output, args.thumbnail)
        print(f"→ Created {created} thumbnails")

    print("\n" + "=" * 60)
    print(f"✓ Complete! Images ready in {args.output}")
    if not args.no_thumbnail:
        print(f"✓ Thumbnails saved in {args.thumbnail}")
    print("=" * 60)


if __name__ == "__main__":
    main()
