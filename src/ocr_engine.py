"""Text detection + recognition for Vietnamese food labels.

Detection: doctr. Recognition: VietOCR (vgg_transformer). Crops from
detection are recognized concurrently via ThreadPoolExecutor.

Used to reproduce Task 3 (End-to-End KIE, see docs/benchmark-tasks.md)
locally for benchmarking — not exposed as a service. See
docs/notes/technical-runtime-notes.md (#3 Pillow/VietOCR ANTIALIAS patch).
"""

from __future__ import annotations

from dataclasses import dataclass


def _patch_pillow_antialias() -> None:
    """Restore PIL.Image.ANTIALIAS removed in Pillow >= 10, required by VietOCR."""
    import PIL.Image

    if not hasattr(PIL.Image, "ANTIALIAS"):
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS


@dataclass
class Word:
    text: str
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1, normalized [0, 1000]
    confidence: float


class OCREngine:
    """Lazy-loads doctr (detection) + VietOCR (recognition) once and reuses them."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._detector = None
        self._recognizer = None

    def _load(self) -> None:
        # TODO: load doctr detection model and VietOCR vgg_transformer predictor
        raise NotImplementedError

    def run(self, image_path: str) -> list[Word]:
        """Detect text regions, crop, recognize in parallel, return reading-order words."""
        # TODO: doctr detection -> crop boxes -> ThreadPoolExecutor + VietOCR -> XY-cut sort
        raise NotImplementedError
