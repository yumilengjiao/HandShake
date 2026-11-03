"""Bounded local slide images for the demo; no PPT macros, URLs or document code."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import warnings

from shared.presentation_protocol import Deck

MAX_FILE_BYTES = 16 * 1024**2
MAX_PIXELS = 8_000_000


def load_slide_images(directory):
    """Read 1..32 lexicographically ordered PNG/JPEG files, resize into bounded RAM.

    Export a PPT/PDF to numbered 001.png, 002.png, ... first. This does not control
    or parse PowerPoint, WPS, LibreOffice, PDF JavaScript or embedded media.
    """
    from PIL import Image
    import numpy as np

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("slide directory does not exist")
    paths = sorted(
        p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not 1 <= len(paths) <= 32:
        raise ValueError("use 1..32 numbered PNG/JPEG slides")
    identity = sha256()
    images = []
    for path in paths:
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError(
                "slide images must be regular files inside the selected directory"
            )
        with path.open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("slide file exceeds 16 MiB")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                if (
                    image.format not in {"PNG", "JPEG"}
                    or image.width * image.height > MAX_PIXELS
                    or getattr(image, "n_frames", 1) != 1
                ):
                    raise ValueError(
                        "slide must be a single PNG/JPEG of at most 8 megapixels"
                    )
                image.thumbnail((960, 540))
                canvas = Image.new("RGB", (960, 540), "white")
                canvas.paste(
                    image.convert("RGB"),
                    ((960 - image.width) // 2, (540 - image.height) // 2),
                )
                images.append(np.asarray(canvas).copy()[:, :, ::-1].copy())
        identity.update(sha256(raw).digest())
    deck = Deck.model_validate_json(
        json.dumps(
            {
                "format_version": "presentation-deck-v1",
                "title": "Local imported slide images",
                "slides": [
                    {
                        "title": f"Slide {i + 1}",
                        "lines": ["User-selected local slide image"],
                    }
                    for i in range(len(images))
                ],
            }
        )
    )
    return (deck, identity.hexdigest()), tuple(images)
