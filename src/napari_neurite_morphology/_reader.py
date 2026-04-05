"""
_reader.py
----------
Napari reader contribution for TIFF microscopy files.

Drag-and-dropping a .tif / .tiff file onto napari triggers this reader.
It uses tifffile for correct handling of 16-bit images, multi-channel TIFFs,
and Z-stacks — all common formats in fluorescence microscopy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def napari_get_reader(path: "str | list[str]") -> "Callable | None":
    """Return a reader function for .tif/.tiff paths, or ``None`` otherwise.

    This is the npe2 reader hook; napari calls it with the path(s) that the
    user dropped onto the viewer or opened via *File → Open*.
    """
    if isinstance(path, list):
        # Accept only if every path in the list is a TIFF
        if not all(_is_tiff(p) for p in path):
            return None
    elif not _is_tiff(path):
        return None

    return _read_tiff


def _is_tiff(path: str) -> bool:
    return Path(path).suffix.lower() in {".tif", ".tiff"}


def _read_tiff(path: "str | list[str]") -> list[tuple]:
    """Load one or more TIFF files and return napari layer data tuples.

    Each returned tuple is ``(data, metadata, layer_type)`` as required by
    the napari reader protocol.
    """
    import tifffile

    if isinstance(path, str):
        paths = [path]
    else:
        paths = list(path)

    layers = []
    for p in paths:
        data: np.ndarray = tifffile.imread(p)
        name = Path(p).stem

        # Build channel-axis metadata so napari displays multi-channel TIFFs
        # correctly without flattening them.
        meta: dict = {"name": name}

        # Detect channel-first layout (C, H, W) where C is small
        if data.ndim == 3 and data.shape[0] <= 4:
            meta["channel_axis"] = 0

        layers.append((data, meta, "image"))

    return layers
