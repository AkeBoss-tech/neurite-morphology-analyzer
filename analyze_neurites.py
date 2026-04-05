"""
analyze_neurites.py
-------------------
CLI wrapper for the napari-neurite-morphology package.

All analysis logic lives in src/napari_neurite_morphology/_analyzer.py.
This file exists purely for backward-compatible command-line usage:

    python analyze_neurites.py <image_dir> <output_dir> [options]

    python analyze_neurites.py ./images ./output --pixel-size 0.325
"""

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the core library.  Works whether the package is installed
# (pip install -e .) or run directly from the repo root.
# ---------------------------------------------------------------------------
try:
    from napari_neurite_morphology._analyzer import (
        NeuriteAnalyzer,
        CELLPOSE_AVAILABLE,
        SKAN_AVAILABLE,
        log,
    )
except ImportError:
    # Not installed yet — add src/ to the path so the import works anyway.
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from napari_neurite_morphology._analyzer import (
        NeuriteAnalyzer,
        CELLPOSE_AVAILABLE,
        SKAN_AVAILABLE,
        log,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated neurite length analysis from fluorescence microscopy images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory for results and annotated images")
    parser.add_argument(
        "--soma-channel", type=int, default=None,
        help="0-indexed channel for soma detection (None = auto/grayscale)",
    )
    parser.add_argument(
        "--neurite-channel", type=int, default=None,
        help="0-indexed channel for neurite detection (None = same as soma)",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=1.0,
        help="Micrometers per pixel",
    )
    parser.add_argument(
        "--min-soma-radius", type=int, default=10,
        help="Minimum soma radius (px)",
    )
    parser.add_argument(
        "--max-soma-radius", type=int, default=40,
        help="Maximum soma radius (px)",
    )
    parser.add_argument(
        "--soma-threshold", type=float, default=0.1,
        help="blob_log detection sensitivity (lower = more detections)",
    )
    parser.add_argument(
        "--neurite-threshold", type=float, default=0.01,
        help="Frangi filter binarization threshold (lower = more neurites)",
    )
    parser.add_argument(
        "--min-branch-length", type=float, default=10.0,
        help="Discard skeleton branches shorter than this (px)",
    )
    parser.add_argument(
        "--no-visualization", action="store_true",
        help="Skip saving annotated images (faster for large batches)",
    )
    parser.add_argument(
        "--no-cellpose", action="store_true",
        help="Force classical blob_log soma detection even if Cellpose is installed",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    analyzer = NeuriteAnalyzer(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        soma_channel=args.soma_channel,
        neurite_channel=args.neurite_channel,
        pixel_size_um=args.pixel_size,
        min_soma_radius=args.min_soma_radius,
        max_soma_radius=args.max_soma_radius,
        soma_threshold=args.soma_threshold,
        neurite_threshold=args.neurite_threshold,
        min_neurite_length=args.min_branch_length,
        save_viz=not args.no_visualization,
        use_cellpose=not args.no_cellpose,
    )

    results = analyzer.run()

    if results.empty:
        log.error("No results. Check your image directory and parameters.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
