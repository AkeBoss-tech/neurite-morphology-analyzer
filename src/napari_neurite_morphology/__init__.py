"""napari-neurite-morphology plugin package."""

from ._analyzer import NeuriteAnalyzer
from ._reader import napari_get_reader

__all__ = ["NeuriteAnalyzer", "napari_get_reader"]

# NeuriteWidget is intentionally NOT imported here at module level.
# napari loads it lazily via the napari.yaml manifest entry, which avoids
# requiring Qt to be installed just to import the core analysis library.

