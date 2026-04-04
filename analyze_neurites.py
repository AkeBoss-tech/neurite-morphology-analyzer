"""
analyze_neurites.py
-------------------
Automated neurite length analysis from fluorescence microscopy images.

Detects soma (cell bodies) and neurites, skeletonizes the neurite network,
builds a connectivity graph, and outputs per-soma measurements.

Usage:
    python3 analyze_neurites.py <image_dir> <output_dir> [options]

    python3 analyze_neurites.py ./images ./output --pixel-size 0.325
"""

import argparse
import logging
import warnings
from collections import deque
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from skimage import color, filters, io, morphology, restoration
from skimage.draw import disk
from skimage.feature import blob_log
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    remove_small_objects,
    skeletonize,
    disk as skdisk,
)

try:
    from cellpose import models as cellpose_models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
import tifffile

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from skan import Skeleton, summarize as skan_summarize
    SKAN_AVAILABLE = True
except ImportError:
    SKAN_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(image: np.ndarray) -> np.ndarray:
    """Stretch image to [0, 1] float32 using 2nd–98th percentile clipping."""
    img = image.astype(np.float32)
    lo, hi = np.percentile(img, (2, 98))
    if hi - lo < 1e-9:
        return np.zeros_like(img)
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return img


def _extract_2d(image: np.ndarray, channel: Optional[int]) -> np.ndarray:
    """
    Reduce any-dimensional array to a single 2D (H, W) float32 plane.

    Handles:
      (H, W)          — single channel
      (H, W, C)       — channel-last (RGB/RGBA or multi-channel TIFF)
      (C, H, W)       — channel-first TIFF convention
      (Z, C, H, W)    — Z-stack, max-projected first
    """
    ndim = image.ndim
    shape = image.shape

    # Z-stack: max project over Z first → (C, H, W)
    if ndim == 4:
        log.info("  Z-stack detected (%s), max-projecting over Z axis", shape)
        image = image.max(axis=0)
        ndim = 3
        shape = image.shape

    if ndim == 2:
        return image.astype(np.float32)

    # 3D Z-stack without channel dim: (Z, H, W) where Z << H, W
    if ndim == 3 and shape[0] < shape[1] and shape[0] < shape[2] and shape[0] < 64:
        log.info("  3D Z-stack detected (%s), max-projecting over Z", shape)
        image = image.max(axis=0)
        return _normalize(image.astype(np.float32))

    if ndim == 3:
        # Decide layout: (H, W, C) vs (C, H, W)
        # Heuristic: if last dim is small (≤4) and others are larger → channel-last
        if shape[2] <= 4 and shape[0] > 4 and shape[1] > 4:
            # (H, W, C)
            if shape[2] == 1:
                return image[:, :, 0].astype(np.float32)
            if channel is not None and channel < shape[2]:
                return image[:, :, channel].astype(np.float32)
            # Default: convert RGB to grayscale or take first channel
            if shape[2] in (3, 4):
                gray = color.rgb2gray(image[:, :, :3].astype(np.float32))
                return gray.astype(np.float32)
            return image[:, :, 0].astype(np.float32)

        elif shape[0] <= 4 and shape[1] > 4 and shape[2] > 4:
            # (C, H, W)
            ch = channel if (channel is not None and channel < shape[0]) else 0
            return image[ch].astype(np.float32)

        else:
            # Ambiguous — treat as (H, W, C)
            if channel is not None and channel < shape[2]:
                return image[:, :, channel].astype(np.float32)
            if shape[2] in (3, 4):
                return color.rgb2gray(image[:, :, :3].astype(np.float32)).astype(np.float32)
            return image[:, :, 0].astype(np.float32)

    raise ValueError(f"Cannot handle image with shape {image.shape}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NeuriteAnalyzer:
    """Full pipeline for neurite length analysis from fluorescence images."""

    def __init__(
        self,
        image_dir: str,
        output_dir: str,
        soma_channel: Optional[int] = None,
        neurite_channel: Optional[int] = None,
        pixel_size_um: float = 1.0,
        min_soma_radius: int = 10,
        max_soma_radius: int = 40,
        soma_threshold: float = 0.1,
        frangi_sigmas: tuple = (1, 2, 3),
        neurite_threshold: float = 0.01,
        min_neurite_length: float = 10.0,
        save_viz: bool = True,
        use_cellpose: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.soma_channel = soma_channel
        self.neurite_channel = neurite_channel
        self.pixel_size_um = pixel_size_um
        self.min_soma_radius = min_soma_radius
        self.max_soma_radius = max_soma_radius
        self.soma_threshold = soma_threshold
        self.frangi_sigmas = frangi_sigmas
        self.neurite_threshold = neurite_threshold
        self.min_neurite_length = min_neurite_length
        self.save_viz = save_viz
        self.use_cellpose = use_cellpose and CELLPOSE_AVAILABLE

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not SKAN_AVAILABLE:
            log.warning(
                "skan not installed — falling back to manual skeleton graph. "
                "Install skan for better branch measurements: pip install skan"
            )

    # ------------------------------------------------------------------
    # Stage 1: I/O
    # ------------------------------------------------------------------

    def load_image(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Load image and return (soma_plane, neurite_plane) both normalized float32 (H, W).
        """
        suffix = path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            raw = tifffile.imread(str(path))
        else:
            raw = io.imread(str(path))

        soma_plane = _extract_2d(raw, self.soma_channel)
        if self.neurite_channel is not None and self.neurite_channel != self.soma_channel:
            neurite_plane = _extract_2d(raw, self.neurite_channel)
        else:
            neurite_plane = soma_plane.copy()

        soma_plane = _normalize(soma_plane)
        neurite_plane = _normalize(neurite_plane)
        return soma_plane, neurite_plane

    # ------------------------------------------------------------------
    # Stage 2: Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Neurite preprocessing: Gaussian blur → rolling-ball background subtraction → normalize.
        Rolling ball removes slow illumination gradients while keeping thin features.
        Use a large radius so it doesn't hollow out soma (soma diameter ≈ 40-70px → radius 80)."""
        blurred = filters.gaussian(image, sigma=1.0, preserve_range=True)
        try:
            ball_radius = max(self.max_soma_radius * 2.5, 80)
            background = restoration.rolling_ball(blurred, radius=ball_radius)
            corrected = np.clip(blurred - background, 0, None)
        except Exception:
            corrected = blurred
        return _normalize(corrected)

    def preprocess_soma(self, image: np.ndarray) -> np.ndarray:
        """Soma preprocessing: only Gaussian blur + normalize.
        No background subtraction — soma are the globally brightest features
        and rolling ball would hollow them out."""
        blurred = filters.gaussian(image, sigma=1.5, preserve_range=True)
        return _normalize(blurred)

    # ------------------------------------------------------------------
    # Stage 3: Soma detection
    # ------------------------------------------------------------------

    def detect_soma(self, image: np.ndarray) -> list[dict]:
        """
        Detect soma. Uses Cellpose (deep learning) when available for best accuracy,
        otherwise falls back to Laplacian-of-Gaussian blob detection.
        Returns list of dicts: {y, x, radius, mean_intensity}.
        """
        if self.use_cellpose:
            return self._detect_soma_cellpose(image)
        return self._detect_soma_blob(image)

    def _detect_soma_cellpose(self, image: np.ndarray) -> list[dict]:
        """Cellpose instance segmentation for soma detection (ML-based)."""
        log.info("  Using Cellpose for soma detection")
        model = cellpose_models.CellposeModel(model_type="cyto3", gpu=False)

        # Cellpose expects uint8 or uint16; scale to uint8
        img8 = (image * 255).clip(0, 255).astype(np.uint8)

        diameter = (self.min_soma_radius + self.max_soma_radius)  # avg expected diameter
        masks, _, _ = model.eval(
            img8,
            diameter=diameter,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )

        soma_list = []
        for label_val in range(1, masks.max() + 1):
            cell_mask = masks == label_val
            area = cell_mask.sum()
            if area < np.pi * self.min_soma_radius**2 * 0.5:
                continue  # too small
            coords = np.argwhere(cell_mask)
            cy, cx = coords.mean(axis=0)
            radius = np.sqrt(area / np.pi)
            if radius > self.max_soma_radius * 1.5:
                continue  # too large — likely a merged cluster
            rr, cc = np.where(cell_mask)
            mean_int = float(image[rr, cc].mean())
            soma_list.append({
                "y": float(cy),
                "x": float(cx),
                "radius": float(radius),
                "mean_intensity": mean_int,
            })

        log.info("  Cellpose detected %d soma", len(soma_list))
        return soma_list

    def _detect_soma_blob(self, image: np.ndarray) -> list[dict]:
        """Fallback: Laplacian-of-Gaussian blob detection (classical)."""
        min_sigma = self.min_soma_radius / np.sqrt(2)
        max_sigma = self.max_soma_radius / np.sqrt(2)

        blobs = blob_log(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=15,
            threshold=self.soma_threshold,
            overlap=0.5,
        )

        # Use 80th percentile as brightness threshold — soma must be in the top 20%
        p80 = np.percentile(image, 80)
        soma_list = []
        for blob in blobs:
            y, x, sigma = blob
            radius = sigma * np.sqrt(2)
            rr, cc = disk((y, x), radius, shape=image.shape)
            if len(rr) == 0:
                continue
            mean_int = image[rr, cc].mean()
            peak_int = image[int(np.clip(round(y), 0, image.shape[0]-1)),
                             int(np.clip(round(x), 0, image.shape[1]-1))]
            # Require both mean and peak to be clearly above background
            if mean_int < p80 * 0.75 or peak_int < p80:
                continue
            soma_list.append({
                "y": float(y),
                "x": float(x),
                "radius": float(radius),
                "mean_intensity": float(mean_int),
            })

        log.info("  Detected %d soma (blob_log)", len(soma_list))
        return soma_list

    def _create_soma_mask(self, soma_list: list[dict], shape: tuple) -> np.ndarray:
        """Return bool mask with filled circles at each soma location."""
        mask = np.zeros(shape, dtype=bool)
        for s in soma_list:
            rr, cc = disk((s["y"], s["x"]), s["radius"], shape=shape)
            mask[rr, cc] = True
        return mask

    # ------------------------------------------------------------------
    # Stage 4: Neurite detection
    # ------------------------------------------------------------------

    def detect_neurites(
        self, image: np.ndarray, soma_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply Frangi vesselness filter and return binary neurite mask.
        Soma regions are excluded.
        """
        vesselness = filters.frangi(
            image,
            sigmas=self.frangi_sigmas,
            black_ridges=False,
        )
        neurite_mask = vesselness > self.neurite_threshold

        # Dilate soma mask to clean boundaries, then exclude
        soma_exclusion = binary_dilation(soma_mask, footprint=skdisk(5))
        neurite_mask = neurite_mask & ~soma_exclusion

        # Remove small noise objects
        neurite_mask = remove_small_objects(neurite_mask, min_size=20)

        # Bridge tiny gaps
        neurite_mask = binary_closing(neurite_mask, footprint=skdisk(2))

        return neurite_mask

    # ------------------------------------------------------------------
    # Stage 5: Skeletonization
    # ------------------------------------------------------------------

    def skeletonize_mask(self, neurite_mask: np.ndarray) -> np.ndarray:
        """Reduce neurite mask to 1-pixel-wide skeleton."""
        if not neurite_mask.any():
            return np.zeros_like(neurite_mask)
        return skeletonize(neurite_mask)

    # ------------------------------------------------------------------
    # Stage 6: Graph construction
    # ------------------------------------------------------------------

    def build_skeleton_graph(
        self, skeleton: np.ndarray
    ) -> tuple:
        """
        Convert skeleton to branch DataFrame using skan (preferred) or fallback.
        Returns (skel_obj_or_None, branch_df).
        """
        if not skeleton.any():
            return None, pd.DataFrame()

        if SKAN_AVAILABLE:
            return self._build_graph_skan(skeleton)
        else:
            return self._build_graph_fallback(skeleton)

    def _build_graph_skan(self, skeleton: np.ndarray):
        # Use spacing=1 so coordinates are always in pixels (not µm).
        # We scale branch-distance to µm ourselves below.
        skel_obj = Skeleton(skeleton.astype(bool), spacing=1.0)
        try:
            branch_df = skan_summarize(skel_obj, separator="-")
        except TypeError:
            branch_df = skan_summarize(skel_obj)

        # Normalize column names (skan versions differ)
        col_map = {}
        for col in branch_df.columns:
            low = col.lower().replace(" ", "-")
            col_map[col] = low
        branch_df = branch_df.rename(columns=col_map)

        # Ensure required columns exist
        required = ["branch-distance", "branch-type",
                    "coord-src-0", "coord-src-1",
                    "coord-dst-0", "coord-dst-1"]
        for r in required:
            if r not in branch_df.columns:
                # Try alternative naming
                alt = r.replace("-", "_")
                if alt in branch_df.columns:
                    branch_df[r] = branch_df[alt]
                else:
                    branch_df[r] = np.nan

        # Filter short branches (branch-distance is still in pixels at this point)
        branch_df = branch_df[
            branch_df["branch-distance"] >= self.min_neurite_length
        ].reset_index(drop=True)

        # Convert branch-distance from pixels → µm
        branch_df["branch-distance"] = branch_df["branch-distance"] * self.pixel_size_um

        log.info("  Skeleton: %d branches after filtering", len(branch_df))
        return skel_obj, branch_df

    def _build_graph_fallback(self, skeleton: np.ndarray):
        """
        Fallback skeleton graph using connected-component labeling.
        Less accurate than skan but works without it.
        """
        labeled, n = ndi.label(skeleton)
        rows = []
        for i in range(1, n + 1):
            coords = np.argwhere(labeled == i)
            if len(coords) < 2:
                continue
            # Approximate length as number of pixels * sqrt(2) for diagonals
            # Build a mini path by nearest-neighbor ordering
            length = self._approx_skeleton_length(coords)
            if length < self.min_neurite_length:
                continue
            src = coords[0]
            dst = coords[-1]
            rows.append({
                "branch-distance": length * self.pixel_size_um,
                "branch-type": 1,
                "coord-src-0": float(src[0]),
                "coord-src-1": float(src[1]),
                "coord-dst-0": float(dst[0]),
                "coord-dst-1": float(dst[1]),
            })
        branch_df = pd.DataFrame(rows)
        log.info("  Skeleton (fallback): %d branches", len(branch_df))
        return None, branch_df

    def _approx_skeleton_length(self, coords: np.ndarray) -> float:
        """Approximate path length through ordered skeleton coordinates."""
        if len(coords) < 2:
            return 0.0
        # Greedy nearest-neighbor ordering
        ordered = [coords[0]]
        remaining = list(coords[1:])
        while remaining:
            last = ordered[-1]
            dists = [np.linalg.norm(last - c) for c in remaining]
            idx = int(np.argmin(dists))
            ordered.append(remaining.pop(idx))
        total = sum(
            np.linalg.norm(np.array(ordered[i+1]) - np.array(ordered[i]))
            for i in range(len(ordered) - 1)
        )
        return float(total)

    # ------------------------------------------------------------------
    # Stage 7: Build networkx branch graph for BFS attribution
    # ------------------------------------------------------------------

    def _build_branch_graph(self, branch_df: pd.DataFrame) -> nx.Graph:
        """
        Build networkx graph where nodes are (row, col) pixel coords
        and edges are skeleton branches weighted by branch-distance.
        """
        G = nx.Graph()
        for idx, row in branch_df.iterrows():
            src = (round(row["coord-src-0"]), round(row["coord-src-1"]))
            dst = (round(row["coord-dst-0"]), round(row["coord-dst-1"]))
            G.add_edge(src, dst, branch_idx=idx, weight=row["branch-distance"])
        return G

    # ------------------------------------------------------------------
    # Stage 8: Attribute branches to soma
    # ------------------------------------------------------------------

    def partition_skeleton_by_soma(
        self,
        skeleton: np.ndarray,
        soma_list: list[dict],
    ) -> np.ndarray:
        """
        Assign each skeleton pixel to the nearest soma center (Voronoi on pixels).
        Returns int array labeled 1…N (0 = no soma within cutoff).
        This is more accurate than branch-midpoint attribution because each pixel
        is independently assigned, avoiding flooding through connected paths.
        """
        if not soma_list or not skeleton.any():
            return np.zeros_like(skeleton, dtype=np.int32)

        skel_coords = np.argwhere(skeleton)  # (K, 2)
        if len(skel_coords) == 0:
            return np.zeros_like(skeleton, dtype=np.int32)

        centers = np.array([[s["y"], s["x"]] for s in soma_list])  # (M, 2)
        # Vectorized pairwise distance: (K, M)
        diffs = skel_coords[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        nearest = dists.argmin(axis=1)  # index of nearest soma per pixel

        labeled = np.zeros_like(skeleton, dtype=np.int32)
        labeled[skel_coords[:, 0], skel_coords[:, 1]] = nearest + 1  # 1-indexed
        return labeled

    # ------------------------------------------------------------------
    # Stage 9: Per-soma measurements
    # ------------------------------------------------------------------

    def _empty_soma_measurement(self, soma_idx: int, soma: dict) -> dict:
        return {
            "soma_id": soma_idx, "soma_x": soma["x"], "soma_y": soma["y"],
            "soma_radius_px": soma["radius"], "soma_mean_intensity": soma["mean_intensity"],
            "n_primary_neurites": 0, "n_branch_points": 0, "n_endpoints": 0,
            "n_branches_attributed": 0, "total_length_um": 0.0,
            "mean_segment_length_um": np.nan, "max_path_length_um": 0.0,
        }

    def measure_soma(
        self,
        soma_idx: int,
        soma: dict,
        branches: pd.DataFrame,
        G: nx.Graph,
    ) -> dict:
        """Compute all neurite metrics for one soma."""
        subG = G  # G is already this soma's sub-graph from per-soma analysis

        # Primary = branch with EITHER endpoint within 2× soma radius
        primary_r = soma["radius"] * 2.0
        n_primary = 0
        for _, row in branches.iterrows():
            src = (row["coord-src-0"], row["coord-src-1"])
            dst = (row["coord-dst-0"], row["coord-dst-1"])
            if (np.hypot(src[0] - soma["y"], src[1] - soma["x"]) <= primary_r or
                    np.hypot(dst[0] - soma["y"], dst[1] - soma["x"]) <= primary_r):
                n_primary += 1

        # Branch points and endpoints
        n_branch_pts = sum(1 for n in subG.nodes if subG.degree(n) >= 3)
        n_endpoints = sum(1 for n in subG.nodes if subG.degree(n) == 1)

        total_length = branches["branch-distance"].sum()
        mean_length = total_length / len(branches) if len(branches) > 0 else np.nan

        # Max path length from soma center via Dijkstra
        max_path = self._max_path_from_soma(soma, subG)

        return {
            "soma_id": soma_idx,
            "soma_x": soma["x"],
            "soma_y": soma["y"],
            "soma_radius_px": soma["radius"],
            "soma_mean_intensity": soma["mean_intensity"],
            "n_primary_neurites": n_primary,
            "n_branch_points": n_branch_pts,
            "n_endpoints": n_endpoints,
            "n_branches_attributed": len(branches),
            "total_length_um": float(total_length),
            "mean_segment_length_um": float(mean_length) if not np.isnan(mean_length) else np.nan,
            "max_path_length_um": float(max_path),
        }

    def _max_path_from_soma(self, soma: dict, subG: nx.Graph) -> float:
        """Longest Dijkstra path from the graph node nearest to soma center."""
        if subG.number_of_nodes() == 0:
            return 0.0
        nodes = list(subG.nodes)
        dists = [np.hypot(n[0] - soma["y"], n[1] - soma["x"]) for n in nodes]
        entry_node = nodes[int(np.argmin(dists))]
        try:
            lengths = nx.single_source_dijkstra_path_length(
                subG, entry_node, weight="weight"
            )
            return max(lengths.values(), default=0.0)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Stage 10: Visualization
    # ------------------------------------------------------------------

    def save_visualization(
        self,
        image: np.ndarray,
        soma_list: list[dict],
        skeleton: np.ndarray,
        branch_df: pd.DataFrame,
        measurements: list[dict],
        all_graphs: list,          # list[nx.Graph], one per soma
        output_path: Path,
    ) -> None:
        """Save 3-panel annotated figure."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

        # Panel 0 — Raw image
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Raw Image", fontsize=11)
        axes[0].axis("off")

        # Panel 1 — Detection overlay
        axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
        # Soma circles
        for s in soma_list:
            circle = mpatches.Circle(
                (s["x"], s["y"]), s["radius"],
                fill=False, edgecolor="cyan", linewidth=1.5
            )
            axes[1].add_patch(circle)
        # Skeleton overlay
        if skeleton.any():
            skel_rgba = np.zeros((*image.shape, 4), dtype=np.float32)
            skel_rgba[skeleton, 1] = 1.0
            skel_rgba[skeleton, 3] = 0.85
            axes[1].imshow(skel_rgba)
        # Branch points and tips — collected per soma sub-graph so degrees are correct.
        # Merging sub-graphs would corrupt degrees at Voronoi boundary nodes.
        all_junctions_y, all_junctions_x = [], []
        all_tips_y, all_tips_x = [], []
        for G_i in all_graphs:
            for n in G_i.nodes:
                d = G_i.degree(n)
                if d >= 3:
                    all_junctions_y.append(n[0])
                    all_junctions_x.append(n[1])
                elif d == 1:
                    all_tips_y.append(n[0])
                    all_tips_x.append(n[1])
        if all_junctions_x:
            axes[1].scatter(all_junctions_x, all_junctions_y,
                            c="red", s=12, zorder=5, label="Branch pts")
        if all_tips_x:
            axes[1].scatter(all_tips_x, all_tips_y,
                            c="yellow", s=12, zorder=5, label="Tips")
        if all_junctions_x or all_tips_x:
            axes[1].legend(fontsize=7, loc="upper right")
        axes[1].set_title("Skeleton + Detections", fontsize=11)
        axes[1].axis("off")

        # Panel 2 — Measurements
        axes[2].imshow(image, cmap="gray", vmin=0, vmax=1)
        if skeleton.any():
            skel_rgba = np.zeros((*image.shape, 4), dtype=np.float32)
            skel_rgba[skeleton, 1] = 1.0
            skel_rgba[skeleton, 3] = 0.6
            axes[2].imshow(skel_rgba)
        for m in measurements:
            label = (
                f"#{m['soma_id']}\n"
                f"L={m['total_length_um']:.0f}µm\n"
                f"P={m['n_primary_neurites']} bp={m['n_branch_points']}"
            )
            axes[2].annotate(
                label,
                xy=(m["soma_x"], m["soma_y"]),
                color="white", fontsize=6, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55),
            )
            circle = mpatches.Circle(
                (m["soma_x"], m["soma_y"]), m["soma_radius_px"],
                fill=False, edgecolor="lime", linewidth=1.2
            )
            axes[2].add_patch(circle)
        axes[2].set_title("Measurements", fontsize=11)
        axes[2].axis("off")

        plt.tight_layout(pad=1.0)
        plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info("  Saved visualization → %s", output_path.name)

    # ------------------------------------------------------------------
    # Stage 11: Results export
    # ------------------------------------------------------------------

    def results_to_dataframe(
        self, all_measurements: list[dict], image_name: str
    ) -> pd.DataFrame:
        if not all_measurements:
            return pd.DataFrame()
        df = pd.DataFrame(all_measurements)
        df.insert(0, "image_file", image_name)
        df["pixel_size_um"] = self.pixel_size_um
        return df

    # ------------------------------------------------------------------
    # Per-image orchestration
    # ------------------------------------------------------------------

    def analyze_image(self, path: Path) -> pd.DataFrame:
        """Run full pipeline on one image. Returns per-soma DataFrame."""
        log.info("Processing: %s", path.name)

        try:
            soma_plane, neurite_plane = self.load_image(path)
        except Exception as e:
            log.error("  Failed to load %s: %s", path.name, e)
            return pd.DataFrame()

        if soma_plane.size == 0 or min(soma_plane.shape) < 50:
            log.warning("  Image too small, skipping")
            return pd.DataFrame()

        # Preprocess
        soma_prep = self.preprocess_soma(soma_plane)
        neurite_prep = self.preprocess(neurite_plane)

        # Soma detection
        soma_list = self.detect_soma(soma_prep)
        if not soma_list:
            log.warning("  No soma detected in %s", path.name)
            return pd.DataFrame()

        soma_mask = self._create_soma_mask(soma_list, soma_plane.shape)

        # Neurite detection and skeletonization
        neurite_mask = self.detect_neurites(neurite_prep, soma_mask)
        skeleton = self.skeletonize_mask(neurite_mask)

        if not skeleton.any():
            log.warning("  No neurites detected in %s", path.name)

        # Partition skeleton pixels by nearest soma (pixel-wise Voronoi)
        labeled_skel = self.partition_skeleton_by_soma(skeleton, soma_list)

        # Per-soma analysis: each soma gets its own sub-skeleton
        measurements = []
        all_branch_dfs: list[pd.DataFrame] = []
        all_graphs: list[nx.Graph] = []

        for si, soma in enumerate(soma_list):
            soma_skel = (labeled_skel == (si + 1))
            if not soma_skel.any():
                measurements.append(self._empty_soma_measurement(si, soma))
                all_branch_dfs.append(pd.DataFrame())
                all_graphs.append(nx.Graph())
                continue
            _, branch_df_i = self.build_skeleton_graph(soma_skel)
            G_i = self._build_branch_graph(branch_df_i) if not branch_df_i.empty else nx.Graph()
            m = self.measure_soma(si, soma, branch_df_i, G_i)
            measurements.append(m)
            all_branch_dfs.append(branch_df_i)
            all_graphs.append(G_i)

        # Merge branch data for visualization (add soma_id column)
        viz_branch_df = pd.DataFrame()
        if any(not df.empty for df in all_branch_dfs):
            parts = []
            for si, df in enumerate(all_branch_dfs):
                if not df.empty:
                    df = df.copy()
                    df["soma_id"] = si
                    parts.append(df)
            if parts:
                viz_branch_df = pd.concat(parts, ignore_index=True)

        # Visualization — pass individual soma graphs so tip/junction degrees are correct
        if self.save_viz:
            viz_path = self.output_dir / (path.stem + "_analyzed.png")
            self.save_visualization(
                soma_prep, soma_list, skeleton, viz_branch_df, measurements, all_graphs, viz_path
            )

        return self.results_to_dataframe(measurements, path.name)

    # ------------------------------------------------------------------
    # Batch runner
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Process all images in image_dir. Return combined results DataFrame."""
        image_paths = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        if not image_paths:
            log.error(
                "No supported images found in %s  "
                "(accepted: %s)",
                self.image_dir,
                ", ".join(SUPPORTED_EXTENSIONS),
            )
            return pd.DataFrame()

        log.info("Found %d image(s) to process", len(image_paths))

        all_dfs = []
        for path in image_paths:
            df = self.analyze_image(path)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            log.warning("No results generated.")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        csv_path = self.output_dir / "results.csv"
        combined.to_csv(str(csv_path), index=False)
        log.info("Results saved → %s  (%d soma total)", csv_path, len(combined))

        self._print_summary(combined)
        return combined

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print a human-readable summary table."""
        print("\n" + "=" * 60)
        print("NEURITE ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Images processed : {df['image_file'].nunique()}")
        print(f"Soma detected    : {len(df)}")
        print(f"Avg total length : {df['total_length_um'].mean():.1f} µm")
        print(f"Avg primary neur : {df['n_primary_neurites'].mean():.1f}")
        print(f"Avg branch pts   : {df['n_branch_points'].mean():.1f}")
        print(f"Avg endpoints    : {df['n_endpoints'].mean():.1f}")
        print(f"Avg max path     : {df['max_path_length_um'].mean():.1f} µm")
        print("=" * 60)
        print("\nPer-soma results:")
        cols = [
            "image_file", "soma_id",
            "n_primary_neurites", "n_branch_points", "n_endpoints",
            "total_length_um", "max_path_length_um",
        ]
        present = [c for c in cols if c in df.columns]
        print(df[present].to_string(index=False))
        print()


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
