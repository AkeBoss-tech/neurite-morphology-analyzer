"""
_widget.py
----------
Napari dockable widget for the Neurite Morphology Analyzer.

The widget exposes all key analysis parameters through a Qt form and runs
the analysis in a background thread so the GUI stays responsive.  Results
are added directly to the napari viewer as Labels layers and a summary is
shown in the in-widget results panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari

from ._analyzer import NeuriteAnalyzer


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

class _AnalysisWorker(QThread):
    """Runs NeuriteAnalyzer.analyze_array() off the main thread."""

    # Emitted on success: (soma_labels, skeleton, df, n_soma)
    finished = Signal(object, object, object, int)
    # Emitted on failure: error message string
    error = Signal(str)

    def __init__(
        self,
        analyzer: NeuriteAnalyzer,
        soma_data: np.ndarray,
        neurite_data: "np.ndarray | None",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._analyzer = analyzer
        self._soma_data = soma_data
        self._neurite_data = neurite_data

    def run(self) -> None:
        try:
            soma_labels, skeleton, df = self._analyzer.analyze_array(
                self._soma_data, self._neurite_data
            )
            n_soma = int(soma_labels.max())
            self.finished.emit(soma_labels, skeleton, df, n_soma)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class NeuriteWidget(QWidget):
    """Dockable napari widget for neurite morphology analysis.

    Users select image layers already open in the viewer, configure analysis
    parameters, then click **Run Analysis**.  The widget adds three outputs:

    * ``Soma Labels``    — integer label image, one region per detected soma
    * ``Neurite Skeleton`` — binary skeleton overlay
    * A summary table displayed in the *Results* panel
    """

    def __init__(self, napari_viewer: "napari.Viewer") -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._worker: "_AnalysisWorker | None" = None
        self._setup_ui()

        # Keep layer-selection combos up-to-date
        self._viewer.layers.events.inserted.connect(self._refresh_layer_list)
        self._viewer.layers.events.removed.connect(self._refresh_layer_list)
        self._viewer.layers.events.reordered.connect(self._refresh_layer_list)
        self._refresh_layer_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout()
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        self.setLayout(root)

        # ---- Layer selection ----------------------------------------
        layer_group = QGroupBox("Image Layers")
        layer_form = QFormLayout()
        layer_group.setLayout(layer_form)

        from qtpy.QtWidgets import QComboBox
        self._soma_combo = QComboBox()
        self._soma_combo.setToolTip(
            "Image layer to use for soma (cell-body) detection."
        )
        layer_form.addRow("Soma layer:", self._soma_combo)

        self._neurite_combo = QComboBox()
        self._neurite_combo.setToolTip(
            "Image layer to use for neurite detection.\n"
            "Select '(same as soma)' to reuse the soma layer."
        )
        layer_form.addRow("Neurite layer:", self._neurite_combo)

        root.addWidget(layer_group)

        # ---- Parameters ---------------------------------------------
        param_group = QGroupBox("Parameters")
        param_form = QFormLayout()
        param_group.setLayout(param_form)

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(0.0001, 1000.0)
        self._pixel_size.setValue(1.0)
        self._pixel_size.setDecimals(4)
        self._pixel_size.setSingleStep(0.05)
        self._pixel_size.setSuffix(" µm/px")
        self._pixel_size.setToolTip("Physical pixel size used to convert lengths to µm.")
        param_form.addRow("Pixel size:", self._pixel_size)

        self._use_cellpose = QCheckBox("Use Cellpose (deep-learning)")
        self._use_cellpose.setChecked(True)
        self._use_cellpose.setToolTip(
            "Use the Cellpose 'cyto3' model for soma detection.\n"
            "Falls back to Laplacian-of-Gaussian blob detection when unchecked\n"
            "or when Cellpose is not installed."
        )
        param_form.addRow(self._use_cellpose)

        self._min_radius = QSpinBox()
        self._min_radius.setRange(1, 500)
        self._min_radius.setValue(10)
        self._min_radius.setSuffix(" px")
        self._min_radius.setToolTip("Minimum expected soma radius in pixels.")
        param_form.addRow("Min soma radius:", self._min_radius)

        self._max_radius = QSpinBox()
        self._max_radius.setRange(1, 1000)
        self._max_radius.setValue(40)
        self._max_radius.setSuffix(" px")
        self._max_radius.setToolTip("Maximum expected soma radius in pixels.")
        param_form.addRow("Max soma radius:", self._max_radius)

        self._soma_thresh = QDoubleSpinBox()
        self._soma_thresh.setRange(0.001, 1.0)
        self._soma_thresh.setValue(0.1)
        self._soma_thresh.setDecimals(3)
        self._soma_thresh.setSingleStep(0.01)
        self._soma_thresh.setToolTip(
            "Blob-log detection sensitivity (only used without Cellpose).\n"
            "Lower value → more detections."
        )
        param_form.addRow("Soma threshold:", self._soma_thresh)

        self._neurite_thresh = QDoubleSpinBox()
        self._neurite_thresh.setRange(0.0001, 1.0)
        self._neurite_thresh.setValue(0.01)
        self._neurite_thresh.setDecimals(4)
        self._neurite_thresh.setSingleStep(0.001)
        self._neurite_thresh.setToolTip(
            "Frangi vesselness binarisation threshold.\n"
            "Lower value → more neurite pixels detected."
        )
        param_form.addRow("Neurite threshold:", self._neurite_thresh)

        self._min_branch = QDoubleSpinBox()
        self._min_branch.setRange(1.0, 10000.0)
        self._min_branch.setValue(10.0)
        self._min_branch.setDecimals(1)
        self._min_branch.setSuffix(" px")
        self._min_branch.setToolTip(
            "Skeleton branches shorter than this value are discarded as noise."
        )
        param_form.addRow("Min branch length:", self._min_branch)

        root.addWidget(param_group)

        # ---- Run button ---------------------------------------------
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setToolTip("Start the analysis on the selected layers.")
        self._run_btn.clicked.connect(self._on_run_clicked)
        btn_row.addWidget(self._run_btn)
        root.addLayout(btn_row)

        # ---- Status label -------------------------------------------
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self._status)

        # ---- Results panel ------------------------------------------
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)

        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMaximumHeight(180)
        self._results_text.setPlaceholderText("Per-soma measurements will appear here after analysis.")
        results_layout.addWidget(self._results_text)

        root.addWidget(results_group)
        root.addStretch()

    # ------------------------------------------------------------------
    # Layer list management
    # ------------------------------------------------------------------

    def _refresh_layer_list(self, event=None) -> None:
        """Repopulate layer combos from current viewer layers."""
        try:
            import napari
            image_names = [
                layer.name
                for layer in self._viewer.layers
                if isinstance(layer, napari.layers.Image)
            ]
        except Exception:  # noqa: BLE001
            image_names = []

        for combo in (self._soma_combo,):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(image_names)
            if prev in image_names:
                combo.setCurrentText(prev)
            combo.blockSignals(False)

        prev_neurite = self._neurite_combo.currentText()
        self._neurite_combo.blockSignals(True)
        self._neurite_combo.clear()
        self._neurite_combo.addItem("(same as soma)")
        self._neurite_combo.addItems(image_names)
        if prev_neurite in image_names:
            self._neurite_combo.setCurrentText(prev_neurite)
        self._neurite_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Run logic
    # ------------------------------------------------------------------

    def _on_run_clicked(self) -> None:
        soma_name = self._soma_combo.currentText()
        if not soma_name:
            self._status.setText("⚠ No image layers available. Open an image first.")
            return

        try:
            soma_layer = self._viewer.layers[soma_name]
        except KeyError:
            self._status.setText(f"⚠ Layer '{soma_name}' not found.")
            return

        soma_data = soma_layer.data

        neurite_name = self._neurite_combo.currentText()
        neurite_data: "np.ndarray | None" = None
        if neurite_name and neurite_name != "(same as soma)":
            try:
                neurite_data = self._viewer.layers[neurite_name].data
            except KeyError:
                self._status.setText(
                    f"⚠ Neurite layer '{neurite_name}' not found; using soma layer."
                )

        analyzer = NeuriteAnalyzer(
            pixel_size_um=self._pixel_size.value(),
            min_soma_radius=self._min_radius.value(),
            max_soma_radius=self._max_radius.value(),
            soma_threshold=self._soma_thresh.value(),
            neurite_threshold=self._neurite_thresh.value(),
            min_neurite_length=self._min_branch.value(),
            save_viz=False,
            use_cellpose=self._use_cellpose.isChecked(),
        )

        self._run_btn.setEnabled(False)
        self._status.setText("⏳ Running analysis…")
        self._results_text.clear()

        self._worker = _AnalysisWorker(analyzer, soma_data, neurite_data, parent=self)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(
        self,
        soma_labels: np.ndarray,
        skeleton: np.ndarray,
        df: pd.DataFrame,
        n_soma: int,
    ) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText(f"✓ Done — {n_soma} soma detected.")

        # Add label layers to viewer (replace existing ones with the same name)
        for name, data in (
            ("Soma Labels", soma_labels),
            ("Neurite Skeleton", skeleton.astype(np.uint8)),
        ):
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)
            self._viewer.add_labels(data, name=name)

        # Show per-soma summary
        if not df.empty:
            cols = [
                "soma_id",
                "n_primary_neurites",
                "n_branch_points",
                "n_endpoints",
                "total_length_um",
                "max_path_length_um",
            ]
            present = [c for c in cols if c in df.columns]
            self._results_text.setText(df[present].to_string(index=False))
        else:
            self._results_text.setText("No soma or neurites detected.")

    def _on_error(self, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText(f"✗ Error: {message}")
