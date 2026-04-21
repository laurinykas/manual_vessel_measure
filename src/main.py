"""
main.py - Pagrindinis langas (QMainWindow)

Rankinis akies kraujagyslių pločio matavimo įrankis.
Paleidimas: python main.py

Lango struktūra:
    +-----------------------------------------------------------+
    | Menu: File | Edit | View | Tools | Help                   |
    +----------------------------------------+------------------+
    |                                        | PROFILIO PANELĖ |
    |        PAGRINDINIS VAIZDAS             | (kortelės)       |
    |        (viewer_widget)                 |                  |
    +----------------------------------------+------------------+
    | Status: img.tif | Zoom:250% | Meas:12 | Mode:Measure     |
    +-----------------------------------------------------------+
"""

import json
import os
import sys
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, QSize, QPointF, QRectF
from PyQt5.QtGui import (QKeySequence, QIcon, QPixmap, QPainter, QColor,
                          QPen, QBrush, QFont, QPainterPath)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSplitter,
                              QFileDialog, QMessageBox,
                              QLabel, QAction, QMenu,
                              QProgressDialog, QToolBar)

import cv2
import numpy as np

from viewer_widget import VesselViewerWidget, Mode
from profile_panel import ProfilePanel
from measurement_manager import (MeasurementManager, VESSEL_ARTERY,
                                  VESSEL_VEIN, VESSEL_UNKNOWN)
from od_detector import ODDetector
from exclusion_zones import ExclusionZoneComputer
from image_processor import apply_clahe_for_display, get_green_channel
from pipeline.masking import create_fundus_mask
from pipeline.preprocessing import preprocessing3
from export import (import_reference_csv, import_reference_session,
                    export_image_results)



# Palaikomi formatai


IMAGE_FILTERS = (
    "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.ppm);;All (*)"
)



# MainWindow


class MainWindow(QMainWindow):
    """Pagrindinis aplikacijos langas."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Manual Vessel Measurement Tool")
        self.setMinimumSize(1024, 680)
        self.resize(1400, 850)

        # --- Komponentai ---
        self._viewer = VesselViewerWidget()
        self._panel = ProfilePanel()
        self._manager = MeasurementManager()
        self._od_detector = ODDetector()
        self._exclusion = ExclusionZoneComputer()

        # Susieti exclusion su manager
        self._manager.exclusion_computer = self._exclusion

        # --- Dabartinis vaizdas ---
        self._current_image_name: Optional[str] = None
        self._current_image_bgr: Optional[np.ndarray] = None
        self._current_green: Optional[np.ndarray] = None
        self._current_green_processed: Optional[np.ndarray] = None
        self._current_mask: Optional[np.ndarray] = None
        self._current_sc: Optional[float] = None

        # --- Eksperto reference matavimai ---
        self._reference_data: dict = {}  # {image_name: [ref_dict, ...]}

        # --- Eksporto aplankas ---
        self._results_dir: Optional[str] = None
        self._exported_images: set = set()  # Jau eksportuoti vaizdai

        # --- Exclusion debounce timer ---
        self._excl_debounce = QTimer()
        self._excl_debounce.setSingleShot(True)
        self._excl_debounce.setInterval(300)
        self._excl_debounce.timeout.connect(self._on_exclusion_debounced)

        # --- UI ---
        self._setup_central()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._setup_shortcuts()
        self._connect_signals()
        self._apply_dark_theme()

        # --- Viewer pradinis režimas ---
        self._viewer.set_mode(Mode.MEASURE)

        # --- Autosave tikrinimas ---
        QTimer.singleShot(300, self._check_autosave)

      ^#s*=+s*R
    # UI setup


    def _setup_central(self):
        """Centrinis widget: viewer + profile panel."""
        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self._viewer)
        splitter.addWidget(self._panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1100, 300])

        self.setCentralWidget(splitter)

    def _setup_menus(self):
        """Meniu juosta."""
        menubar = self.menuBar()

        # === File ===
        file_menu = menubar.addMenu("&File")

        self._act_open = file_menu.addAction("Open &Image")
        self._act_open.setShortcut(QKeySequence("Ctrl+O"))
        self._act_open.triggered.connect(self._on_open_image)

        file_menu.addSeparator()


        self._act_import_ref = file_menu.addAction(
            "Import &Expert Measurements"
        )
        self._act_import_ref.triggered.connect(self._on_import_reference)

        file_menu.addSeparator()

        self._act_export = file_menu.addAction("&Export Results")
        self._act_export.setShortcut(QKeySequence("Ctrl+E"))
        self._act_export.triggered.connect(self._on_export_results)

        file_menu.addSeparator()

        self._act_load_session = file_menu.addAction("&Load Session")
        self._act_load_session.setShortcut(QKeySequence("Ctrl+L"))
        self._act_load_session.triggered.connect(self._on_load_session)

        self._act_save_session = file_menu.addAction("&Save Session")
        self._act_save_session.setShortcut(QKeySequence("Ctrl+S"))
        self._act_save_session.triggered.connect(self._on_save_session)

        file_menu.addSeparator()

        act_exit = file_menu.addAction("E&xit")
        act_exit.setShortcut(QKeySequence("Ctrl+Q"))
        act_exit.triggered.connect(self.close)

        # === Edit ===
        edit_menu = menubar.addMenu("&Edit")

        self._act_undo = edit_menu.addAction("&Undo")
        self._act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self._act_undo.triggered.connect(self._on_undo)

        self._act_redo = edit_menu.addAction("&Redo")
        self._act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        self._act_redo.triggered.connect(self._on_redo)

        edit_menu.addSeparator()

        self._act_delete = edit_menu.addAction("&Delete Selected")
        self._act_delete.setShortcut(QKeySequence.Delete)
        self._act_delete.triggered.connect(self._on_delete_selected)

        self._act_clear = edit_menu.addAction("&Clear All")
        self._act_clear.setShortcut(QKeySequence("Ctrl+Shift+Del"))
        self._act_clear.triggered.connect(self._on_clear_all)

        # === View ===
        view_menu = menubar.addMenu("&View")

        self._act_clahe = view_menu.addAction("Toggle &CLAHE")
        self._act_clahe.setShortcut(QKeySequence("C"))
        self._act_clahe.triggered.connect(self._on_toggle_clahe)

        self._act_green = view_menu.addAction("Toggle &Green Channel")
        self._act_green.setShortcut(QKeySequence("G"))
        self._act_green.triggered.connect(self._on_toggle_green)

        view_menu.addSeparator()

        self._act_zones_circles = view_menu.addAction(
            "Toggle &Zone Circles"
        )
        self._act_zones_circles.setShortcut(QKeySequence("Z"))
        self._act_zones_circles.triggered.connect(
            self._viewer.toggle_od_circles
        )

        self._act_excl_zones = view_menu.addAction(
            "Toggle Exclusion &Zones"
        )
        self._act_excl_zones.setShortcut(QKeySequence("R"))
        self._act_excl_zones.triggered.connect(
            self._on_toggle_zones
        )

        self._act_reference = view_menu.addAction(
            "Toggle &Expert Reference"
        )
        self._act_reference.setShortcut(QKeySequence("E"))
        self._act_reference.triggered.connect(self._on_toggle_reference)

        view_menu.addSeparator()

        self._act_fit = view_menu.addAction("&Fit to Window")
        self._act_fit.setShortcut(QKeySequence("F"))
        self._act_fit.triggered.connect(self._viewer.fit_to_window)

        self._act_reset_view = view_menu.addAction("&Reset View")
        self._act_reset_view.setShortcut(QKeySequence("Home"))
        self._act_reset_view.triggered.connect(self._viewer.reset_view)

        # === Tools ===
        tools_menu = menubar.addMenu("&Tools")

        self._act_auto_od = tools_menu.addAction("Auto &Detect OD")
        self._act_auto_od.setShortcut(QKeySequence("D"))
        self._act_auto_od.triggered.connect(self._on_auto_od)

        self._act_manual_od = tools_menu.addAction("&Manual OD")
        self._act_manual_od.setShortcut(QKeySequence("Shift+D"))
        self._act_manual_od.triggered.connect(self._on_manual_od)

        tools_menu.addSeparator()

        self._act_paint_excl = tools_menu.addAction(
            "&Paint Exclusion Zones"
        )
        self._act_paint_excl.setShortcut(QKeySequence("P"))
        self._act_paint_excl.triggered.connect(self._on_paint_exclusion)

        self._act_clear_excl = tools_menu.addAction(
            "C&lear Exclusion Zones"
        )
        self._act_clear_excl.triggered.connect(self._on_clear_exclusion)

   
    # Toolbar
   

    @staticmethod
    def _make_icon(draw_func, size: int = 28) -> QIcon:
        """Sukuria ikoną iš piešimo funkcijos."""
        pix = QPixmap(size, size)
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        draw_func(p, size)
        p.end()
        return QIcon(pix)

    def _setup_toolbar(self):
        """Pagrindinis įrankių juosta virš vaizdo."""
        tb = QToolBar("Tools")
        tb.setIconSize(QSize(24, 24))
        tb.setMovable(False)
        tb.setStyleSheet("""
            QToolBar {
                background: #2d2d30;
                border-bottom: 1px solid #444;
                spacing: 2px;
                padding: 2px 4px;
            }
            QToolButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 3px 5px;
                color: #ccc;
            }
            QToolButton:hover {
                background: #3e3e42;
                border: 1px solid #555;
            }
            QToolButton:checked {
                background: #094771;
                border: 1px solid #1177bb;
            }
            QToolButton:pressed {
                background: #0a5a8a;
            }
        """)
        self.addToolBar(Qt.TopToolBarArea, tb)

        # --- File ---
        tb.addAction(self._make_tb_action(
            tb, self._icon_folder, "Open Image (Ctrl+O)",
            self._on_open_image
        ))

        tb.addSeparator()

        # --- Mode toggles (radio-like: tik vienas aktyvus) ---
        self._tb_view = self._make_tb_action(
            tb, self._icon_hand, "Navigate — drag to pan, click to select",
            lambda: self._set_viewer_mode(Mode.VIEW),
            checkable=True
        )
        tb.addAction(self._tb_view)

        self._tb_measure = self._make_tb_action(
            tb, self._icon_crosshair, "Measure — draw lines across vessels",
            lambda: self._set_viewer_mode(Mode.MEASURE),
            checkable=True, checked=True
        )
        tb.addAction(self._tb_measure)

        self._tb_paint = self._make_tb_action(
            tb, self._icon_brush,
            "Paint Exclusion Zones (P)\n"
            "LMB = paint, Shift+LMB = erase, Scroll = brush size",
            self._on_paint_exclusion,
            checkable=True
        )
        tb.addAction(self._tb_paint)

        tb.addSeparator()

        # --- OD ---
        tb.addAction(self._make_tb_action(
            tb, self._icon_od_auto, "Auto Detect Optic Disc (D)",
            self._on_auto_od
        ))
        tb.addAction(self._make_tb_action(
            tb, self._icon_od_manual,
            "Manual OD — draw freehand lasso (Shift+D)",
            self._on_manual_od
        ))

        tb.addSeparator()

        # --- View toggles ---
        self._tb_clahe = self._make_tb_action(
            tb, self._icon_clahe, "Toggle CLAHE enhancement (C)",
            self._on_toggle_clahe, checkable=True
        )
        tb.addAction(self._tb_clahe)

        self._tb_green = self._make_tb_action(
            tb, self._icon_green, "Toggle Green channel (G)",
            self._on_toggle_green, checkable=True
        )
        tb.addAction(self._tb_green)

        self._tb_zones = self._make_tb_action(
            tb, self._icon_zones, "Toggle Exclusion overlay (R)",
            self._on_toggle_zones, checkable=True
        )
        tb.addAction(self._tb_zones)

        self._tb_od_circles = self._make_tb_action(
            tb, self._icon_circles, "Toggle OD zone circles (Z)",
            self._on_toggle_od_circles, checkable=True
        )
        tb.addAction(self._tb_od_circles)

        self._tb_reference = self._make_tb_action(
            tb, self._icon_reference, "Toggle Expert reference overlay (E)",
            self._on_toggle_reference, checkable=True, checked=True
        )
        tb.addAction(self._tb_reference)

        tb.addSeparator()

        # --- Fit / Undo / Redo ---
        tb.addAction(self._make_tb_action(
            tb, self._icon_fit, "Fit to Window (F)",
            self._on_fit_view
        ))

        tb.addSeparator()

        tb.addAction(self._make_tb_action(
            tb, self._icon_undo, "Undo (Ctrl+Z)",
            self._on_undo
        ))
        tb.addAction(self._make_tb_action(
            tb, self._icon_redo, "Redo (Ctrl+Y)",
            self._on_redo
        ))

        tb.addSeparator()

        # --- Export ---
        tb.addAction(self._make_tb_action(
            tb, self._icon_export, "Export Results (Ctrl+E)",
            self._on_export_results
        ))

        self._toolbar = tb

    def _make_tb_action(self, parent, icon_func, tooltip, slot,
                         checkable=False, checked=False) -> QAction:
        """Sukuria toolbar veiksmą su ikona ir tooltip."""
        icon = self._make_icon(icon_func)
        action = QAction(icon, "", parent)
        action.setToolTip(tooltip)
        action.triggered.connect(slot)
        if checkable:
            action.setCheckable(True)
            action.setChecked(checked)
        return action

    # --- Toolbar handler wrappers ---

    def _on_toggle_zones(self):
        self._viewer.toggle_exclusion_zones()
        if hasattr(self, '_tb_zones'):
            self._tb_zones.setChecked(self._viewer._show_exclusion_zones)

    def _on_toggle_od_circles(self):
        self._viewer.toggle_od_circles()
        if hasattr(self, '_tb_od_circles'):
            self._tb_od_circles.setChecked(self._viewer._show_od_circles)

    def _on_toggle_reference(self):
        self._viewer.toggle_reference()
        if hasattr(self, '_tb_reference'):
            self._tb_reference.setChecked(self._viewer._show_reference)

    def _on_fit_view(self):
        self._viewer.fit_to_window()


    def _set_viewer_mode(self, mode: Mode):
        """Nustato viewer režimą ir sinchronizuoja toolbar."""
        self._viewer.set_mode(mode)
        self._sync_toolbar_checks()

    def _sync_toolbar_checks(self):
        """Sinchronizuoja toolbar mygtukus su viewer state."""
        if not hasattr(self, '_tb_measure'):
            return
        mode = self._viewer.get_mode()
        self._tb_view.setChecked(mode == Mode.VIEW)
        self._tb_measure.setChecked(mode == Mode.MEASURE)
        self._tb_paint.setChecked(mode == Mode.PAINT_EXCLUSION)

   
    # Ikonų piešimo funkcijos (28×28 QPainter)
   

    @staticmethod
    def _icon_folder(p: QPainter, s: int):
        c = QColor(220, 180, 80)
        p.setPen(QPen(c, 1.5))
        p.setBrush(QBrush(c.darker(130)))
        p.drawRoundedRect(3, 8, s - 6, s - 12, 2, 2)
        p.setBrush(QBrush(c))
        tab = QPainterPath()
        tab.moveTo(3, 10)
        tab.lineTo(3, 6)
        tab.lineTo(10, 6)
        tab.lineTo(12, 8)
        tab.lineTo(12, 10)
        tab.closeSubpath()
        p.drawPath(tab)


    @staticmethod
    def _icon_hand(p: QPainter, s: int):
        """Ranka — pan/navigate režimas."""
        c = QColor(200, 200, 220)
        p.setPen(QPen(c, 1.5))
        p.setBrush(QBrush(c.darker(140)))
        # Delnas
        p.drawRoundedRect(7, 11, 14, 13, 3, 3)
        # Pirštai (4 vertikalūs stačiakampiai)
        p.setBrush(QBrush(c.darker(120)))
        for i, x in enumerate([8, 12, 16, 20]):
            h = 8 if i in (1, 2) else 6
            p.drawRoundedRect(x, 11 - h, 3, h + 2, 1, 1)
        # Nykštys
        p.drawRoundedRect(4, 13, 5, 3, 1, 1)

    @staticmethod
    def _icon_crosshair(p: QPainter, s: int):
        c = QColor(100, 220, 100)
        p.setPen(QPen(c, 2))
        mid = s / 2
        p.drawLine(QPointF(mid, 4), QPointF(mid, s - 4))
        p.drawLine(QPointF(4, mid), QPointF(s - 4, mid))
        p.setPen(QPen(c, 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(mid, mid), 7, 7)

    @staticmethod
    def _icon_brush(p: QPainter, s: int):
        c = QColor(255, 100, 100)
        p.setPen(QPen(QColor(180, 150, 100), 3))
        p.drawLine(QPointF(s - 6, 6), QPointF(12, 16))
        p.setPen(QPen(c, 2))
        p.setBrush(QBrush(c.darker(130)))
        p.drawEllipse(QPointF(9, 19), 5, 4)

    @staticmethod
    def _icon_od_auto(p: QPainter, s: int):
        c = QColor(100, 200, 255)
        mid = s / 2
        p.setPen(QPen(c, 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(mid, mid), 9, 7)
        p.setBrush(QBrush(c.darker(150)))
        p.drawEllipse(QPointF(mid, mid), 4, 4)
        p.setPen(QPen(QColor(255, 220, 50), 2))
        p.drawLine(QPointF(mid + 2, 3), QPointF(mid - 1, mid))
        p.drawLine(QPointF(mid - 1, mid), QPointF(mid + 2, mid))
        p.drawLine(QPointF(mid + 2, mid), QPointF(mid - 1, s - 3))

    @staticmethod
    def _icon_od_manual(p: QPainter, s: int):
        c = QColor(100, 200, 255)
        mid = s / 2
        pen = QPen(c, 1.5, Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(mid, mid), 9, 9)
        p.setPen(QPen(QColor(255, 200, 150), 2))
        p.drawLine(QPointF(mid, mid + 4), QPointF(mid, mid - 3))
        p.drawLine(QPointF(mid - 3, mid + 1), QPointF(mid, mid - 3))
        p.drawLine(QPointF(mid + 3, mid + 1), QPointF(mid, mid - 3))

    @staticmethod
    def _icon_clahe(p: QPainter, s: int):
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(80, 80, 80)))
        p.drawRect(4, 6, (s - 8) // 2, s - 12)
        p.setBrush(QBrush(QColor(220, 220, 220)))
        p.drawRect(4 + (s - 8) // 2, 6, (s - 8) // 2, s - 12)
        p.setPen(QPen(QColor(255, 200, 50), 2))
        font = QFont("Arial", 10, QFont.Bold)
        p.setFont(font)
        p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, "C")

    @staticmethod
    def _icon_green(p: QPainter, s: int):
        c = QColor(50, 200, 50)
        p.setPen(QPen(c, 2))
        font = QFont("Arial", 12, QFont.Bold)
        p.setFont(font)
        p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, "G")
        p.setPen(QPen(c.darker(130), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(3, 3, s - 6, s - 6, 3, 3)

    @staticmethod
    def _icon_zones(p: QPainter, s: int):
        c = QColor(255, 80, 80)
        p.setPen(QPen(c, 1.5))
        p.setBrush(QBrush(QColor(255, 60, 60, 60)))
        p.drawRoundedRect(4, 4, s - 8, s - 8, 3, 3)
        p.setPen(QPen(c, 2))
        font = QFont("Arial", 11, QFont.Bold)
        p.setFont(font)
        p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, "R")

    @staticmethod
    def _icon_circles(p: QPainter, s: int):
        c = QColor(180, 180, 180)
        mid = s / 2
        p.setPen(QPen(c, 1))
        p.setBrush(Qt.NoBrush)
        for r in [4, 7, 10]:
            p.drawEllipse(QPointF(mid, mid), r, r)
        p.setPen(QPen(QColor(255, 255, 255), 1.5))
        font = QFont("Arial", 8, QFont.Bold)
        p.setFont(font)
        p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, "Z")

    @staticmethod
    def _icon_reference(p: QPainter, s: int):
        """Eksperto reference — dvi brūkšninės linijos."""
        c = QColor(255, 200, 60, 200)
        pen = QPen(c, 2, Qt.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(4, 10), QPointF(s - 4, 10))
        c2 = QColor(60, 220, 255, 200)
        pen2 = QPen(c2, 2, Qt.DashLine)
        p.setPen(pen2)
        p.drawLine(QPointF(4, 18), QPointF(s - 4, 18))
        # E raidė
        p.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
        font = QFont("Arial", 8, QFont.Bold)
        p.setFont(font)
        p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, "E")

    @staticmethod
    def _icon_fit(p: QPainter, s: int):
        c = QColor(200, 200, 200)
        p.setPen(QPen(c, 2))
        d = 7
        p.drawLine(QPointF(3, 3 + d), QPointF(3, 3))
        p.drawLine(QPointF(3, 3), QPointF(3 + d, 3))
        p.drawLine(QPointF(s - 3 - d, 3), QPointF(s - 3, 3))
        p.drawLine(QPointF(s - 3, 3), QPointF(s - 3, 3 + d))
        p.drawLine(QPointF(3, s - 3 - d), QPointF(3, s - 3))
        p.drawLine(QPointF(3, s - 3), QPointF(3 + d, s - 3))
        p.drawLine(QPointF(s - 3 - d, s - 3), QPointF(s - 3, s - 3))
        p.drawLine(QPointF(s - 3, s - 3), QPointF(s - 3, s - 3 - d))

    @staticmethod
    def _icon_undo(p: QPainter, s: int):
        c = QColor(180, 200, 220)
        p.setPen(QPen(c, 2))
        path = QPainterPath()
        path.moveTo(s - 8, 10)
        path.cubicTo(s - 8, 6, 8, 6, 8, 12)
        p.drawPath(path)
        p.drawLine(QPointF(8, 12), QPointF(4, 9))
        p.drawLine(QPointF(8, 12), QPointF(11, 8))
        p.drawLine(QPointF(s - 8, 10), QPointF(s - 8, s - 8))

    @staticmethod
    def _icon_redo(p: QPainter, s: int):
        c = QColor(180, 200, 220)
        p.setPen(QPen(c, 2))
        path = QPainterPath()
        path.moveTo(8, 10)
        path.cubicTo(8, 6, s - 8, 6, s - 8, 12)
        p.drawPath(path)
        p.drawLine(QPointF(s - 8, 12), QPointF(s - 4, 9))
        p.drawLine(QPointF(s - 8, 12), QPointF(s - 11, 8))
        p.drawLine(QPointF(8, 10), QPointF(8, s - 8))

    @staticmethod
    def _icon_export(p: QPainter, s: int):
        """Export — dėžutė su rodykle aukštyn."""
        c = QColor(120, 220, 120)
        p.setPen(QPen(c, 2))
        # Dėžutės apačia
        p.drawLine(QPointF(5, 14), QPointF(5, s - 4))
        p.drawLine(QPointF(5, s - 4), QPointF(s - 5, s - 4))
        p.drawLine(QPointF(s - 5, s - 4), QPointF(s - 5, 14))
        # Rodyklė aukštyn
        mid = s / 2
        p.drawLine(QPointF(mid, 3), QPointF(mid, 16))
        p.drawLine(QPointF(mid, 3), QPointF(mid - 5, 9))
        p.drawLine(QPointF(mid, 3), QPointF(mid + 5, 9))

    def _setup_statusbar(self):
        """Status bar."""
        self._status_image = QLabel("No image")
        self._status_zoom = QLabel("Zoom: 100%")
        self._status_meas = QLabel("Meas: 0")
        self._status_undo = QLabel("Undo:0/Redo:0")
        self._status_mode = QLabel("Mode: Measure")

        sb = self.statusBar()
        sb.addWidget(self._status_image, 1)
        sb.addPermanentWidget(self._status_zoom)
        sb.addPermanentWidget(self._status_meas)
        sb.addPermanentWidget(self._status_undo)
        sb.addPermanentWidget(self._status_mode)

    def _setup_shortcuts(self):
        """Papildomi shortkeys (kurie netelpa į meniu)."""
        pass  # Visi pagrindiniai shortkeys jau meniu

    def _connect_signals(self):
        """Sujungia signalus tarp komponentų."""
        # Viewer → Main
        self._viewer.measurement_added.connect(self._on_measurement_added)
        self._viewer.measurement_selected.connect(
            self._on_measurement_selected
        )
        self._viewer.od_manual_finished.connect(self._on_od_lasso_finished)
        self._viewer.redraw_finished.connect(self._on_redraw_finished)
        self._viewer.view_changed.connect(self._update_status)
        self._viewer.mode_changed.connect(self._on_mode_changed)
        self._viewer.exclusion_painted.connect(self._on_exclusion_painted)
        self._viewer.exclusion_stroke_started.connect(
            self._exclusion.save_snapshot
        )

        # Perduoti exclusion computer viewer'iui
        self._viewer.set_exclusion_computer(self._exclusion)

        # Panel → Main
        self._panel.measurement_selected.connect(
            self._on_panel_measurement_selected
        )
        self._panel.measurement_deleted.connect(self._on_delete_measurement)
        self._panel.type_changed.connect(self._on_type_changed)
        self._panel.vessel_id_changed.connect(self._on_vessel_id_changed)
        self._panel.edges_adjusted.connect(self._on_edges_adjusted)
        self._panel.edit_requested.connect(self._on_edit_requested)
        self._panel.redraw_requested.connect(self._on_redraw_requested)

   
    # Dark theme
   

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QMenuBar { background: #2D2D2D; color: #FFFFFF; }
            QMenuBar::item:selected { background: #2266CC; color: #FFFFFF; }
            QMenu { background: #1E1E1E; color: #FFFFFF; border: 1px solid #555; }
            QMenu::item { padding: 4px 20px; }
            QMenu::item:selected { background: #2266CC; color: #FFFFFF; }
            QMenu::separator { background: #444; height: 1px; margin: 4px 8px; }
            QStatusBar { background: #2D2D2D; color: #E0E0E0; }
            QLabel { color: #E0E0E0; }
            QSplitter::handle { background: #444; width: 3px; }
            QScrollBar:vertical {
                background: #1E1E1E; width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #555; min-height: 20px; border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

   
    # File menu handlers
   

    def _on_open_image(self):
        """File → Open Image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Fundus Image", "", IMAGE_FILTERS
        )
        if not path:
            return

        self._load_image(path)

    def _save_current_exclusion(self):
        """Išsaugo dabartinę exclusion kaukę į manager."""
        if self._current_image_name and self._exclusion.has_zones:
            self._manager.store_exclusion_mask(
                self._current_image_name,
                self._exclusion.serialize_mask()
            )

    def _load_image(self, path: str):
        """Įkelia vaizdą ir paruošia komponentus."""
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            QMessageBox.warning(self, "Error", f"Cannot load: {path}")
            return

        # --- Išsaugoti SENO vaizdo duomenis prieš keičiant ---
        self._save_current_exclusion()

        # --- Naujas vaizdas ---
        self._current_image_name = os.path.basename(path)
        self._current_image_bgr = image_bgr
        self._current_green = get_green_channel(image_bgr)

        # Registruoti pilną kelią sesijos atkūrimui
        self._manager.register_image_path(
            self._current_image_name, os.path.abspath(path)
        )

        # Kaukė ir apdorotas kanalas — bus skaičiuojami lazy
        self._current_mask = None
        self._current_sc = None
        self._current_green_processed = None

        # CLAHE versija atvaizdavimui (greita operacija)
        clahe_bgr = apply_clahe_for_display(image_bgr)

        # Viewer
        self._viewer.set_image(image_bgr, clahe_bgr)

        # Matavimai — jei jau buvo šiam vaizdui
        self._refresh_viewer_measurements()
        self._refresh_panel()

        # OD — atkurti arba išvalyti
        od_data = self._manager.get_od_data(self._current_image_name)
        if od_data[2] > 0:
            self._viewer.set_od(*od_data)
            self._od_detector.set_manual(*od_data)
        else:
            # Išvalyti seną OD
            self._viewer.set_od(0, 0, 0)

        # Exclusion — atkurti arba išvalyti
        self._exclusion.reset()

        excl_data = self._manager.get_exclusion_mask(
            self._current_image_name
        )
        if excl_data is not None:
            h, w = image_bgr.shape[:2]
            self._exclusion.init_mask(h, w)
            self._exclusion.deserialize_mask(excl_data)
            self._viewer.set_exclusion_overlays(
                self._exclusion.overlay_rgba, None
            )
        else:
            self._viewer.set_exclusion_overlays(None, None)

        # Eksperto reference
        self._update_reference_for_current_image()

        self._update_status()

    def _ensure_processed_green(self) -> bool:
        """
        Lazy skaičiavimas: kaukė + preprocessing3.
        Kviečiama tik kai reikia pirmo matavimo.
        Rodo progress dialogą nes užtrunka ~10-30s.

        Returns:
            True jei pavyko
        """
        if self._current_green_processed is not None:
            return True

        if self._current_image_bgr is None:
            return False

        progress = QProgressDialog(
            "Preprocessing image for measurements...", None, 0, 0, self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()

        try:
            h, w = self._current_image_bgr.shape[:2]

            # Dideliam vaizdui (>2000px pločio) — CLAHE be denoising
            # nes fastNlMeansDenoising crashina su dideliais vaizdais
            if max(h, w) > 2000:
                print(f"[Preprocessing] Large image ({w}x{h}), "
                      f"using CLAHE-only (skipping denoising)", flush=True)
                clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
                green = self._current_image_bgr[:, :, 1]

                # Kaukė
                self._current_mask, self._current_sc = create_fundus_mask(
                    green
                )

                # CLAHE × 2 (kaip preprocessing3, bet be denoising)
                enhanced = clahe.apply(green)
                enhanced = clahe.apply(enhanced)

                # Normalizuoti kaukės viduje
                if self._current_mask is not None:
                    mask_bool = self._current_mask > 0
                    vals = enhanced[mask_bool]
                    if len(vals) > 0:
                        mn, mx = vals.min(), vals.max()
                        if mx > mn:
                            enhanced = np.clip(
                                (enhanced.astype(float) - mn) / (mx - mn) * 255,
                                0, 255
                            ).astype(np.uint8)

                self._current_green_processed = enhanced
            else:
                # Normalus preprocessing3
                self._current_mask, self._current_sc = create_fundus_mask(
                    self._current_green
                )
                img_processed = preprocessing3(
                    self._current_image_bgr, self._current_mask
                )
                self._current_green_processed = img_processed[:, :, 1]

        except Exception as e:
            print(f"[Preprocessing] Error: {e}", flush=True)
            # Fallback — raw žalias kanalas
            self._current_green_processed = self._current_green
        finally:
            progress.close()

        return self._current_green_processed is not None

    def _on_import_reference(self):
        """File → Import Expert Measurements.

        JSON: pilnai atkuria — įkelia nuotrauką, OD, exclusion kaukę,
              matavimus rodo kaip reference overlay.
        CSV:  tik reference overlay ant dabartinio vaizdo.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Expert Measurements", "",
            "Session JSON (*.json);;Detailed CSV (*.csv);;All (*.*)"
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()

        # --- CSV: tik reference overlay ---
        if ext in ('.csv', '.txt'):
            ref_data = import_reference_csv(path)
            if not ref_data:
                QMessageBox.warning(
                    self, "Import Error",
                    "No measurements found in CSV."
                )
                return
            self._reference_data = ref_data
            total = sum(len(v) for v in ref_data.values())
            self.statusBar().showMessage(
                f"Imported {total} reference measurements", 5000
            )
            self._update_reference_for_current_image()
            return

        # --- JSON: pilna sesijos atkūrimas ---
        try:
            with open(path, 'r', encoding='utf-8') as f:
                session = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.warning(
                self, "Import Error", f"Cannot read JSON:\n{e}"
            )
            return

        # 1. Surasti ir įkrauti originalią nuotrauką
        image_paths = session.get('image_paths', {})
        session_dir = os.path.dirname(os.path.abspath(path))

        # Surinkti visus vaizdus session folderyje
        img_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.ppm')
        local_files = []
        try:
            local_files = [
                f for f in os.listdir(session_dir)
                if f.lower().endswith(img_exts)
            ]
        except OSError:
            pass

        loaded_image_path = None
        loaded_image_name = None

        # Pirmasis image_name iš session (dažniausiai vienintelis)
        first_session_name = next(iter(
            session.get('measurements', {})
        ), next(iter(image_paths), None))

        if first_session_name:
            base = os.path.splitext(first_session_name)[0]

            for candidate in local_files:
                # Prioritetas: *_original.* > tikslus pavadinimas > base match
                if candidate.startswith(base + '_original'):
                    loaded_image_path = os.path.join(
                        session_dir, candidate
                    )
                    loaded_image_name = first_session_name
                    break

            # Jei nerado _original — bandyti tikslų pavadinimą
            if loaded_image_path is None:
                for candidate in local_files:
                    if candidate == first_session_name:
                        loaded_image_path = os.path.join(
                            session_dir, candidate
                        )
                        loaded_image_name = first_session_name
                        break

            # Bandyti originalų kelią iš image_paths
            if loaded_image_path is None:
                orig = image_paths.get(first_session_name, '')
                if orig and os.path.exists(orig):
                    loaded_image_path = orig
                    loaded_image_name = first_session_name

        # Jei vis dar nerasta — paklausti vartotojo
        if loaded_image_path is None:
            img_path, _ = QFileDialog.getOpenFileName(
                self, "Select Original Image",
                session_dir,
                "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.ppm)"
            )
            if img_path:
                loaded_image_path = img_path
                loaded_image_name = (
                    first_session_name
                    or os.path.basename(img_path)
                )
            else:
                QMessageBox.warning(
                    self, "Import Error",
                    "Cannot find original image for this session."
                )
                return

        # 2. Įkrauti nuotrauką
        self._load_image(loaded_image_path)

        if self._current_image_bgr is None:
            return

        # Perrašyti image name kad atitiktų sesijos raktus
        # (_load_image nustato "01_h_original.tif", bet sesijoje "01_h.tif")
        if (loaded_image_name and
                loaded_image_name != self._current_image_name):
            self._current_image_name = loaded_image_name
            self._manager.register_image_path(
                loaded_image_name, os.path.abspath(loaded_image_path)
            )

        # 3. Atkurti OD
        od_section = session.get('optic_disc', {})
        od_entry = od_section.get(
            loaded_image_name,
            od_section.get(
                next(iter(od_section), ''), {}
            )
        )
        if isinstance(od_entry, dict) and od_entry.get('r', 0) > 0:
            ox = int(od_entry.get('x', 0))
            oy = int(od_entry.get('y', 0))
            r = int(od_entry.get('r', 0))
            self._viewer.set_od(ox, oy, r)
            self._manager.set_od_data(
                self._current_image_name, ox, oy, r
            )
            self._od_detector.set_manual(ox, oy, r)

        # 4. Atkurti exclusion kaukę
        excl_section = session.get('exclusion_masks', {})
        excl_rle = excl_section.get(
            loaded_image_name,
            excl_section.get(next(iter(excl_section), ''))
        )
        h, w = self._current_image_bgr.shape[:2]

        if excl_rle is not None:
            self._exclusion.init_mask(h, w)
            ok = self._exclusion.deserialize_mask(excl_rle)
            if ok:
                self._viewer.set_exclusion_overlays(
                    self._exclusion.overlay_rgba, None
                )
                self._manager.store_exclusion_mask(
                    self._current_image_name, excl_rle
                )
        else:
            # Fallback: bandyti *_exclusion.png iš to paties folderio
            base = os.path.splitext(loaded_image_name)[0]
            for f in local_files:
                if f.startswith(base + '_exclusion') and f.endswith('.png'):
                    excl_path = os.path.join(session_dir, f)
                    mask = cv2.imread(excl_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        # Resize jei reikia
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask, (w, h),
                                              interpolation=cv2.INTER_NEAREST)
                        _, mask = cv2.threshold(
                            mask, 127, 255, cv2.THRESH_BINARY
                        )
                        self._exclusion.init_mask(h, w)
                        self._exclusion._mask = mask
                        self._exclusion.rebuild_overlay()
                        self._viewer.set_exclusion_overlays(
                            self._exclusion.overlay_rgba, None
                        )
                    break

        # 5. Matavimai kaip reference overlay
        ref_data = import_reference_session(path)
        if ref_data:
            self._reference_data = ref_data
            self._update_reference_for_current_image()

        # Suvestinė
        n_meas = sum(len(v) for v in ref_data.values()) if ref_data else 0
        has_od = od_entry.get('r', 0) > 0 if isinstance(od_entry, dict) else False
        has_excl = self._exclusion.has_zones

        parts = [f"{n_meas} measurements"]
        if has_od:
            parts.append("OD")
        if has_excl:
            parts.append("exclusion mask")

        self.statusBar().showMessage(
            f"Expert session loaded: {', '.join(parts)}", 5000
        )

    def _update_reference_for_current_image(self):
        """Atnaujina viewer reference sluoksnį dabartiniam vaizdui."""
        if not self._current_image_name:
            self._viewer.set_reference_measurements([])
            return

        refs = self._reference_data.get(self._current_image_name, [])

        # Bandyti be plėtinio jei nerasta (01_h.tif vs 01_h)
        if not refs:
            base = os.path.splitext(self._current_image_name)[0]
            for key in self._reference_data:
                if os.path.splitext(key)[0] == base:
                    refs = self._reference_data[key]
                    break

        self._viewer.set_reference_measurements(refs)

        if refs and hasattr(self, '_tb_reference'):
            self._tb_reference.setChecked(True)
            self._viewer._show_reference = True

   
    # Navigacija tarp vaizdų
   

    def keyPressEvent(self, event):
        """Tipo shortkeys ir kiti."""
        key = event.key()

        # Tipo priskyrimas (bet kuriuo metu)
        if key in (Qt.Key_1, Qt.Key_A):
            self._assign_type(VESSEL_ARTERY)
            return
        if key == Qt.Key_2:
            self._assign_type(VESSEL_VEIN)
            return
        if key in (Qt.Key_0, Qt.Key_U):
            self._assign_type(VESSEL_UNKNOWN)
            return

        # Edit selected
        if key == Qt.Key_E:
            if self._viewer._selected_id is not None:
                self._on_edit_requested(self._viewer._selected_id)
            return

        # Zoom
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self._viewer.zoom_in()
            return
        if key == Qt.Key_Minus:
            self._viewer.zoom_out()
            return

        # Redo alternatyva
        if (key == Qt.Key_Z and
                event.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier)):
            self._on_redo()
            return

        # Escape — cancel mode
        if key == Qt.Key_Escape:
            self._viewer.cancel_mode()
            return

        super().keyPressEvent(event)


   
    # Measurement handlers
   

    def _on_measurement_added(self, x1: float, y1: float,
                               x2: float, y2: float):
        """Viewer signalas: nauja matavimo linija nubrėžta."""
        if self._current_image_name is None:
            return

        self._exported_images.discard(self._current_image_name)

        if not self._ensure_processed_green():
            return

        # Paveldėti vessel_id iš paskutinio matavimo šiame vaizde
        existing = self._manager.get_measurements(self._current_image_name)
        last_vid = existing[-1]['vessel_id'] if existing else None

        try:
            m = self._manager.add_measurement(
                self._current_image_name,
                x1, y1, x2, y2,
                self._current_green_processed,
                vessel_id=last_vid,
            )
        except Exception as e:
            print(f"[Measurement] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return

        if m is not None:
            self._panel.add_card(m)
            self._refresh_viewer_measurements()
            self._refresh_vessel_options()
            self._viewer.set_selected_measurement(m['id'])
            self._panel.highlight_card(m['id'])

        self._update_status()

    def _on_measurement_selected(self, measurement_id: int):
        """Viewer: matavimas paspaustas."""
        self._viewer.set_selected_measurement(measurement_id)
        self._panel.highlight_card(measurement_id)

        # Centruoti
        m = self._manager._find(self._current_image_name, measurement_id)
        if m:
            self._viewer.center_on_image_point(m['cx'], m['cy'])

    def _on_panel_measurement_selected(self, measurement_id: int):
        """Panel: kortelė paspausta → centruoti viewer."""
        self._viewer.set_selected_measurement(measurement_id)

        m = self._manager._find(self._current_image_name, measurement_id)
        if m:
            self._viewer.center_on_image_point(m['cx'], m['cy'])

    def _on_delete_measurement(self, measurement_id: int):
        """Panel: Delete mygtukas."""
        if self._current_image_name is None:
            return

        self._exported_images.discard(self._current_image_name)
        self._manager.delete_measurement(
            self._current_image_name, measurement_id
        )
        self._panel.remove_card(measurement_id)
        self._refresh_viewer_measurements()
        self._refresh_vessel_options()
        self._update_status()

    def _on_delete_selected(self):
        """Edit → Delete Selected."""
        sel = self._viewer._selected_id
        if sel is not None:
            self._on_delete_measurement(sel)

    def _on_clear_all(self):
        """Edit → Clear All (su patvirtinimu)."""
        if self._current_image_name is None:
            return

        reply = QMessageBox.question(
            self, "Clear All",
            "Remove all measurements for this image?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._manager.clear_all(self._current_image_name)
            self._panel.clear_cards()
            self._refresh_viewer_measurements()
            self._update_status()

    def _on_type_changed(self, measurement_id: int, new_type: int):
        """Panel: tipo mygtukas."""
        if self._current_image_name is None:
            return

        self._manager.update_type(
            self._current_image_name, measurement_id, new_type
        )
        self._refresh_card(measurement_id)
        self._refresh_viewer_measurements()
        self._update_status()

    def _on_vessel_id_changed(self, measurement_id: int,
                               new_vessel_id: int):
        """Panel: vessel dropdown."""
        if self._current_image_name is None:
            return

        # "New" → priskirti mažiausią nenaudojamą vessel ID
        if new_vessel_id == -1:
            new_vessel_id = self._manager._next_vessel_id(
                self._current_image_name
            )

        self._exported_images.discard(self._current_image_name)
        self._manager.update_vessel_id(
            self._current_image_name, measurement_id, new_vessel_id
        )
        self._refresh_card(measurement_id)
        self._refresh_vessel_options()
        self._update_status()

    def _on_edges_adjusted(self, measurement_id: int,
                            left: float, right: float):
        """Panel: Adjust Edges markeriai."""
        if self._current_image_name is None:
            return

        self._manager.update_edges(
            self._current_image_name, measurement_id, left, right
        )
        self._refresh_card(measurement_id)
        self._refresh_viewer_measurements()

    def _assign_type(self, vessel_type: int):
        """Tipo priskyrimas shortkey (1/2/0)."""
        sel = self._viewer._selected_id
        if sel is not None and self._current_image_name:
            self._on_type_changed(sel, vessel_type)

   
    # Edit handlers
   

    def _on_edit_requested(self, measurement_id: int):
        """Panel: Edit mygtukas."""
        self._panel.set_card_editing(measurement_id, True)
        self._viewer.set_selected_measurement(measurement_id)

    def _on_redraw_requested(self, measurement_id: int):
        """Panel: Redraw Line mygtukas."""
        if self._current_image_name is None:
            return

        m = self._manager._find(self._current_image_name, measurement_id)
        if m is None:
            return

        ghost = {
            'x1': m['x1'], 'y1': m['y1'],
            'x2': m['x2'], 'y2': m['y2'],
        }
        self._viewer.start_redraw(measurement_id, ghost)

    def _on_redraw_finished(self, measurement_id: int,
                             x1: float, y1: float,
                             x2: float, y2: float):
        """Viewer: Redraw Line baigta."""
        if self._current_image_name is None:
            return

        if not self._ensure_processed_green():
            return

        try:
            self._manager.update_line(
                self._current_image_name, measurement_id,
                x1, y1, x2, y2, self._current_green_processed
            )
        except Exception as e:
            print(f"[Redraw] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return
        self._refresh_card(measurement_id)
        self._refresh_viewer_measurements()
        self._panel.set_card_editing(measurement_id, False)
        self._update_status()

   
    # Undo / Redo
   

    def _on_undo(self):
        # Exclusion paint mode — atkurti ankstesnį potėpį
        if (self._viewer.get_mode() == Mode.PAINT_EXCLUSION
                and self._exclusion.can_undo):
            self._exclusion.undo()
            self._viewer.set_exclusion_overlays(
                self._exclusion.overlay_rgba, None
            )
            self._exported_images.discard(self._current_image_name)
            self._update_status()
            return

        # Matavimų undo
        if self._manager.undo():
            self._exported_images.discard(self._current_image_name)
            self._refresh_panel()
            self._refresh_viewer_measurements()
            self._update_status()

    def _on_redo(self):
        # Exclusion paint mode — grąžinti atšauktą potėpį
        if (self._viewer.get_mode() == Mode.PAINT_EXCLUSION
                and self._exclusion.can_redo):
            self._exclusion.redo()
            self._viewer.set_exclusion_overlays(
                self._exclusion.overlay_rgba, None
            )
            self._exported_images.discard(self._current_image_name)
            self._update_status()
            return

        # Matavimų redo
        if self._manager.redo():
            self._exported_images.discard(self._current_image_name)
            self._refresh_panel()
            self._refresh_viewer_measurements()
            self._update_status()

   
    # OD handlers
   

    def _on_auto_od(self):
        """Tools → Auto Detect OD."""
        if self._current_image_bgr is None:
            return

        # Progress dialogas (gali užtrukti)
        progress = QProgressDialog(
            "Detecting optic disc...", None, 0, 0, self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()

        result = self._od_detector.auto_detect(self._current_image_bgr)

        progress.close()

        if result.is_valid:
            self._viewer.set_od(result.x, result.y, result.r)
            self._manager.set_od_data(
                self._current_image_name, result.x, result.y, result.r
            )
            self.statusBar().showMessage(
                f"OD detected: ({result.x}, {result.y}) r={result.r}", 3000
            )
        else:
            QMessageBox.warning(
                self, "OD Detection",
                "Could not detect optic disc.\n"
                "Use Shift+D for manual detection."
            )

    def _on_manual_od(self):
        """Tools → Manual OD (freehand lasso)."""
        self._viewer.start_od_manual()
        self.statusBar().showMessage(
            "Draw a freehand circle around the optic disc...", 5000
        )

    def _on_od_lasso_finished(self, points: list):
        """Viewer: freehand lasso baigta."""
        try:
            result = self._od_detector.set_from_contour(points)
            self._viewer.set_od(result.x, result.y, result.r)
            self._manager.set_od_data(
                self._current_image_name, result.x, result.y, result.r
            )
            self.statusBar().showMessage(
                f"Manual OD: ({result.x}, {result.y}) r={result.r}", 3000
            )
        except ValueError as e:
            self.statusBar().showMessage(str(e), 3000)

   
    # View toggles
   

    def _on_toggle_clahe(self):
        self._viewer.toggle_clahe()
        if hasattr(self, '_tb_clahe'):
            self._tb_clahe.setChecked(self._viewer._show_clahe)

    def _on_toggle_green(self):
        self._viewer.toggle_green_channel()
        if hasattr(self, '_tb_green'):
            self._tb_green.setChecked(self._viewer._show_green)

   
    # Exclusion zones
   

    def _on_paint_exclusion(self):
        """Tools → Paint Exclusion Zones (P)."""
        if self._current_image_bgr is None:
            return

        self._viewer.toggle_paint_exclusion()

        if self._viewer.get_mode() == Mode.PAINT_EXCLUSION:
            self._viewer._show_exclusion_zones = True
            self.statusBar().showMessage(
                "Paint mode: LMB=paint, Shift+LMB=erase, "
                "Scroll=brush size, P=exit", 5000
            )
        else:
            self.statusBar().showMessage("Paint mode off", 2000)

        self._sync_toolbar_checks()

    def _on_clear_exclusion(self):
        """Tools → Clear Exclusion Zones."""
        self._exclusion.clear()
        self._viewer.set_exclusion_overlays(
            self._exclusion.overlay_rgba, None
        )
        self._recheck_all_exclusions()

        # Pašalinti iš manager
        if self._current_image_name:
            self._manager.store_exclusion_mask(
                self._current_image_name, None
            )

        self.statusBar().showMessage("Exclusion zones cleared", 3000)

    def _on_exclusion_painted(self):
        """Viewer: piešimo brūkšnys baigtas (mouse release).

        Tik restartina debounce timer — sunkus darbas (recheck +
        serialize) vyksta _on_exclusion_debounced kai vartotojas
        sustoja 300ms.
        """
        self._exported_images.discard(self._current_image_name)
        self._viewer.update()
        self._excl_debounce.start()

    def _on_exclusion_debounced(self):
        """Debounced: perskaičiuoti exclusion status + serialize."""
        self._recheck_all_exclusions()

        if self._current_image_name:
            self._manager.store_exclusion_mask(
                self._current_image_name,
                self._exclusion.serialize_mask()
            )


    def _recheck_all_exclusions(self):
        """Perskaičiuoja exclusion status visiems matavimams."""
        if self._current_image_name is None:
            return

        for m in self._manager.get_measurements(self._current_image_name):
            self._manager._check_exclusion(m, self._current_image_name)

        self._refresh_panel()

   
    # Export handlers
   

    def _on_export_results(self):
        """File → Export Results (Ctrl+E)."""
        if self._current_image_name is None or self._current_image_bgr is None:
            QMessageBox.information(self, "Info", "No image loaded.")
            return

        meas = self._manager.measurements.get(
            self._current_image_name, []
        )
        if not meas:
            reply = QMessageBox.question(
                self, "No Measurements",
                "No measurements for this image.\n"
                "Export anyway (image + exclusion zones)?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Paklausti folderio (tik pirmą kartą)
        if self._results_dir is None:
            d = QFileDialog.getExistingDirectory(
                self, "Select Results Directory"
            )
            if not d:
                return
            self._results_dir = d

        self._export_current_image()

    def _export_current_image(self, silent: bool = False) -> bool:
        """
        Eksportuoja dabartinį vaizdą į results folderį.

        Args:
            silent: True = nerodo pranešimo (naudojama iš navigate)

        Returns:
            True jei pavyko
        """
        if (self._current_image_name is None or
                self._current_image_bgr is None or
                self._results_dir is None):
            return False

        # Surinkti duomenis
        self._save_current_exclusion()

        image_name = self._current_image_name
        meas = self._manager.measurements.get(image_name, [])
        od_data = self._manager.get_od_data(image_name)
        excl_mask = self._exclusion.get_mask()
        refs = self._reference_data.get(image_name, [])
        orig_path = self._manager.get_image_path(image_name)

        try:
            folder = export_image_results(
                output_dir=self._results_dir,
                image_name=image_name,
                image_bgr=self._current_image_bgr,
                measurements=meas,
                od_data=od_data,
                exclusion_mask=excl_mask,
                original_path=orig_path,
                reference_measurements=refs if refs else None,
                all_measurements=self._manager.measurements,
                all_od_data=self._manager._od_data,
                exclusion_masks_rle=self._manager._exclusion_masks,
                image_paths=self._manager.get_all_image_paths(),
            )
            if not silent:
                self.statusBar().showMessage(
                    f"Exported to {folder}", 5000
                )
            self._exported_images.add(image_name)
            return True

        except Exception as e:
            if not silent:
                QMessageBox.warning(
                    self, "Export Error", f"Export failed:\n{e}"
                )
            print(f"[Export] Error: {e}", flush=True)
            return False

    def _on_save_session(self):
        """File → Save Session."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "session.json",
            "JSON (*.json)"
        )
        if path:
            # Išsaugoti dabartinę exclusion kaukę prieš session save
            self._save_current_exclusion()
            ok = self._manager.save_session(path)
            if ok:
                self.statusBar().showMessage(f"Session saved to {path}", 3000)

    def _on_load_session(self):
        """File → Load Session."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "JSON (*.json)"
        )
        if not path:
            return

        ok = self._manager.load_session(path)
        if not ok:
            QMessageBox.warning(self, "Error", "Cannot load session.")
            return

        self._restore_session_images()
        self.statusBar().showMessage(f"Session loaded from {path}", 3000)

    def _check_autosave(self):
        """Tikrina ar yra autosave — pasiūlo atkurti."""
        if self._manager.has_autosave():
            reply = QMessageBox.question(
                self, "Restore Session",
                "Auto-saved session found.\nRestore previous session?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._manager.load_autosave()
                self._restore_session_images()

   
    # Session atkūrimas
   

    def _restore_session_images(self):
        """
        Atkuria vaizdą po sesijos įkėlimo.
        Suranda pirmą vaizdą su matavimais ir jį įkelia.
        """
        paths = self._manager.get_all_image_paths()
        if not paths:
            self.statusBar().showMessage(
                "Session restored but no image paths found.", 3000
            )
            self._refresh_panel()
            self._update_status()
            return

        # Rasti pirmą egzistuojantį vaizdą su matavimais
        first_path = None
        missing = []
        for img_name, img_path in paths.items():
            if not os.path.exists(img_path):
                missing.append(img_name)
                continue
            if first_path is None:
                first_path = img_path
            if self._manager.get_measurement_count(img_name) > 0:
                first_path = img_path
                break

        if missing:
            QMessageBox.warning(
                self, "Missing Files",
                f"Could not find {len(missing)} image(s):\n"
                + "\n".join(missing[:5])
                + ("\n..." if len(missing) > 5 else "")
            )

        if first_path is None:
            self._refresh_panel()
            self._update_status()
            return

        self._load_image(first_path)

        total = sum(
            len(m) for m in self._manager.measurements.values()
        )
        self.statusBar().showMessage(
            f"Session restored: {total} measurements", 5000
        )

   
    # Refresh helpers
   

    def _refresh_viewer_measurements(self):
        """Atnaujina viewer matavimų sąrašą."""
        if self._current_image_name:
            meas = self._manager.get_measurements(self._current_image_name)
            self._viewer.set_measurements(meas)

    def _refresh_panel(self):
        """Perkuria visas panel korteles iš dabartinio vaizdo matavimų."""
        self._panel.clear_cards()
        if self._current_image_name:
            for m in self._manager.get_measurements(self._current_image_name):
                self._panel.add_card(m)
            self._refresh_vessel_options()

    def _refresh_card(self, measurement_id: int):
        """Atnaujina vieną kortelę."""
        if self._current_image_name:
            m = self._manager._find(self._current_image_name, measurement_id)
            if m:
                self._panel.update_card(measurement_id, m)

    def _refresh_vessel_options(self):
        """Atnaujina vessel ID dropdown visose kortelėse."""
        if self._current_image_name:
            ids_info = self._manager.get_vessel_ids_in_use(
                self._current_image_name
            )
            meas = self._manager.get_measurements(self._current_image_name)
            self._panel.update_vessel_options(ids_info, meas)

   
    # Status bar
   

    def _update_status(self):
        """Atnaujina status bar."""
        if self._current_image_name:
            self._status_image.setText(self._current_image_name)
        else:
            self._status_image.setText("No image")

        # Zoom
        self._status_zoom.setText(
            f"Zoom: {self._viewer.get_zoom_percent()}%"
        )

        # Matavimai
        count = 0
        if self._current_image_name:
            count = self._manager.get_measurement_count(
                self._current_image_name
            )
        self._status_meas.setText(f"Meas: {count}")

        # Undo/Redo
        undo_n = len(self._manager.undo_stack)
        redo_n = len(self._manager.redo_stack)
        self._status_undo.setText(f"Undo:{undo_n}/Redo:{redo_n}")

        # Mode
        self._status_mode.setText(f"Mode: {self._viewer.get_mode().name}")

        # Edit menu descriptions
        self._act_undo.setText(
            self._manager.get_undo_description() or "Undo"
        )
        self._act_redo.setText(
            self._manager.get_redo_description() or "Redo"
        )

    def _on_mode_changed(self, mode_name: str):
        if mode_name == 'PAINT_EXCLUSION':
            brush = self._exclusion.brush_size
            self._status_mode.setText(f"Mode: Paint (brush:{brush})")
        else:
            self._status_mode.setText(f"Mode: {mode_name}")
        self._sync_toolbar_checks()

   
    # Close event
   

    def closeEvent(self, event):
        """Patvirtinimas prieš uždarant jei yra matavimų."""
        total = sum(
            len(m) for m in self._manager.measurements.values()
        )
        has_excl = self._exclusion.has_zones

        if total > 0 or has_excl:
            reply = QMessageBox.question(
                self, "Exit",
                f"You have {total} measurements"
                f"{' and exclusion zones' if has_excl else ''}.\n"
                "Session will be auto-saved.\nExit?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

        # Išsaugoti dabartinę exclusion kaukę prieš autosave
        self._save_current_exclusion()
        self._manager._auto_save()
        event.accept()



# Paleidimas


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Manual Vessel Measurement Tool")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()