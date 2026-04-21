
import math
from enum import Enum, auto
from typing import Optional, List, Tuple

from PyQt5.QtCore import (Qt, pyqtSignal, QPointF, QRectF, QPoint)
from PyQt5.QtGui import (QPainter, QColor, QPen, QBrush, QFont, QImage,
                          QPixmap, QCursor, QWheelEvent, QMouseEvent,
                          QPaintEvent, QKeyEvent, QResizeEvent,
                          QPainterPath)
from PyQt5.QtWidgets import QWidget, QSizePolicy, QMenu, QApplication

import cv2
import numpy as np


from measurement_manager import MeasurementManager

# Konstantos

class Mode(Enum):
    VIEW = auto()
    MEASURE = auto()
    EDIT_LINE = auto()
    OD_MANUAL = auto()
    PAINT_EXCLUSION = auto()


# Spalvos
COLOR_ARTERY = QColor('#FF4444')
COLOR_VEIN = QColor('#4488FF')
COLOR_UNKNOWN = QColor('#FFD700')
COLOR_OD_CIRCLE = QColor(255, 255, 255, 100)
COLOR_ACTIVE_LINE = QColor('#FFFFFF')
COLOR_GHOST_LINE = QColor(255, 255, 255, 80)
COLOR_LASSO = QColor(255, 255, 255, 200)
COLOR_PIXEL_GRID = QColor(255, 255, 255, 30)
COLOR_SELECTED_GLOW = QColor(255, 255, 255, 100)
COLOR_OD_CENTER = QColor(255, 255, 255, 180)
COLOR_LABEL_BG = QColor(0, 0, 0, 140)

TYPE_COLORS = {0: COLOR_UNKNOWN, 1: COLOR_ARTERY, 2: COLOR_VEIN}

# Zoom
ZOOM_MIN = 0.25
ZOOM_MAX = 32.0
ZOOM_STEP = 1.25
ZOOM_FINE_STEP = 1.05
PIXEL_GRID_THRESHOLD = 4.0  # >400% zoom

# Matavimo linija
LINE_WIDTH = 2
LINE_HOVER_WIDTH = 4
LINE_HIT_DISTANCE = 8
EDGE_TICK_LENGTH = 8  # perpendikularaus tick ilgis (px ekrane)

# OD apskritimai (numatytieji)
DEFAULT_OD_CIRCLES = [1.5, 2.0, 2.5, 3.0]

# Kampo fiksavimo žingsnis (Shift)
ANGLE_SNAP = 15


# VesselViewerWidget

class VesselViewerWidget(QWidget):
    """
    Pagrindinis vaizdo peržiūros ir matavimo widget.

    Valdo:
    - Vaizdo atvaizdavimą su zoom/pan
    - Matavimo linijų brėžimą ir vizualizaciją
    - OD apskritimų piešimą
    - Exclusion zone overlay
    - Freehand OD lasso
    - Linijos redagavimą (Redraw Line)
    """

    # Signalai
    measurement_added = pyqtSignal(float, float, float, float)
    measurement_selected = pyqtSignal(int)
    od_manual_finished = pyqtSignal(list)
    redraw_finished = pyqtSignal(int, float, float, float, float)
    view_changed = pyqtSignal()
    mode_changed = pyqtSignal(str)
    exclusion_painted = pyqtSignal()  # Kai pieš. zona pasikeitė
    exclusion_stroke_started = pyqtSignal()  # Prieš naują potėpį (undo)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)

        # --- Vaizdo duomenys ---
        self._original_bgr: Optional[np.ndarray] = None
        self._display_qimage: Optional[QImage] = None
        self._image_w: int = 0
        self._image_h: int = 0

        # --- Zoom / Pan ---
        self._zoom: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._panning: bool = False
        self._pan_start: QPoint = QPoint()
        self._pan_offset_start: Tuple[float, float] = (0.0, 0.0)

        # --- Režimas ---
        self._mode: Mode = Mode.VIEW
        self._prev_mode: Mode = Mode.VIEW

        # --- Matavimo brėžimas ---
        self._drawing: bool = False
        self._draw_start: QPointF = QPointF()  # vaizdo koordinatės
        self._draw_end: QPointF = QPointF()

        # --- Redraw Line ---
        self._redraw_id: Optional[int] = None
        self._redraw_ghost: Optional[dict] = None  # sena linija ghost

        # --- OD freehand lasso ---
        self._lasso_points: List[QPointF] = []  # vaizdo koordinatės

        # --- Matavimų duomenys (gauna iš main.py) ---
        self._measurements: List[dict] = []
        self._reference_measurements: List[dict] = []  # Eksperto
        self._show_reference: bool = True
        self._selected_id: Optional[int] = None
        self._hover_id: Optional[int] = None

        # --- OD duomenys ---
        self._od_x: int = 0
        self._od_y: int = 0
        self._od_r: int = 0
        self._od_circles: List[float] = list(DEFAULT_OD_CIRCLES)
        self._show_od_circles: bool = True

        # --- Overlay duomenys ---
        self._exclusion_overlay: Optional[np.ndarray] = None  # RGBA
        self._exclusion_pixmap: Optional[QPixmap] = None  # Cached
        self._exclusion_dirty: bool = True  # Reikia rebuilidinti pixmap
        self._markers_overlay: Optional[np.ndarray] = None    # RGBA
        self._show_exclusion_zones: bool = False
        self._show_exclusion_markers: bool = False

        # --- Display mode ---
        self._show_clahe: bool = False
        self._show_green: bool = False
        self._clahe_bgr: Optional[np.ndarray] = None
        self._green_channel: Optional[np.ndarray] = None

        # --- Paint exclusion ---
        self._exclusion_computer = None  # nustatomas iš main.py
        self._paint_last: Optional[Tuple[int, int]] = None
        self._paint_erasing: bool = False
#_______________________________________________________________________________
    # Vaizdo nustatymas

    def set_image(self, image_bgr: np.ndarray,
                  clahe_bgr: Optional[np.ndarray] = None):
        """
        Nustato naują vaizdą.

        Args:
            image_bgr: Originalus BGR vaizdas
            clahe_bgr: CLAHE versija (optional, sukurs jei nėra)
        """
        self._original_bgr = image_bgr
        self._image_h, self._image_w = image_bgr.shape[:2]
        self._green_channel = image_bgr[:, :, 1]

        if clahe_bgr is not None:
            self._clahe_bgr = clahe_bgr
        else:
            self._clahe_bgr = None

        self._rebuild_display()
        self.fit_to_window()
        self.update()

    def _rebuild_display(self):
        """Perkuria QImage pagal dabartinį display mode."""
        if self._original_bgr is None:
            self._display_qimage = None
            return

        if self._show_green and self._green_channel is not None:
            # Žalias kanalas → pilkas
            gray = self._green_channel
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        elif self._show_clahe and self._clahe_bgr is not None:
            rgb = cv2.cvtColor(self._clahe_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(self._original_bgr, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        self._display_qimage = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).copy()  # .copy() nes rgb gali būti GC surinktas

    #______________________________________________
    # Display toggles

    def toggle_clahe(self):
        """Toggle CLAHE atvaizdavimą (C klavišas)."""
        self._show_clahe = not self._show_clahe
        if self._show_clahe:
            self._show_green = False
        self._rebuild_display()
        self.update()

    def toggle_green_channel(self):
        """Toggle žalią kanalą (G klavišas)."""
        self._show_green = not self._show_green
        if self._show_green:
            self._show_clahe = False
        self._rebuild_display()
        self.update()

    def set_clahe_image(self, clahe_bgr: np.ndarray):
        """Nustato CLAHE versiją."""
        self._clahe_bgr = clahe_bgr
        if self._show_clahe:
            self._rebuild_display()
            self.update()

    #______________________________________________________
    # OD nustatymai

    def set_od(self, od_x: int, od_y: int, od_r: int):
        """Nustato OD koordinates."""
        self._od_x = od_x
        self._od_y = od_y
        self._od_r = od_r
        self.update()

    def toggle_od_circles(self):
        """Toggle OD apskritimų rodymą (Z klavišas)."""
        self._show_od_circles = not self._show_od_circles
        self.update()

    def set_od_circle_multipliers(self, multipliers: List[float]):
        """Nustato OD apskritimų daugiklius."""
        self._od_circles = sorted(multipliers)
        self.update()

    # ____________________________________________________________
    # Exclusion overlay

    def set_exclusion_overlays(self, zones_rgba: Optional[np.ndarray],
                                markers_rgba: Optional[np.ndarray]):
        """Nustato exclusion zone ir markerių overlay."""
        self._exclusion_overlay = zones_rgba
        self._markers_overlay = markers_rgba
        self._exclusion_dirty = True
        self._exclusion_pixmap = None
        self.update()

    def set_exclusion_computer(self, computer):
        """Nustato ExclusionZoneComputer nuorodą piešimui."""
        self._exclusion_computer = computer

    def toggle_paint_exclusion(self):
        """Toggle exclusion zonų piešimo režimą (P klavišas)."""
        if self._mode == Mode.PAINT_EXCLUSION:
            self.set_mode(Mode.MEASURE)
        else:
            if (self._exclusion_computer is not None and
                    not self._exclusion_computer.is_initialized and
                    self._image_h > 0):
                self._exclusion_computer.init_mask(self._image_h, self._image_w)
                self._exclusion_overlay = self._exclusion_computer.overlay_rgba
                self._exclusion_dirty = True
                self._show_exclusion_zones = True
            self.set_mode(Mode.PAINT_EXCLUSION)

    def toggle_exclusion_zones(self):
        """Toggle exclusion zonų rodymą (R klavišas)."""
        self._show_exclusion_zones = not self._show_exclusion_zones
        self.update()

    def toggle_exclusion_markers(self):
        """Toggle exclusion markerių rodymą (M klavišas)."""
        self._show_exclusion_markers = not self._show_exclusion_markers
        self.update()

    #__________________________________________________
    # Matavimų duomenys

    def set_measurements(self, measurements: List[dict]):
        """Nustato matavimų sąrašą piešimui."""
        self._measurements = measurements
        self.update()

    def set_reference_measurements(self, measurements: List[dict]):
        """Nustato eksperto reference matavimus."""
        self._reference_measurements = measurements
        self.update()

    def toggle_reference(self):
        """Įjungia/išjungia reference sluoksnio matomumą."""
        self._show_reference = not self._show_reference
        self.update()

    def set_selected_measurement(self, measurement_id: Optional[int]):
        """Nustato pasirinktą matavimą."""
        self._selected_id = measurement_id
        self.update()

    # Režimas


    def set_mode(self, mode: Mode):
        """Keičia režimą."""
        self._prev_mode = self._mode
        self._mode = mode
        self._update_cursor()
        self.mode_changed.emit(mode.name)
        self.update()

    def get_mode(self) -> Mode:
        return self._mode

    def start_redraw(self, measurement_id: int, ghost: dict):
        """Pradeda Redraw Line režimą."""
        self._redraw_id = measurement_id
        self._redraw_ghost = ghost
        self.set_mode(Mode.EDIT_LINE)

    def start_od_manual(self):
        """Pradeda freehand OD lasso režimą."""
        self._lasso_points.clear()
        self.set_mode(Mode.OD_MANUAL)

    def cancel_mode(self):
        """Atšaukia dabartinį režimą."""
        self._drawing = False
        self._lasso_points.clear()
        self._redraw_id = None
        self._redraw_ghost = None
        self.set_mode(Mode.MEASURE)

   
    # Koordinačių konversija

    def screen_to_image(self, sx: float, sy: float) -> Tuple[float, float]:
        """Ekrano koordinatės → vaizdo koordinatės."""
        ix = (sx - self._offset_x) / self._zoom
        iy = (sy - self._offset_y) / self._zoom
        return ix, iy

    def image_to_screen(self, ix: float, iy: float) -> Tuple[float, float]:
        """Vaizdo koordinatės → ekrano koordinatės."""
        sx = ix * self._zoom + self._offset_x
        sy = iy * self._zoom + self._offset_y
        return sx, sy

   
    # Zoom / Pan

    def fit_to_window(self):
        """Pritaiko vaizdą prie lango (F klavišas)."""
        if self._image_w == 0 or self._image_h == 0:
            return

        zoom_x = self.width() / self._image_w
        zoom_y = self.height() / self._image_h
        self._zoom = min(zoom_x, zoom_y) * 0.95

        self._offset_x = (self.width() - self._image_w * self._zoom) / 2
        self._offset_y = (self.height() - self._image_h * self._zoom) / 2

        self.view_changed.emit()
        self.update()

    def reset_view(self):
        """Grąžina pradinį zoom/pan (Home klavišas)."""
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self.view_changed.emit()
        self.update()

    def zoom_in(self):
        """Zoom in centruotas."""
        self._apply_zoom(ZOOM_STEP, self.width() / 2, self.height() / 2)

    def zoom_out(self):
        """Zoom out centruotas."""
        self._apply_zoom(1.0 / ZOOM_STEP, self.width() / 2, self.height() / 2)

    def _apply_zoom(self, factor: float, cx: float, cy: float):
        """Pritaiko zoom ties nurodyta ekrano pozicija."""
        new_zoom = self._zoom * factor
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, new_zoom))

        # Zoom ties pelės pozicija
        ix, iy = self.screen_to_image(cx, cy)
        self._zoom = new_zoom
        self._offset_x = cx - ix * self._zoom
        self._offset_y = cy - iy * self._zoom

        self.view_changed.emit()
        self.update()

    def get_zoom_percent(self) -> int:
        """Grąžina zoom procentais."""
        return int(round(self._zoom * 100))

    def center_on_image_point(self, ix: float, iy: float):
        """Centruoja vaizdą ties nurodyta vaizdo koordinate."""
        self._offset_x = self.width() / 2 - ix * self._zoom
        self._offset_y = self.height() / 2 - iy * self._zoom
        self.view_changed.emit()
        self.update()

   
    # Piešimas (paintEvent)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Fonas
        painter.fillRect(self.rect(), QColor('#1E1E1E'))

        if self._display_qimage is None:
            painter.setPen(QColor('#666'))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Open an image (Ctrl+O)")
            painter.end()
            return

        # --- 1. Fundus vaizdas ---
        self._draw_image(painter)

        # --- 2. Pikselių grid (>400% zoom) ---
        if self._zoom >= PIXEL_GRID_THRESHOLD:
            self._draw_pixel_grid(painter)

        # --- 3. Exclusion zone overlay ---
        if self._show_exclusion_zones and self._exclusion_overlay is not None:
            self._draw_rgba_overlay(painter, self._exclusion_overlay)

        # --- 4. Exclusion markeriai ---
        if self._show_exclusion_markers and self._markers_overlay is not None:
            self._draw_rgba_overlay(painter, self._markers_overlay)

        # --- 5. OD apskritimai ---
        if self._show_od_circles and self._od_r > 0:
            self._draw_od_circles(painter)

        # --- 6. Ghost linija (Redraw Line) ---
        if self._redraw_ghost is not None:
            self._draw_ghost_line(painter)

        # --- 7. Eksperto reference linijos ---
        if self._show_reference and self._reference_measurements:
            self._draw_reference_measurements(painter)

        # --- 8. Matavimo linijos ---
        self._draw_measurements(painter)

        # --- 9. Aktyvi brėžiama linija ---
        if self._drawing:
            self._draw_active_line(painter)

        # --- 9. OD lasso kontūras ---
        if self._mode == Mode.OD_MANUAL and self._lasso_points:
            self._draw_lasso(painter)

        # --- 10. Paint brush preview ---
        if self._mode == Mode.PAINT_EXCLUSION and self._exclusion_computer:
            self._draw_brush_cursor(painter)

        painter.end()

    def _draw_image(self, painter: QPainter):
        """Piešia fundus vaizdą su zoom/pan."""
        target = QRectF(
            self._offset_x, self._offset_y,
            self._image_w * self._zoom,
            self._image_h * self._zoom
        )

        # Interpoliacijos režimas pagal zoom
        if self._zoom < 1.0:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        else:
            painter.setRenderHint(QPainter.SmoothPixmapTransform,
                                  self._zoom < PIXEL_GRID_THRESHOLD)

        painter.drawImage(target, self._display_qimage)

    def _draw_pixel_grid(self, painter: QPainter):
        """Piešia pikselių tinklelį kai zoom > 400%."""
        pen = QPen(COLOR_PIXEL_GRID, 1)
        painter.setPen(pen)

        # Matoma sritis vaizdo koordinatėmis
        vx0, vy0 = self.screen_to_image(0, 0)
        vx1, vy1 = self.screen_to_image(self.width(), self.height())

        x_start = max(0, int(vx0))
        x_end = min(self._image_w, int(vx1) + 1)
        y_start = max(0, int(vy0))
        y_end = min(self._image_h, int(vy1) + 1)

        # Vertikalios linijos
        for x in range(x_start, x_end + 1):
            sx, sy0 = self.image_to_screen(x, y_start)
            _, sy1 = self.image_to_screen(x, y_end)
            painter.drawLine(QPointF(sx, sy0), QPointF(sx, sy1))

        # Horizontalios linijos
        for y in range(y_start, y_end + 1):
            sx0, sy = self.image_to_screen(x_start, y)
            sx1, _ = self.image_to_screen(x_end, y)
            painter.drawLine(QPointF(sx0, sy), QPointF(sx1, sy))

    def _draw_rgba_overlay(self, painter: QPainter, overlay_rgba: np.ndarray):
        """Piešia RGBA overlay ant vaizdo.

        Naudoja cached QPixmap — rebuilidina tik kai dirty flag
        nustatytas. Piešimo metu (paint mode) naudoja QImage be
        kopijos (shared memory su numpy).
        """
        h, w = overlay_rgba.shape[:2]

        if self._exclusion_dirty or self._exclusion_pixmap is None:
            # Tiesioginė QImage be .copy() — numpy laiko duomenis
            self._excl_qimg_ref = overlay_rgba  # Nuoroda, kad GC neištrintų
            qimg = QImage(
                overlay_rgba.data, w, h, w * 4,
                QImage.Format_RGBA8888
            )

            # Jei ne paint mode — konvertuoti į QPixmap (greitesnė piešti)
            if self._paint_last is None:
                self._exclusion_pixmap = QPixmap.fromImage(qimg)
                self._exclusion_dirty = False
            else:
                # Piešimo metu — piešti tiesiai iš QImage (vengia
                # QPixmap konversijos kiekviename mouse move)
                target = QRectF(
                    self._offset_x, self._offset_y,
                    self._image_w * self._zoom,
                    self._image_h * self._zoom
                )
                painter.drawImage(target, qimg)
                return

        target = QRectF(
            self._offset_x, self._offset_y,
            self._image_w * self._zoom,
            self._image_h * self._zoom
        )
        painter.drawPixmap(target, self._exclusion_pixmap,
                           QRectF(self._exclusion_pixmap.rect()))

    def _draw_od_circles(self, painter: QPainter):
        """Piešia OD centro kryželį ir orientacinius apskritimus."""
        scx, scy = self.image_to_screen(self._od_x, self._od_y)

        # Centro kryželis
        pen = QPen(COLOR_OD_CENTER, 2)
        painter.setPen(pen)
        s = 8
        painter.drawLine(QPointF(scx - s, scy), QPointF(scx + s, scy))
        painter.drawLine(QPointF(scx, scy - s), QPointF(scx, scy + s))

        # Apskritimai
        pen = QPen(COLOR_OD_CIRCLE, 1.5, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        font = QFont("Monospace", 8)
        painter.setFont(font)

        for mult in self._od_circles:
            r_screen = self._od_r * mult * self._zoom
            painter.drawEllipse(QPointF(scx, scy), r_screen, r_screen)

            # Užrašas
            label = f"{mult:.1f} rOD"
            lx = scx + r_screen + 4
            ly = scy - 4
            painter.setPen(COLOR_OD_CIRCLE)
            painter.drawText(QPointF(lx, ly), label)
            painter.setPen(pen)

    def _draw_ghost_line(self, painter: QPainter):
        """Piešia ghost liniją (Redraw Line — sena linija)."""
        g = self._redraw_ghost
        sx1, sy1 = self.image_to_screen(g['x1'], g['y1'])
        sx2, sy2 = self.image_to_screen(g['x2'], g['y2'])

        pen = QPen(COLOR_GHOST_LINE, LINE_WIDTH, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))

    def _draw_reference_measurements(self, painter: QPainter):
        """Piešia eksperto reference matavimus — dashed, geltona/žydra."""
        from measurement_manager import MeasurementManager

        REF_COLORS = {
            0: QColor(255, 200, 60, 180),   # Unknown — geltona
            1: QColor(60, 220, 255, 180),    # Artery — žydra
            2: QColor(255, 140, 60, 180),    # Vein — oranžinė
        }

        for m in self._reference_measurements:
            vtype = m.get('vessel_type', 0)
            color = REF_COLORS.get(vtype, REF_COLORS[0])

            # Extended arba paprastos koordinatės
            ex1 = m.get('ext_x1', m.get('x1', 0))
            ey1 = m.get('ext_y1', m.get('y1', 0))
            ex2 = m.get('ext_x2', m.get('x2', 0))
            ey2 = m.get('ext_y2', m.get('y2', 0))

            sex1, sey1 = self.image_to_screen(ex1, ey1)
            sex2, sey2 = self.image_to_screen(ex2, ey2)

            # Dashed linija
            pen = QPen(color, 2, Qt.DashLine)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(sex1, sey1), QPointF(sex2, sey2))

            # Edge tick-marks (jei yra profilio duomenys)
            edges = m.get('profile_edges_manual') or m.get('profile_edges')
            if edges is not None:
                left_edge, right_edge = edges
                dx = ex2 - ex1
                dy = ey2 - ey1
                elen = math.sqrt(dx * dx + dy * dy)
                if elen > 1e-6:
                    perp_x = -dy / elen
                    perp_y = dx / elen
                    tick_img = EDGE_TICK_LENGTH / self._zoom

                    tick_pen = QPen(color, 2)
                    tick_pen.setCapStyle(Qt.RoundCap)
                    painter.setPen(tick_pen)

                    for edge_sample in [left_edge, right_edge]:
                        try:
                            img_x, img_y = MeasurementManager.edge_to_image_coords(
                                m, edge_sample
                            )
                        except (KeyError, ZeroDivisionError):
                            continue
                        t1x = img_x + perp_x * tick_img
                        t1y = img_y + perp_y * tick_img
                        t2x = img_x - perp_x * tick_img
                        t2y = img_y - perp_y * tick_img
                        st1x, st1y = self.image_to_screen(t1x, t1y)
                        st2x, st2y = self.image_to_screen(t2x, t2y)
                        painter.drawLine(
                            QPointF(st1x, st1y), QPointF(st2x, st2y)
                        )

            # Etiketė — "REF: 8.5px"
            sx1, sy1 = self.image_to_screen(
                m.get('x1', 0), m.get('y1', 0)
            )
            sx2, sy2 = self.image_to_screen(
                m.get('x2', 0), m.get('y2', 0)
            )
            w = m.get('width_manual') or m.get('width_px', 0)
            label = f"REF:{w:.1f}"
            font = QFont("Monospace", 7)
            painter.setFont(font)
            painter.setPen(color)
            mid_sx = (sx1 + sx2) / 2
            mid_sy = (sy1 + sy2) / 2 - 8
            painter.drawText(QPointF(mid_sx, mid_sy), label)

    def _draw_measurements(self, painter: QPainter):
        """Piešia visas matavimo linijas su kraštų markeriais."""
        for m in self._measurements:
            mid = m['id']
            vtype = m.get('vessel_type', 0)
            color = TYPE_COLORS.get(vtype, COLOR_UNKNOWN)

            # Piešiame extended liniją (profilio sritį)
            ex1 = m.get('ext_x1', m['x1'])
            ey1 = m.get('ext_y1', m['y1'])
            ex2 = m.get('ext_x2', m['x2'])
            ey2 = m.get('ext_y2', m['y2'])

            sex1, sey1 = self.image_to_screen(ex1, ey1)
            sex2, sey2 = self.image_to_screen(ex2, ey2)

            # Hover / selected
            is_hover = (mid == self._hover_id)
            is_selected = (mid == self._selected_id)
            width = LINE_HOVER_WIDTH if is_hover else LINE_WIDTH

            # Selected glow
            if is_selected:
                glow_pen = QPen(COLOR_SELECTED_GLOW, width + 4)
                glow_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(glow_pen)
                painter.drawLine(QPointF(sex1, sey1), QPointF(sex2, sey2))

            # Pagrindinė linija (extended)
            pen = QPen(color, width)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(sex1, sey1), QPointF(sex2, sey2))

            # --- Kraštų tick-marks (perpendikularūs) ---
            self._draw_edge_ticks(painter, m, color, is_selected)

            # Etiketė
            sx1, sy1 = self.image_to_screen(m['x1'], m['y1'])
            sx2, sy2 = self.image_to_screen(m['x2'], m['y2'])
            self._draw_line_label(painter, m, sx1, sy1, sx2, sy2, color)

    def _draw_edge_ticks(self, painter: QPainter, m: dict,
                         color: QColor, is_selected: bool):
        """
        Piešia perpendikliarius tick-marks kraujagyslės kraštuose.

        Naudoja profile_edges_manual jei koreguota, kitaip profile_edges.
        Konvertuoja edge sample pozicijas į vaizdo koordinates
        per extended line (ext_x1..ext_y2).
        """
        # Pasirinkti aktualius kraštus
        edges = m.get('profile_edges_manual') or m.get('profile_edges')
        if edges is None:
            return

        left_edge, right_edge = edges

        ex1 = m.get('ext_x1', m['x1'])
        ey1 = m.get('ext_y1', m['y1'])
        ex2 = m.get('ext_x2', m['x2'])
        ey2 = m.get('ext_y2', m['y2'])
        dx = ex2 - ex1
        dy = ey2 - ey1
        elen = math.sqrt(dx * dx + dy * dy)
        if elen < 1e-6:
            return
        perp_x = -dy / elen
        perp_y = dx / elen

        # Tick ilgis ekrano pikseliais  vaizdo pikseliais
        tick_img = EDGE_TICK_LENGTH / self._zoom

        # Edge spalva
        tick_color = QColor(color)
        tick_width = 3 if is_selected else 2
        pen = QPen(tick_color, tick_width)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)

        for edge_sample in [left_edge, right_edge]:
            ix, iy = MeasurementManager.edge_to_image_coords(m, edge_sample)

            # Tick endpoints
            t1x = ix - perp_x * tick_img
            t1y = iy - perp_y * tick_img
            t2x = ix + perp_x * tick_img
            t2y = iy + perp_y * tick_img

            st1x, st1y = self.image_to_screen(t1x, t1y)
            st2x, st2y = self.image_to_screen(t2x, t2y)

            painter.drawLine(QPointF(st1x, st1y), QPointF(st2x, st2y))

    def _draw_line_label(self, painter: QPainter, m: dict,
                         sx1: float, sy1: float,
                         sx2: float, sy2: float,
                         color: QColor):
        """Piešia matavimo linijos etiketę (#N ir plotį)."""
        # Efektyvus plotis
        w = m.get('width_manual') if m.get('width_manual') is not None else m.get('width_px', 0)
        label = f"#{m['id']} {w:.1f}px"

        mid_x = (sx1 + sx2) / 2
        mid_y = (sy1 + sy2) / 2

        font = QFont("Monospace", 8)
        painter.setFont(font)

        # Fonas etiketei
        fm = painter.fontMetrics()
        text_rect = fm.boundingRect(label)
        bg_rect = QRectF(
            mid_x - text_rect.width() / 2 - 3,
            mid_y - text_rect.height() - 6,
            text_rect.width() + 6,
            text_rect.height() + 4
        )
        painter.fillRect(bg_rect, QBrush(COLOR_LABEL_BG))

        painter.setPen(color)
        painter.drawText(bg_rect, Qt.AlignCenter, label)

    def _draw_active_line(self, painter: QPainter):
        """Piešia aktyvią brėžiamą liniją."""
        sx1, sy1 = self.image_to_screen(
            self._draw_start.x(), self._draw_start.y()
        )
        sx2, sy2 = self.image_to_screen(
            self._draw_end.x(), self._draw_end.y()
        )

        pen = QPen(COLOR_ACTIVE_LINE, LINE_WIDTH, Qt.DashDotLine)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))

        # Ilgis realiu laiku
        dx = self._draw_end.x() - self._draw_start.x()
        dy = self._draw_end.y() - self._draw_start.y()
        length = math.sqrt(dx * dx + dy * dy)
        label = f"{length:.1f}px"

        font = QFont("Monospace", 9)
        painter.setFont(font)
        painter.setPen(COLOR_ACTIVE_LINE)
        painter.drawText(QPointF(sx2 + 8, sy2 - 8), label)

    def _draw_lasso(self, painter: QPainter):
        """Piešia OD freehand lasso kontūrą."""
        if len(self._lasso_points) < 2:
            return

        pen = QPen(COLOR_LASSO, 2, Qt.DashLine)
        painter.setPen(pen)

        path = QPainterPath()
        sx0, sy0 = self.image_to_screen(
            self._lasso_points[0].x(), self._lasso_points[0].y()
        )
        path.moveTo(sx0, sy0)

        for pt in self._lasso_points[1:]:
            sx, sy = self.image_to_screen(pt.x(), pt.y())
            path.lineTo(sx, sy)

        painter.drawPath(path)

        # Fitintas apskritimas preview
        if len(self._lasso_points) >= 3:
            contour = np.array(
                [(p.x(), p.y()) for p in self._lasso_points],
                dtype=np.float32
            ).reshape(-1, 1, 2)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)

            scx, scy = self.image_to_screen(cx, cy)
            sr = radius * self._zoom

            pen_preview = QPen(QColor(0, 255, 0, 150), 2, Qt.DashLine)
            painter.setPen(pen_preview)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(scx, scy), sr, sr)

    def _draw_brush_cursor(self, painter: QPainter):
        """Piešia brush apskritimo preview ties pelės pozicija."""
        pos = self.mapFromGlobal(QCursor.pos())
        if not self.rect().contains(pos):
            return

        brush_r = self._exclusion_computer.brush_size * self._zoom

        if self._paint_erasing or (QApplication.keyboardModifiers() & Qt.ShiftModifier):
            # Trintukas — mėlynas
            pen = QPen(QColor(100, 200, 255, 180), 2, Qt.DashLine)
        else:
            # Piešimas — raudonas
            pen = QPen(QColor(255, 80, 80, 180), 2, Qt.DashLine)

        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(pos.x(), pos.y()), brush_r, brush_r)

   
    # Mouse events
   

    def mousePressEvent(self, event: QMouseEvent):
        # --- Pan (middle arba right) ---
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = True
            self._pan_start = event.pos()
            self._pan_offset_start = (self._offset_x, self._offset_y)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        ix, iy = self.screen_to_image(event.x(), event.y())

        # --- MEASURE mode ---
        if self._mode == Mode.MEASURE:
            self._drawing = True
            self._draw_start = QPointF(ix, iy)
            self._draw_end = QPointF(ix, iy)
            event.accept()
            return

        # --- EDIT_LINE mode ---
        if self._mode == Mode.EDIT_LINE:
            self._drawing = True
            self._draw_start = QPointF(ix, iy)
            self._draw_end = QPointF(ix, iy)
            event.accept()
            return

        # --- OD_MANUAL mode ---
        if self._mode == Mode.OD_MANUAL:
            self._lasso_points = [QPointF(ix, iy)]
            self._drawing = True
            event.accept()
            return

        # --- PAINT_EXCLUSION mode ---
        if self._mode == Mode.PAINT_EXCLUSION:
            ix, iy = self.screen_to_image(event.x(), event.y())
            self._paint_erasing = bool(event.modifiers() & Qt.ShiftModifier)
            self._paint_last = (int(round(ix)), int(round(iy)))
            if self._exclusion_computer is not None:
                self.exclusion_stroke_started.emit()
                self._exclusion_computer.paint(
                    int(round(ix)), int(round(iy)), self._paint_erasing
                )
                self._exclusion_overlay = self._exclusion_computer.overlay_rgba
                self._exclusion_dirty = True
                self.update()
            event.accept()
            return

        if self._mode == Mode.VIEW:
            hit = self._hit_test_measurement(event.x(), event.y())
            if hit is not None:
                self._selected_id = hit
                self.measurement_selected.emit(hit)
                self.update()
            else:
                # Pradėti pan su left mouse
                self._panning = True
                self._pan_start = event.pos()
                self._pan_offset_start = (self._offset_x, self._offset_y)
                self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

    def mouseMoveEvent(self, event: QMouseEvent):
        # --- Pan ---
        if self._panning:
            dx = event.x() - self._pan_start.x()
            dy = event.y() - self._pan_start.y()
            self._offset_x = self._pan_offset_start[0] + dx
            self._offset_y = self._pan_offset_start[1] + dy
            self.view_changed.emit()
            self.update()
            event.accept()
            return

        ix, iy = self.screen_to_image(event.x(), event.y())

        # --- Brėžimas (MEASURE / EDIT_LINE) ---
        if self._drawing and self._mode in (Mode.MEASURE, Mode.EDIT_LINE):
            end = QPointF(ix, iy)

            # Shift = kampo fiksavimas kas 15°
            if event.modifiers() & Qt.ShiftModifier:
                end = self._snap_angle(self._draw_start, end, ANGLE_SNAP)
            # H = horizontali
            elif self._is_key_held('H'):
                end = QPointF(ix, self._draw_start.y())

            self._draw_end = end
            self.update()
            event.accept()
            return

        # --- OD lasso ---
        if self._drawing and self._mode == Mode.OD_MANUAL:
            self._lasso_points.append(QPointF(ix, iy))
            self.update()
            event.accept()
            return

        # --- Paint exclusion ---
        if self._mode == Mode.PAINT_EXCLUSION and self._paint_last is not None:
            nx, ny = int(round(ix)), int(round(iy))
            if self._exclusion_computer is not None:
                self._exclusion_computer.paint_line(
                    self._paint_last[0], self._paint_last[1],
                    nx, ny, self._paint_erasing
                )
                self._exclusion_overlay = self._exclusion_computer.overlay_rgba
                self._exclusion_dirty = True
                self._paint_last = (nx, ny)
                self.update()
            event.accept()
            return

        # --- Hover ant matavimo linijos ---
        if self._mode in (Mode.VIEW, Mode.MEASURE):
            hit = self._hit_test_measurement(event.x(), event.y())
            if hit != self._hover_id:
                self._hover_id = hit
                self.update()

        # --- Brush cursor sekimas ---
        if self._mode == Mode.PAINT_EXCLUSION:
            self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        # --- Pan end (middle/right arba VIEW mode left) ---
        if self._panning and (
            event.button() in (Qt.MiddleButton, Qt.RightButton) or
            (event.button() == Qt.LeftButton and self._mode == Mode.VIEW)
        ):
            self._panning = False
            self._update_cursor()
            event.accept()
            return

        if event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return

        # --- MEASURE finish ---
        if self._drawing and self._mode == Mode.MEASURE:
            self._drawing = False
            x1, y1 = self._draw_start.x(), self._draw_start.y()
            x2, y2 = self._draw_end.x(), self._draw_end.y()

            # Min ilgis
            dx = x2 - x1
            dy = y2 - y1
            if math.sqrt(dx * dx + dy * dy) >= 2.0:
                self.measurement_added.emit(x1, y1, x2, y2)

            self.update()
            event.accept()
            return

        # --- EDIT_LINE finish ---
        if self._drawing and self._mode == Mode.EDIT_LINE:
            self._drawing = False
            x1, y1 = self._draw_start.x(), self._draw_start.y()
            x2, y2 = self._draw_end.x(), self._draw_end.y()

            dx = x2 - x1
            dy = y2 - y1
            if math.sqrt(dx * dx + dy * dy) >= 2.0 and self._redraw_id is not None:
                self.redraw_finished.emit(
                    self._redraw_id, x1, y1, x2, y2
                )

            self._redraw_id = None
            self._redraw_ghost = None
            self.set_mode(Mode.MEASURE)
            self.update()
            event.accept()
            return

        # --- PAINT_EXCLUSION finish ---
        if self._mode == Mode.PAINT_EXCLUSION and self._paint_last is not None:
            self._paint_last = None
            self.exclusion_painted.emit()
            event.accept()
            return

        # --- OD lasso finish ---
        if self._drawing and self._mode == Mode.OD_MANUAL:
            self._drawing = False

            if len(self._lasso_points) >= 3:
                points = [(p.x(), p.y()) for p in self._lasso_points]
                self.od_manual_finished.emit(points)

            self._lasso_points.clear()
            self.set_mode(Mode.MEASURE)
            self.update()
            event.accept()
            return

    def wheelEvent(self, event: QWheelEvent):
        """Zoom su pelės ratuku, arba brush dydis paint režime."""
        # Paint mode — ratukas keičia brush dydį
        if self._mode == Mode.PAINT_EXCLUSION and self._exclusion_computer:
            degrees = event.angleDelta().y() / 8
            steps = int(degrees / 15)
            if steps > 0:
                self._exclusion_computer.increase_brush(3 * steps)
            elif steps < 0:
                self._exclusion_computer.decrease_brush(3 * abs(steps))
            self.update()
            event.accept()
            return

        degrees = event.angleDelta().y() / 8
        steps = degrees / 15

        if event.modifiers() & Qt.ControlModifier:
            factor = ZOOM_FINE_STEP ** steps
        else:
            factor = ZOOM_STEP ** steps

        pos = event.position() if hasattr(event, 'position') else event.posF()
        self._apply_zoom(factor, pos.x(), pos.y())
        event.accept()

    def contextMenuEvent(self, event):
        """Kontekstinis meniu ant matavimo linijos."""
        hit = self._hit_test_measurement(event.x(), event.y())
        if hit is None:
            return

        self._selected_id = hit
        self.measurement_selected.emit(hit)
        self.update()

        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #1E1E1E; color: #FFFFFF; "
            "  border: 1px solid #555; padding: 4px; }"
            "QMenu::item { padding: 4px 20px; }"
            "QMenu::item:selected { background: #2266CC; color: #FFFFFF; }"
            "QMenu::separator { background: #444; height: 1px; "
            "  margin: 4px 8px; }"
        )

        act_artery = menu.addAction("Artery (1)")
        act_vein = menu.addAction("Vein (2)")
        act_unknown = menu.addAction("Unknown (0)")
        menu.addSeparator()
        act_edit = menu.addAction("Edit (E)")
        act_delete = menu.addAction("Delete")

        action = menu.exec_(event.globalPos())

        # Signalai apdorojami per main.py
        if action == act_delete:
            self.measurement_selected.emit(hit)
            # Delete bus apdorojamas main.py per shortkey

   
    # Keyboard
   

    _held_keys = set()

    def keyPressEvent(self, event: QKeyEvent):
        VesselViewerWidget._held_keys.add(event.key())
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        VesselViewerWidget._held_keys.discard(event.key())
        super().keyReleaseEvent(event)

    @classmethod
    def _is_key_held(cls, key_char: str) -> bool:
        """Tikrina ar klavišas laikomas (ne shortkey, o hold)."""
        return getattr(Qt, f'Key_{key_char}', -1) in cls._held_keys

   
    # Hit testing
   

    def _hit_test_measurement(self, sx: int, sy: int) -> Optional[int]:
        """
        Tikrina ar pelė virš matavimo linijos.
        Grąžina measurement_id arba None.
        """
        for m in reversed(self._measurements):
            ex1 = m.get('ext_x1', m['x1'])
            ey1 = m.get('ext_y1', m['y1'])
            ex2 = m.get('ext_x2', m['x2'])
            ey2 = m.get('ext_y2', m['y2'])

            sx1, sy1 = self.image_to_screen(ex1, ey1)
            sx2, sy2 = self.image_to_screen(ex2, ey2)

            dist = self._point_to_segment_distance(
                sx, sy, sx1, sy1, sx2, sy2
            )
            if dist <= LINE_HIT_DISTANCE:
                return m['id']

        return None

    @staticmethod
    def _point_to_segment_distance(px: float, py: float,
                                    x1: float, y1: float,
                                    x2: float, y2: float) -> float:
        """Atstumas nuo taško iki atkarpos."""
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy

        if length_sq < 1e-10:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

   
    # Kampo fiksavimas

    @staticmethod
    def _snap_angle(start: QPointF, end: QPointF,
                    snap_degrees: int) -> QPointF:
        """Fiksuoja kampą kas snap_degrees laipsnių."""
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.sqrt(dx * dx + dy * dy)

        if length < 1:
            return end

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        snapped = round(angle_deg / snap_degrees) * snap_degrees
        snapped_rad = math.radians(snapped)

        return QPointF(
            start.x() + length * math.cos(snapped_rad),
            start.y() + length * math.sin(snapped_rad)
        )

   
    # Cursor

    def _update_cursor(self):
        """Atnaujina kursorių pagal režimą."""
        cursors = {
            Mode.VIEW: Qt.OpenHandCursor,
            Mode.MEASURE: Qt.CrossCursor,
            Mode.EDIT_LINE: Qt.CrossCursor,
            Mode.OD_MANUAL: Qt.CrossCursor,
            Mode.PAINT_EXCLUSION: Qt.BlankCursor,  # Piešiame patys
        }
        self.setCursor(cursors.get(self._mode, Qt.ArrowCursor))

   
    # Resize

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.view_changed.emit()