
from typing import Optional, Dict, List

from PyQt5.QtCore import (Qt, pyqtSignal, QRectF, QPointF,
                           QTimer)
from PyQt5.QtGui import (QPainter, QColor, QPen, QBrush, QFont,
                          QPainterPath)
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                              QScrollArea, QFrame, QLabel, QPushButton,
                              QComboBox, QSizePolicy)

import numpy as np



# Spalvos

COLOR_TEXT = QColor('#E0E0E0')
COLOR_TEXT_DIM = QColor('#888888')
COLOR_ARTERY = QColor('#FF4444')
COLOR_VEIN = QColor('#4488FF')
COLOR_UNKNOWN = QColor('#FFD700')
COLOR_PROFILE_LINE = QColor('#00CC00')
COLOR_EDGE_LINE = QColor('#FF4444')
COLOR_EDGE_LINE_FAINT = QColor(255, 68, 68, 80)
COLOR_EDGE_FILL = QColor(255, 68, 68, 64)
COLOR_MARKER_HANDLE = QColor('#FF6666')
COLOR_MARKER_HOVER = QColor('#FFAAAA')
COLOR_WARNING_TEXT = QColor('#FFB800')
COLOR_SELECTED_BORDER = QColor('#4488FF')
COLOR_CARD_BG = QColor('#2D2D2D')
COLOR_CARD_BORDER = QColor('#444444')
COLOR_GRAPH_BG = QColor('#1A1A1A')

# Tipo spalvos dict
TYPE_COLORS = {0: COLOR_UNKNOWN, 1: COLOR_ARTERY, 2: COLOR_VEIN}
TYPE_LABELS = {0: '?', 1: 'A', 2: 'V'}

# Kortelės dydžiai
CARD_WIDTH = 280
CARD_SPACING = 4
GRAPH_HEIGHT = 100
MARKER_WIDTH = 8
MARKER_HIT_MARGIN = 6

class ProfileGraphWidget(QWidget):
    """
    Profilio grafikas su interaktyviais kraštų markeriais.

    Piešiamas QPainter (NE matplotlib):
    - Fonas: tamsus
    - Žalia linija: intensyvumo profilis
    - Raudonos punktyrinės: kraštų pozicijos
    - Šviesiai raudonas užpildymas: plotis tarp kraštų
    - Adjust mode: vilkomi markeriai + blankios originalios pozicijos

    Signals:
        edges_changed(float, float) — naujos kraštų pozicijos
    """

    edges_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumHeight(GRAPH_HEIGHT)
        self.setMaximumHeight(GRAPH_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)

        # Duomenys
        self._profile: Optional[np.ndarray] = None
        self._left_edge: float = 0.0
        self._right_edge: float = 0.0
        self._left_original: float = 0.0
        self._right_original: float = 0.0
        self._threshold: float = 0.0
        self._samples_per_pixel: float = 1.0
        self._has_edges: bool = False

        # Adjust Edges režimas
        self._adjust_mode: bool = False
        self._dragging: Optional[str] = None  # 'left' arba 'right'
        self._hover: Optional[str] = None

        # Piešimo padding
        self._pad_left = 24
        self._pad_right = 8
        self._pad_top = 6
        self._pad_bottom = 14

   
    # Duomenų nustatymas

    def set_profile_data(self, profile: np.ndarray,
                         left_edge: Optional[float],
                         right_edge: Optional[float],
                         threshold: Optional[float],
                         samples_per_pixel: float,
                         manual_left: Optional[float] = None,
                         manual_right: Optional[float] = None):
        """Nustato profilio duomenis piešimui."""
        self._profile = profile
        self._left_original = left_edge
        self._right_original = right_edge
        self._threshold = threshold
        self._samples_per_pixel = samples_per_pixel

        # Jei yra rankiniai kraštai - naudoti juos
        if manual_left is not None and manual_right is not None:
            self._left_edge = manual_left
            self._right_edge = manual_right
            self._has_edges = True
        elif left_edge is not None and right_edge is not None:
            self._left_edge = left_edge
            self._right_edge = right_edge
            self._has_edges = True
        else:
            self._left_edge = 0.0
            self._right_edge = 0.0
            self._has_edges = False

        self.update()

    def set_adjust_mode(self, enabled: bool):
        """Įjungia/išjungia kraštų vilkimo režimą."""
        self._adjust_mode = enabled
        self._dragging = None
        self._hover = None
        self.setCursor(Qt.ArrowCursor)

        # Jei nėra kraštų, inicializuoti ties 1/3 ir 2/3 profilio
        if enabled and not self._has_edges and self._profile is not None:
            n = len(self._profile)
            self._left_edge = n * 0.33
            self._right_edge = n * 0.66
            self._has_edges = True

        self.update()

    def get_edges(self):
        """Grąžina dabartines kraštų pozicijas."""
        return self._left_edge, self._right_edge

    def get_width_px(self) -> float:
        """Grąžina dabartinį plotį pikseliais."""
        if not self._has_edges:
            return 0.0
        if self._samples_per_pixel > 0:
            return abs(self._right_edge - self._left_edge) / self._samples_per_pixel
        return 0.0

   
    # Koordinačių konversija

    def _graph_rect(self) -> QRectF:
        """Grafiko piešimo sritis."""
        return QRectF(
            self._pad_left,
            self._pad_top,
            self.width() - self._pad_left - self._pad_right,
            self.height() - self._pad_top - self._pad_bottom
        )

    def _sample_to_x(self, sample_idx: float) -> float:
        """Konvertuoja sample indeksą į widget X koordinatę."""
        rect = self._graph_rect()
        if self._profile is None or len(self._profile) < 2:
            return rect.left()
        t = sample_idx / (len(self._profile) - 1)
        return rect.left() + t * rect.width()

    def _x_to_sample(self, x: float) -> float:
        """Konvertuoja widget X koordinatę į sample indeksą."""
        rect = self._graph_rect()
        if self._profile is None or len(self._profile) < 2:
            return 0.0
        t = (x - rect.left()) / rect.width()
        t = max(0.0, min(1.0, t))
        return t * (len(self._profile) - 1)

    def _value_to_y(self, value: float, v_min: float, v_max: float) -> float:
        """Konvertuoja intensyvumo reikšmę į widget Y koordinatę."""
        rect = self._graph_rect()
        if v_max == v_min:
            return rect.center().y()
        t = (value - v_min) / (v_max - v_min)
        return rect.bottom() - t * rect.height()

   
    # Piešimas

    def paintEvent(self, event):
        if self._profile is None or len(self._profile) < 2:
            painter = QPainter(self)
            painter.fillRect(self.rect(), COLOR_GRAPH_BG)
            painter.setPen(COLOR_TEXT_DIM)
            painter.drawText(self.rect(), Qt.AlignCenter, "No profile")
            painter.end()
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self._graph_rect()
        profile = self._profile
        n = len(profile)

        # Fiksuotas 0–255 diapazonas (profilis jau normuotas)
        v_min = 0.0
        v_max = 255.0

        # --- Fonas ---
        painter.fillRect(self.rect(), COLOR_GRAPH_BG)

        # --- Y ašies etiketės (0, 128, 255) ---
        axis_font = QFont("Monospace", 7)
        painter.setFont(axis_font)
        painter.setPen(QColor(255, 255, 255, 60))
        for val in [0, 128, 255]:
            y_pos = self._value_to_y(val, v_min, v_max)
            painter.drawText(
                QRectF(0, y_pos - 6, self._pad_left - 2, 12),
                Qt.AlignRight | Qt.AlignVCenter,
                str(val)
            )
            # Horizontali linija
            painter.drawLine(
                QPointF(rect.left(), y_pos),
                QPointF(rect.right(), y_pos)
            )

        # --- Pločio užpildymas tarp kraštų ---
        if self._has_edges:
            left_x = self._sample_to_x(self._left_edge)
            right_x = self._sample_to_x(self._right_edge)

            painter.fillRect(
                QRectF(left_x, rect.top(), right_x - left_x, rect.height()),
                QBrush(COLOR_EDGE_FILL)
            )

        # --- Slenkščio horizontali linija ---
        if self._threshold is not None and self._has_edges:
            thresh_y = self._value_to_y(self._threshold, v_min, v_max)
            pen_thresh = QPen(QColor(255, 255, 255, 50), 1, Qt.DashLine)
            painter.setPen(pen_thresh)
            painter.drawLine(QPointF(rect.left(), thresh_y),
                             QPointF(rect.right(), thresh_y))

        # --- Profilio linija ---
        pen_profile = QPen(COLOR_PROFILE_LINE, 1.5)
        painter.setPen(pen_profile)

        path = QPainterPath()
        for i in range(n):
            x = self._sample_to_x(i)
            y = self._value_to_y(profile[i], v_min, v_max)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        painter.drawPath(path)

        # --- Kraštų linijos ---
        if self._has_edges:
            left_x = self._sample_to_x(self._left_edge)
            right_x = self._sample_to_x(self._right_edge)

            if self._adjust_mode and self._left_original is not None:
                # Originalios pozicijos — blankios punktyrinės
                pen_faint = QPen(COLOR_EDGE_LINE_FAINT, 1, Qt.DashLine)
                painter.setPen(pen_faint)
                orig_left_x = self._sample_to_x(self._left_original)
                orig_right_x = self._sample_to_x(self._right_original)
                painter.drawLine(QPointF(orig_left_x, rect.top()),
                                 QPointF(orig_left_x, rect.bottom()))
                painter.drawLine(QPointF(orig_right_x, rect.top()),
                                 QPointF(orig_right_x, rect.bottom()))

            # Aktualios kraštų linijos
            pen_edge = QPen(COLOR_EDGE_LINE, 1.5, Qt.DashLine)
            painter.setPen(pen_edge)
            painter.drawLine(QPointF(left_x, rect.top()),
                             QPointF(left_x, rect.bottom()))
            painter.drawLine(QPointF(right_x, rect.top()),
                             QPointF(right_x, rect.bottom()))

            # --- Vilkomi markeriai (tik adjust mode) ---
            if self._adjust_mode:
                self._draw_marker(painter, left_x, rect, 'left')
                self._draw_marker(painter, right_x, rect, 'right')

        # --- Plotis po grafiku ---
        width_px = self.get_width_px()
        font = QFont("Monospace", 8)
        painter.setFont(font)
        painter.setPen(COLOR_TEXT_DIM)
        label = f"{width_px:.2f} px" if self._has_edges else "-- px"
        painter.drawText(
            QRectF(rect.left(), rect.bottom() + 1, rect.width(), 12),
            Qt.AlignCenter, label
        )

        painter.end()

    def _draw_marker(self, painter: QPainter, x: float,
                     rect: QRectF, side: str):
        """Piešia vilkomą markerį (stačiakampis su trikampiu)."""
        is_hover = (self._hover == side)
        is_drag = (self._dragging == side)

        color = COLOR_MARKER_HOVER if (is_hover or is_drag) else COLOR_MARKER_HANDLE

        # Stačiakampis markeris
        marker_h = rect.height() * 0.4
        marker_y = rect.center().y() - marker_h / 2

        painter.fillRect(
            QRectF(x - MARKER_WIDTH / 2, marker_y,
                   MARKER_WIDTH, marker_h),
            QBrush(color)
        )

        # Kraštinė linija
        pen = QPen(color.darker(120), 1)
        painter.setPen(pen)
        painter.drawRect(
            QRectF(x - MARKER_WIDTH / 2, marker_y,
                   MARKER_WIDTH, marker_h)
        )

   
    # Pelės įvykiai (Adjust Edges)
    def _hit_test_marker(self, x: float) -> Optional[str]:
        """Tikrina ar pelė virš kairiojo ar dešiniojo markerio."""
        if not self._adjust_mode or self._profile is None:
            return None

        left_x = self._sample_to_x(self._left_edge)
        right_x = self._sample_to_x(self._right_edge)

        if abs(x - left_x) <= MARKER_WIDTH / 2 + MARKER_HIT_MARGIN:
            return 'left'
        if abs(x - right_x) <= MARKER_WIDTH / 2 + MARKER_HIT_MARGIN:
            return 'right'

        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._adjust_mode:
            hit = self._hit_test_marker(event.x())
            if hit:
                self._dragging = hit
                self.setCursor(Qt.SizeHorCursor)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._adjust_mode:
            new_sample = self._x_to_sample(event.x())

            if self._dragging == 'left':
                # Neperleisti per dešinį kraštą
                self._left_edge = min(new_sample, self._right_edge - 1)
                self._left_edge = max(0, self._left_edge)
            elif self._dragging == 'right':
                # Neperleisti per kairį kraštą
                self._right_edge = max(new_sample, self._left_edge + 1)
                self._right_edge = min(len(self._profile) - 1,
                                       self._right_edge)

            self.update()
            # Realiu laiku signalas
            self.edges_changed.emit(self._left_edge, self._right_edge)
            event.accept()
            return

        # Hover efektas
        if self._adjust_mode:
            hit = self._hit_test_marker(event.x())
            if hit != self._hover:
                self._hover = hit
                self.setCursor(
                    Qt.SizeHorCursor if hit else Qt.ArrowCursor
                )
                self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = None
            # Galutinis signalas
            self.edges_changed.emit(self._left_edge, self._right_edge)
            event.accept()
            return

        super().mouseReleaseEvent(event)



# MeasurementCard - viena matavimo kortelė

class MeasurementCard(QFrame):
    """
    Vieno matavimo kortelė šoniniame panely.

    Normalus režimas:
        #5 | Vessel: [3 v] | [A] [V] [?]
        Width: 7.23 px (auto)
        Pos: (412, 338)  Zone: 2.1 rOD
        (!) Near bifurcation (12px away)
        [profilio grafikas]
        [Edit] [Delete]

    Redagavimo režimas:
        #5 | EDITING...
        Vessel: [3 v]  Type: [A] [V] [?]
        [Redraw Line] [Adjust Edges] [Done]
        [profilio grafikas su vilkomais markeriais]

    Signals:
        selected(int)               — click ant kortelės
        deleted(int)                — Delete mygtukas
        type_changed(int, int)      — (id, new_type)
        vessel_id_changed(int, int) — (id, new_vessel_id)
        edges_adjusted(int, float, float) — (id, left, right)
        edit_requested(int)         — Edit mygtukas
        redraw_requested(int)       — Redraw Line mygtukas
    """

    selected = pyqtSignal(int)
    deleted = pyqtSignal(int)
    type_changed = pyqtSignal(int, int)
    vessel_id_changed = pyqtSignal(int, int)
    edges_adjusted = pyqtSignal(int, float, float)
    edit_requested = pyqtSignal(int)
    redraw_requested = pyqtSignal(int)

    def __init__(self, measurement: dict, parent=None):
        super().__init__(parent)

        self._measurement = measurement
        self._measurement_id = measurement['id']
        self._is_selected = False
        self._is_editing = False

        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setFixedWidth(CARD_WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.setCursor(Qt.PointingHandCursor)

        self._setup_ui()
        self._update_display()

    def _setup_ui(self):
        """Sukuria UI komponentus."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(3)

        # --- Viršutinė eilutė: #N | Vessel dropdown | Tipo mygtukai ---
        top_row = QHBoxLayout()
        top_row.setSpacing(4)

        self._label_id = QLabel()
        self._label_id.setStyleSheet(
            f"color: {COLOR_TEXT.name()}; font-weight: bold; font-size: 11px;"
        )
        top_row.addWidget(self._label_id)

        # Vessel ID dropdown
        self._vessel_combo = QComboBox()
        self._vessel_combo.setFixedWidth(70)
        self._vessel_combo.setStyleSheet(
            "QComboBox { background: #1E1E1E; color: #FFFFFF; "
            "border: 1px solid #555; font-size: 10px; padding: 2px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { "
            "  background: #1E1E1E; color: #FFFFFF; "
            "  selection-background-color: #2266CC; "
            "  selection-color: #FFFFFF; "
            "  border: 1px solid #555; "
            "  font-size: 10px; "
            "}"
        )
        self._vessel_combo.currentIndexChanged.connect(self._on_vessel_changed)
        top_row.addWidget(self._vessel_combo)

        top_row.addStretch()

        # Tipo mygtukai [A] [V] [?]
        self._btn_artery = self._create_type_button('A', COLOR_ARTERY, 1)
        self._btn_vein = self._create_type_button('V', COLOR_VEIN, 2)
        self._btn_unknown = self._create_type_button('?', COLOR_UNKNOWN, 0)
        top_row.addWidget(self._btn_artery)
        top_row.addWidget(self._btn_vein)
        top_row.addWidget(self._btn_unknown)

        layout.addLayout(top_row)

        # --- Info eilutė: plotis, pozicija ---
        self._label_info = QLabel()
        self._label_info.setStyleSheet(
            f"color: {COLOR_TEXT.name()}; font-size: 10px;"
        )
        self._label_info.setWordWrap(True)
        layout.addWidget(self._label_info)

        # --- Perspėjimas (exclusion zone) ---
        self._label_warning = QLabel()
        self._label_warning.setStyleSheet(
            f"background-color: rgba(255,200,0,40); "
            f"color: {COLOR_WARNING_TEXT.name()}; "
            f"font-size: 10px; padding: 2px 4px; "
            f"border-radius: 2px;"
        )
        self._label_warning.setWordWrap(True)
        self._label_warning.hide()
        layout.addWidget(self._label_warning)

        # --- Profilio grafikas ---
        self._graph = ProfileGraphWidget(self)
        self._graph.edges_changed.connect(self._on_edges_changed)
        layout.addWidget(self._graph)

        # --- Normal mode mygtukai ---
        self._normal_buttons = QWidget()
        btn_layout = QHBoxLayout(self._normal_buttons)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)

        self._btn_edit = QPushButton("Edit")
        self._btn_edit.setFixedHeight(22)
        self._btn_edit.setStyleSheet(self._button_style())
        self._btn_edit.clicked.connect(
            lambda: self.edit_requested.emit(self._measurement_id)
        )
        btn_layout.addWidget(self._btn_edit)

        self._btn_delete = QPushButton("Delete")
        self._btn_delete.setFixedHeight(22)
        self._btn_delete.setStyleSheet(self._button_style('#663333'))
        self._btn_delete.clicked.connect(
            lambda: self.deleted.emit(self._measurement_id)
        )
        btn_layout.addWidget(self._btn_delete)

        btn_layout.addStretch()
        layout.addWidget(self._normal_buttons)

        # --- Edit mode mygtukai ---
        self._edit_buttons = QWidget()
        edit_layout = QHBoxLayout(self._edit_buttons)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_layout.setSpacing(4)

        self._btn_redraw = QPushButton("Redraw Line")
        self._btn_redraw.setFixedHeight(22)
        self._btn_redraw.setStyleSheet(self._button_style('#335566'))
        self._btn_redraw.clicked.connect(
            lambda: self.redraw_requested.emit(self._measurement_id)
        )
        edit_layout.addWidget(self._btn_redraw)

        self._btn_adjust = QPushButton("Adjust Edges")
        self._btn_adjust.setFixedHeight(22)
        self._btn_adjust.setStyleSheet(self._button_style('#335566'))
        self._btn_adjust.setCheckable(True)
        self._btn_adjust.toggled.connect(self._on_adjust_toggled)
        edit_layout.addWidget(self._btn_adjust)

        self._btn_done = QPushButton("Done")
        self._btn_done.setFixedHeight(22)
        self._btn_done.setStyleSheet(self._button_style('#336633'))
        self._btn_done.clicked.connect(self._on_done)
        edit_layout.addWidget(self._btn_done)

        layout.addWidget(self._edit_buttons)
        self._edit_buttons.hide()

    def _create_type_button(self, label: str, color: QColor,
                            vessel_type: int) -> QPushButton:
        """Sukuria tipo mygtuką [A], [V], [?]."""
        btn = QPushButton(label)
        btn.setFixedSize(24, 22)
        btn.setCheckable(True)
        btn.setStyleSheet(
            f"QPushButton {{ background: #3A3A3A; color: {color.name()}; "
            f"border: 1px solid #555; font-weight: bold; font-size: 11px; }}"
            f"QPushButton:checked {{ background: {color.name()}; color: #1E1E1E; "
            f"border: 1px solid {color.lighter(130).name()}; }}"
        )
        btn.clicked.connect(lambda: self._on_type_clicked(vessel_type))
        return btn

    @staticmethod
    def _button_style(bg: str = '#3A3A3A') -> str:
        return (
            f"QPushButton {{ background: {bg}; color: #E0E0E0; "
            f"border: 1px solid #555; font-size: 10px; padding: 2px 6px; }}"
            f"QPushButton:hover {{ background: #555; }}"
            f"QPushButton:pressed {{ background: #666; }}"
        )

   
    # Duomenų atnaujinimas

    def update_measurement(self, measurement: dict):
        """Atnaujina kortelę su naujais matavimo duomenimis."""
        self._measurement = measurement
        self._update_display()

    def _update_display(self):
        """Atnaujina visus UI elementus pagal measurement dict."""
        m = self._measurement

        # ID
        self._label_id.setText(f"#{m['id']}")

        # Tipo mygtukai
        vtype = m.get('vessel_type', 0)
        self._btn_artery.setChecked(vtype == 1)
        self._btn_vein.setChecked(vtype == 2)
        self._btn_unknown.setChecked(vtype == 0)

        # Info
        width = m.get('width_manual') if m.get('width_manual') is not None else m.get('width_px', 0)
        has_edges = m.get('profile_edges') is not None or m.get('profile_edges_manual') is not None
        width_str = f"{width:.2f} px" if has_edges else "-- px"
        cx = m.get('cx', 0)
        cy = m.get('cy', 0)
        zone_rod = m.get('zone_rod', 0)

        self._label_info.setText(
            f"Width: {width_str}\n"
            f"Pos: ({cx:.0f}, {cy:.0f})  Zone: {zone_rod:.1f} rOD"
        )

        # Perspėjimas
        if m.get('in_exclusion_zone'):
            from exclusion_zones import ExclusionZoneComputer
            warning = ExclusionZoneComputer.format_warning(
                True, m.get('exclusion_reason', '')
            )
            self._label_warning.setText(warning)
            self._label_warning.show()
        else:
            self._label_warning.hide()

        # Profilio grafikas
        profile = m.get('profile')
        if profile is not None and len(profile) > 1:
            manual_edges = m.get('profile_edges_manual')
            ml = manual_edges[0] if manual_edges else None
            mr = manual_edges[1] if manual_edges else None
            edges = m.get('profile_edges')

            self._graph.set_profile_data(
                profile=profile,
                left_edge=edges[0] if edges else None,
                right_edge=edges[1] if edges else None,
                threshold=m.get('threshold'),
                samples_per_pixel=m.get('samples_per_pixel', 1),
                manual_left=ml,
                manual_right=mr
            )

        # Rėmelio spalva
        self._update_border()

    def _update_border(self):
        """Atnaujina kortelės rėmelį pagal pasirinkimo būseną."""
        if self._is_selected:
            self.setStyleSheet(
                f"MeasurementCard {{ "
                f"background-color: {COLOR_CARD_BG.name()}; "
                f"border: 2px solid {COLOR_SELECTED_BORDER.name()}; }}"
            )
        else:
            self.setStyleSheet(
                f"MeasurementCard {{ "
                f"background-color: {COLOR_CARD_BG.name()}; "
                f"border: 1px solid {COLOR_CARD_BORDER.name()}; }}"
            )

   
    # Vessel ID dropdown

    def update_vessel_options(self, vessel_ids_info: Dict[int, dict],
                              current_vessel_id: int):
        """
        Atnaujina vessel ID dropdown su esamais ID.

        Args:
            vessel_ids_info: {vid: {'type': int, 'count': int}}
            current_vessel_id: Dabartinis šio matavimo vessel_id
        """
        self._vessel_combo.blockSignals(True)
        self._vessel_combo.clear()

        # "New" opcija
        self._vessel_combo.addItem("New", -1)

        # Esami vessel ID
        for vid, info in sorted(vessel_ids_info.items()):
            type_label = TYPE_LABELS.get(info['type'], '?')
            count = info['count']
            self._vessel_combo.addItem(
                f"{vid} ({type_label}, {count})", vid
            )

        # Pasirinkti dabartinį
        for i in range(self._vessel_combo.count()):
            if self._vessel_combo.itemData(i) == current_vessel_id:
                self._vessel_combo.setCurrentIndex(i)
                break

        self._vessel_combo.blockSignals(False)

   
    # Pasirinkimas

    def set_selected(self, selected: bool):
        self._is_selected = selected
        self._update_border()

    def is_selected(self) -> bool:
        return self._is_selected

   
    # Redagavimo režimas

    def set_editing(self, editing: bool):
        """Įjungia/išjungia redagavimo režimą."""
        self._is_editing = editing
        self._normal_buttons.setVisible(not editing)
        self._edit_buttons.setVisible(editing)

        if not editing:
            self._btn_adjust.setChecked(False)
            self._graph.set_adjust_mode(False)

    def _on_done(self):
        """Done mygtukas — išjungia redagavimo režimą."""
        self.set_editing(False)

    def _on_adjust_toggled(self, checked: bool):
        """Adjust Edges toggle."""
        self._graph.set_adjust_mode(checked)

   
    # Signalų apdorojimas

    def _on_type_clicked(self, vessel_type: int):
        """Tipo mygtukas paspaustas."""
        self.type_changed.emit(self._measurement_id, vessel_type)

    def _on_vessel_changed(self, index: int):
        """Vessel dropdown pakeistas."""
        vid = self._vessel_combo.itemData(index)
        if vid is not None:
            # vid == -1 reiškia "New" → main.py priskirs sekantį ID
            self.vessel_id_changed.emit(self._measurement_id, vid)

    def _on_edges_changed(self, left: float, right: float):
        """Kraštai pakeisti grafike — perduoda signalą aukščiau."""
        self.edges_adjusted.emit(self._measurement_id, left, right)

   
    # Mouse events

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selected.emit(self._measurement_id)
        super().mousePressEvent(event)

    @property
    def measurement_id(self) -> int:
        return self._measurement_id



# ProfilePanel — pagrindinis šoninine panele

class ProfilePanel(QWidget):
    """
    Šonine panele su visomis matavimo kortelėmis.

    QScrollArea su vertikaliu QVBoxLayout.
    Kortelės surikiuotos pagal matavimo ID.

    Signals:
        measurement_selected(int)
        measurement_deleted(int)
        type_changed(int, int)
        vessel_id_changed(int, int)
        edges_adjusted(int, float, float)
        edit_requested(int)
        redraw_requested(int)
    """

    measurement_selected = pyqtSignal(int)
    measurement_deleted = pyqtSignal(int)
    type_changed = pyqtSignal(int, int)
    vessel_id_changed = pyqtSignal(int, int)
    edges_adjusted = pyqtSignal(int, float, float)
    edit_requested = pyqtSignal(int)
    redraw_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(CARD_WIDTH + 20)
        self.setMaximumWidth(CARD_WIDTH + 30)

        # {measurement_id: MeasurementCard}
        self._cards: Dict[int, MeasurementCard] = {}

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("  Measurements")
        header.setStyleSheet(
            f"color: {COLOR_TEXT.name()}; font-weight: bold; "
            f"font-size: 12px; padding: 4px; "
            f"background-color: {COLOR_CARD_BG.name()};"
        )
        header.setFixedHeight(24)
        main_layout.addWidget(header)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setStyleSheet(
            "QScrollArea { border: none; background: #1E1E1E; }"
        )

        # Kortelių konteineris
        self._container = QWidget()
        self._container.setStyleSheet("background: #1E1E1E;")
        self._cards_layout = QVBoxLayout(self._container)
        self._cards_layout.setContentsMargins(4, 4, 4, 4)
        self._cards_layout.setSpacing(CARD_SPACING)
        self._cards_layout.addStretch()

        self._scroll.setWidget(self._container)
        main_layout.addWidget(self._scroll)

   
    # Kortelių valdymas

    def add_card(self, measurement: dict):
        """Prideda naują kortelę ir prijungia signalus."""
        mid = measurement['id']

        if mid in self._cards:
            self._cards[mid].update_measurement(measurement)
            return

        card = MeasurementCard(measurement, self._container)

        # Signalų jungimas
        card.selected.connect(self._on_card_selected)
        card.deleted.connect(self.measurement_deleted.emit)
        card.type_changed.connect(self.type_changed.emit)
        card.vessel_id_changed.connect(self.vessel_id_changed.emit)
        card.edges_adjusted.connect(self.edges_adjusted.emit)
        card.edit_requested.connect(self._on_edit_requested)
        card.redraw_requested.connect(self.redraw_requested.emit)

        self._cards[mid] = card

        # Įterpti prieš stretch
        idx = self._cards_layout.count() - 1  # Prieš stretch
        self._cards_layout.insertWidget(idx, card)

        # Auto-scroll prie naujausios
        QTimer.singleShot(50, self._scroll_to_latest)

    def update_card(self, measurement_id: int, measurement: dict):
        """Atnaujina kortelę po redagavimo."""
        if measurement_id in self._cards:
            self._cards[measurement_id].update_measurement(measurement)

    def remove_card(self, measurement_id: int):
        """Pašalina kortelę."""
        if measurement_id in self._cards:
            card = self._cards.pop(measurement_id)
            self._cards_layout.removeWidget(card)
            card.deleteLater()

    def insert_card(self, measurement: dict):
        """Įterpia kortelę (redo atveju)."""
        self.add_card(measurement)

    def clear_cards(self):
        """Pašalina visas korteles."""
        for mid in list(self._cards.keys()):
            self.remove_card(mid)

   
    # Kortelių atnaujinimas (vessel dropdown)

    def update_vessel_options(self, vessel_ids_info: Dict[int, dict],
                              measurements: List[dict]):
        """
        Atnaujina vessel ID dropdown visose kortelėse.

        Args:
            vessel_ids_info: {vid: {'type', 'count'}}
            measurements: Matavimų sąrašas (su vessel_id)
        """
        for m in measurements:
            mid = m['id']
            if mid in self._cards:
                self._cards[mid].update_vessel_options(
                    vessel_ids_info, m['vessel_id']
                )

   
    # Pasirinkimas

    def highlight_card(self, measurement_id: int):
        """Paryškina kortelę ir nuima senąjį pasirinkimą."""
        for mid, card in self._cards.items():
            card.set_selected(mid == measurement_id)

        # Scroll prie pasirinktos
        if measurement_id in self._cards:
            self._scroll.ensureWidgetVisible(
                self._cards[measurement_id]
            )

    def _on_card_selected(self, measurement_id: int):
        """Kortelė paspausdinta — perduoda signalą."""
        self.highlight_card(measurement_id)
        self.measurement_selected.emit(measurement_id)

   
    # Redagavimas

    def _on_edit_requested(self, measurement_id: int):
        """Edit mygtukas — įjungia redagavimo režimą kortelėje."""
        if measurement_id in self._cards:
            self._cards[measurement_id].set_editing(True)
        self.edit_requested.emit(measurement_id)

    def set_card_editing(self, measurement_id: int, editing: bool):
        """Nustatyti redagavimo režimą iš išorės."""
        if measurement_id in self._cards:
            self._cards[measurement_id].set_editing(editing)

   
    # Pagalbinės

    def _scroll_to_latest(self):
        """Automatiškai nuscrollina prie naujausios kortelės."""
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def get_card_count(self) -> int:
        return len(self._cards)