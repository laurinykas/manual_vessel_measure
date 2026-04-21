"""
exclusion_zones.py - Rankinė draudžiamų zonų žymėjimas

Medikas pats nuspalvina zonas kur nematuos (bifurkacijos, kryžmos,
per ploni segmentai ir t.t.). Piešimas vyksta viewer_widget per
PAINT_EXCLUSION režimą.

Saugojimas:
    Binarinė kaukė (0=OK, 255=draudžiama) — saugoma sesijoje kaip
    RLE (run-length encoding) kad neužimtų daug vietos JSON'e.

Vizualizacija:
    RGBA overlay — raudona spalva su alpha=80 ant draudžiamų zonų.
    Toggle R klavišu.
"""

from typing import Tuple, Optional

import cv2
import numpy as np



# Spalvos


ZONE_COLOR = [255, 60, 60, 80]  # Raudona su alpha



# ExclusionZoneComputer


class ExclusionZoneComputer:
    """
    Rankinė draudžiamų zonų sistema.

    Medikas piešia ant vaizdo — spalvina zonas kur nematuos.
    Brush dydis reguliuojamas. Shift+brush = trintukas.
    """

    MAX_UNDO = 30

    def __init__(self):
        self._mask: Optional[np.ndarray] = None   # uint8 (H,W), 0 arba 255
        self.overlay_rgba: Optional[np.ndarray] = None
        self.markers_rgba: Optional[np.ndarray] = None  # None (suderinamumui)
        self.brush_size: int = 20
        self._undo_stack: list = []
        self._redo_stack: list = []

   
    # Inicializacija
   

    def init_mask(self, h: int, w: int):
        """Sukuria tuščią kaukę naujam vaizdui."""
        self._mask = np.zeros((h, w), dtype=np.uint8)
        self.overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        self._undo_stack.clear()
        self._redo_stack.clear()

    @property
    def is_initialized(self) -> bool:
        return self._mask is not None

    def get_mask(self) -> Optional[np.ndarray]:
        """Grąžina binarinę kaukę (0/255) arba None."""
        return self._mask

   
    # Piešimas
   

    def paint(self, x: int, y: int, erase: bool = False):
        """
        Nupiešia arba ištrina apskritimą draudžiamoje zonoje.

        Args:
            x, y: Centro koordinatės (vaizdo erdvė)
            erase: True = trintukas (pašalina zoną)
        """
        if self._mask is None:
            return

        color = 0 if erase else 255
        cv2.circle(self._mask, (x, y), self.brush_size, int(color), -1)
        self._update_overlay_region(x, y)

    def paint_line(self, x1: int, y1: int, x2: int, y2: int,
                   erase: bool = False):
        """
        Nupiešia liniją tarp dviejų taškų (interpoliuoja tarp
        mouse move žingsnių).
        """
        if self._mask is None:
            return

        color = 0 if erase else 255
        cv2.line(self._mask, (x1, y1), (x2, y2), int(color),
                 self.brush_size * 2)
        self._update_overlay_region_line(x1, y1, x2, y2)

   
    # Undo / Redo
   

    def save_snapshot(self):
        """Išsaugo dabartinę kaukę prieš naują potėpį."""
        if self._mask is None:
            return
        self._undo_stack.append(self._mask.copy())
        self._redo_stack.clear()
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)

    def undo(self) -> bool:
        """Atkuria ankstesnę kaukės būseną. Grąžina True jei pavyko."""
        if not self._undo_stack or self._mask is None:
            return False
        self._redo_stack.append(self._mask.copy())
        self._mask = self._undo_stack.pop()
        self.rebuild_overlay()
        return True

    def redo(self) -> bool:
        """Grąžina atšauktą potėpį. Grąžina True jei pavyko."""
        if not self._redo_stack or self._mask is None:
            return False
        self._undo_stack.append(self._mask.copy())
        self._mask = self._redo_stack.pop()
        self.rebuild_overlay()
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

   
    # Overlay
   

    def rebuild_overlay(self):
        """Perkuria visą RGBA overlay iš kaukės."""
        if self._mask is None:
            return

        h, w = self._mask.shape
        self.overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        zone = self._mask > 0
        self.overlay_rgba[zone] = ZONE_COLOR

    def _update_overlay_region(self, cx: int, cy: int):
        """Atnaujina tik brush regioną overlay."""
        if self._mask is None or self.overlay_rgba is None:
            return

        h, w = self._mask.shape
        r = self.brush_size + 2
        y0 = max(0, cy - r)
        y1 = min(h, cy + r)
        x0 = max(0, cx - r)
        x1 = min(w, cx + r)

        region_mask = self._mask[y0:y1, x0:x1]
        region_overlay = self.overlay_rgba[y0:y1, x0:x1]

        region_overlay[:] = 0
        zone = region_mask > 0
        region_overlay[zone] = ZONE_COLOR

    def _update_overlay_region_line(self, x1: int, y1: int,
                                     x2: int, y2: int):
        """Atnaujina overlay regioną aplink liniją."""
        if self._mask is None or self.overlay_rgba is None:
            return

        h, w = self._mask.shape
        r = self.brush_size + 2
        ya = max(0, min(y1, y2) - r)
        yb = min(h, max(y1, y2) + r)
        xa = max(0, min(x1, x2) - r)
        xb = min(w, max(x1, x2) + r)

        region_mask = self._mask[ya:yb, xa:xb]
        region_overlay = self.overlay_rgba[ya:yb, xa:xb]

        region_overlay[:] = 0
        zone = region_mask > 0
        region_overlay[zone] = ZONE_COLOR

   
    # Matavimo patikrinimas
   

    def check_measurement(self, cx: float, cy: float,
                          od_x: int = 0, od_y: int = 0, od_r: int = 0
                          ) -> Tuple[bool, str]:
        """
        Patikrina ar matavimo centras patenka į draudžiamą zoną.
        """
        if self._mask is None:
            return False, ""

        ix = int(round(cx))
        iy = int(round(cy))
        h, w = self._mask.shape

        if not (0 <= ix < w and 0 <= iy < h):
            return False, ""

        if self._mask[iy, ix] > 0:
            return True, "excluded_zone"

        return False, ""

    @staticmethod
    def format_warning(in_zone: bool, reason: str) -> str:
        """Perspėjimo tekstas kortelei."""
        if not in_zone:
            return ""
        return "(!) In excluded zone"

   
    # Brush dydis
   

    def increase_brush(self, step: int = 5):
        self.brush_size = min(200, self.brush_size + step)

    def decrease_brush(self, step: int = 5):
        self.brush_size = max(2, self.brush_size - step)

   
    # Serializacija (session save/load)
   

    def serialize_mask(self) -> Optional[dict]:
        """
        Konvertuoja kaukę į RLE formatą JSON saugojimui.
        Numpy-based RLE — greita net su 3504×2336 vaizdais.

        Returns:
            {'width', 'height', 'start_value', 'rle'} arba None
        """
        if self._mask is None or not self.has_zones:
            return None

        h, w = self._mask.shape
        flat = (self._mask.ravel() > 0).astype(np.uint8)

        # Numpy RLE: rasti vietas kur reikšmė keičiasi
        changes = np.where(np.diff(flat) != 0)[0] + 1
        # Run ilgiai: tarpai tarp change pozicijų
        boundaries = np.concatenate(([0], changes, [len(flat)]))
        runs = np.diff(boundaries).tolist()

        return {
            'width': w,
            'height': h,
            'start_value': int(flat[0]),
            'rle': runs,
        }

    def deserialize_mask(self, data: dict) -> bool:
        """Atkuria kaukę iš RLE formato."""
        try:
            w = data['width']
            h = data['height']
            start = data['start_value']
            runs = data['rle']

            flat = np.zeros(h * w, dtype=np.uint8)
            pos = 0
            current = start
            for run_len in runs:
                if current:
                    flat[pos:pos + run_len] = 255
                pos += run_len
                current = 1 - current

            self._mask = flat.reshape(h, w)
            self.rebuild_overlay()
            return True

        except (KeyError, ValueError) as e:
            print(f"[ExclusionZone] Deserialize klaida: {e}")
            return False

   
    # Reset / Clear

    def clear(self):
        """Išvalo visą kaukę (palieka inicializuotą)."""
        if self._mask is not None:
            self._mask[:] = 0
            self.rebuild_overlay()

    def reset(self):
        """Pilnas reset (naujas vaizdas)."""
        self._mask = None
        self.overlay_rgba = None
        self.markers_rgba = None
        self._undo_stack.clear()
        self._redo_stack.clear()

    @property
    def is_computed(self) -> bool:
        return self._mask is not None

    @property
    def has_zones(self) -> bool:
        if self._mask is None:
            return False
        return np.any(self._mask > 0)

    @property
    def junction_count(self):
        """Suderinamumui — grąžina (0, 0)."""
        return (0, 0)