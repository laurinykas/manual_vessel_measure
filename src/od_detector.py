"""
od_detector.py - Optinio disko aptikimas ir rankinis perrašymas

Automatinis režimas:
    Kviečia pilną pipeline'ą: masking → preprocessing1 → bwe1 → detect_optic_disc

Rankinis režimas (Shift+D → freehand lasso):
    Medikas piešia laisvą liniją aplink OD, cv2.minEnclosingCircle
    fitina apskritimą iš kontūro taškų.

Naudojimas iš main.py / viewer_widget.py:
    detector = ODDetector()
    result = detector.auto_detect(image_bgr)
    result = detector.set_from_contour(points)
    detector.undo()
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List

from pipeline.masking import create_fundus_mask
from pipeline.preprocessing import preprocessing1
from pipeline.vessel_extraction import bwe1
from pipeline.optic_disc import detect_optic_disc


class ODResult:
    """Optinio disko aptikimo rezultatas."""

    __slots__ = ('x', 'y', 'r', 'is_manual')

    def __init__(self, x: int = 0, y: int = 0, r: int = 0,
                 is_manual: bool = False):
        self.x = x
        self.y = y
        self.r = r
        self.is_manual = is_manual

    @property
    def is_valid(self) -> bool:
        """Ar OD aptiktas (spindulys > 0)."""
        return self.r > 0

    def as_tuple(self) -> Tuple[int, int, int]:
        """Grąžina (x, y, r) tuple."""
        return (self.x, self.y, self.r)

    def copy(self) -> 'ODResult':
        return ODResult(self.x, self.y, self.r, self.is_manual)

    def __repr__(self) -> str:
        mode = "manual" if self.is_manual else "auto"
        return f"ODResult(x={self.x}, y={self.y}, r={self.r}, {mode})"


class ODDetector:
    """
    Optinio disko detektorius su automatine ir rankine detekcija.

    Palaiko undo/redo per išorinį MeasurementManager arba
    vidinį _history stack'ą.
    """

    def __init__(self):
        self._result: Optional[ODResult] = None
        self._history: List[Optional[ODResult]] = []  # undo stack

        # Tarpiniai pipeline rezultatai (reikalingi exclusion_zones)
        self._img_mask: Optional[np.ndarray] = None
        self._sc: Optional[float] = None
        self._img_proc_thn: Optional[np.ndarray] = None

   
    # Properties

    @property
    def result(self) -> Optional[ODResult]:
        """Dabartinis OD rezultatas."""
        return self._result

    @property
    def od_x(self) -> int:
        return self._result.x if self._result else 0

    @property
    def od_y(self) -> int:
        return self._result.y if self._result else 0

    @property
    def od_r(self) -> int:
        return self._result.r if self._result else 0

    @property
    def is_valid(self) -> bool:
        return self._result is not None and self._result.is_valid

    @property
    def is_manual(self) -> bool:
        return self._result is not None and self._result.is_manual

    @property
    def img_mask(self) -> Optional[np.ndarray]:
        """Akies dugno kaukė (sukurta automatinės detekcijos metu)."""
        return self._img_mask

    @property
    def sc(self) -> Optional[float]:
        """Skalės koeficientas."""
        return self._sc

    @property
    def img_proc_thn(self) -> Optional[np.ndarray]:
        """Suplonintas kraujagyslių vaizdas (reikalingas exclusion_zones)."""
        return self._img_proc_thn

   
    # Automatinis aptikimas

    def auto_detect(self, image_bgr: np.ndarray) -> ODResult:
        """
        Pilnas automatinis OD aptikimas per pipeline'ą.

        Pipeline žingsniai:
        1. createMask() → img_mask, sc
        2. preprocessing1() → apdorotas vaizdas
        3. bwe1() → binarinė + suploninta kraujagyslių nuotrauka
        4. detect_optic_disc() → (od_x, od_y, od_r)

        Args:
            image_bgr: Originalus BGR paveikslėlis

        Returns:
            ODResult su aptiktu OD
        """
        import sys

        # Išsaugoti seną rezultatą į istoriją
        self._push_history()

        h, w = image_bgr.shape[:2]
        print(f"[OD] auto_detect START — image size: {w}x{h}", flush=True)

        # 1. Kaukė ir skalės koeficientas
        print("[OD] Step 1: createMask...", flush=True)
        img_green = image_bgr[:, :, 1]
        self._img_mask, self._sc = create_fundus_mask(img_green)
        print(f"[OD] Step 1 DONE — sc={self._sc:.3f}", flush=True)

        # 2. Preprocessing kraujagyslėms
        print("[OD] Step 2: preprocessing1...", flush=True)
        sys.stdout.flush()
        img_preprocessed = preprocessing1(image_bgr, self._img_mask, self._sc)
        img_green_processed = img_preprocessed[:, :, 1]
        print("[OD] Step 2 DONE", flush=True)

        # 3. Kraujagyslių išskyrimas + ploninimas
        print("[OD] Step 3: bwe1 (vessel extraction + thinning)...", flush=True)
        sys.stdout.flush()
        _, self._img_proc_thn = bwe1(img_green_processed, self._img_mask,
                                     self._sc)
        print("[OD] Step 3 DONE", flush=True)

        # 4. OD aptikimas
        print("[OD] Step 4: detect_optic_disc...", flush=True)
        sys.stdout.flush()
        od_x, od_y, od_r = detect_optic_disc(
            img_green_processed, self._img_proc_thn,
            self._img_mask, self._sc
        )
        print(f"[OD] Step 4 DONE — od=({od_x}, {od_y}), r={od_r}", flush=True)

        self._result = ODResult(od_x, od_y, od_r, is_manual=False)
        return self._result

   
    # Rankinis perrašymas (freehand lasso)

    def set_from_contour(self, points: List[Tuple[float, float]]) -> ODResult:
        """
        Nustato OD iš freehand lasso kontūro taškų.

        Medikas piešia laisvą liniją aplink OD viewer_widget'e,
        taškai surenkami į sąrašą. Čia fitinamas mažiausias
        apgaubiantis apskritimas.

        Args:
            points: Kontūro taškai [(x1,y1), (x2,y2), ...] vaizdo
                    koordinatėmis. Minimaliai 3 taškai.

        Returns:
            ODResult su rankiniu OD

        Raises:
            ValueError: Jei mažiau nei 3 taškai
        """
        if len(points) < 3:
            raise ValueError(
                f"Reikia bent 3 taškų apskritimui fitinti, "
                f"gauta: {len(points)}"
            )

        # Išsaugoti seną rezultatą į istoriją
        self._push_history()

        # Konvertuoti į numpy kontūro formatą
        contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

        # Mažiausias apgaubiantis apskritimas
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        self._result = ODResult(
            x=int(round(cx)),
            y=int(round(cy)),
            r=int(round(radius)),
            is_manual=True
        )
        return self._result

    def set_manual(self, x: int, y: int, r: int) -> ODResult:
        """
        Tiesiogiai nustato OD koordinates (pvz. iš sesijos JSON).

        Args:
            x, y: OD centro koordinatės
            r: OD spindulys

        Returns:
            ODResult
        """
        self._push_history()
        self._result = ODResult(x, y, r, is_manual=True)
        return self._result

   
    # Undo

    def undo(self) -> bool:
        """
        Grąžina ankstesnį OD rezultatą.

        Returns:
            True jei pavyko, False jei istorija tuščia
        """
        if not self._history:
            return False
        self._result = self._history.pop()
        return True

    def _push_history(self):
        """Išsaugo dabartinį rezultatą į undo istoriją."""
        if self._result is not None:
            self._history.append(self._result.copy())
        else:
            self._history.append(None)

        # Limitas
        if len(self._history) > 20:
            self._history.pop(0)

   
    # Reset

    def reset(self):
        """Išvalo viską (naujas vaizdas)."""
        self._result = None
        self._history.clear()
        self._img_mask = None
        self._sc = None
        self._img_proc_thn = None