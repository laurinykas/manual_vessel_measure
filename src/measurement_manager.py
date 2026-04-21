"""
measurement_manager.py - Matavimų valdymas su UNDO/REDO

Pagrindinė projekto logika:
- Matavimų struktūra (dict)
- Profilio ištraukimas iš žalio kanalo (bilinear interpolation, 3x oversampling)
- Pločio skaičiavimas HHFW metodu (Half-Height Full-Width)
- Pilnas UNDO/REDO su visais veiksmų tipais
- Auto-save į JSON (~/.manual_vessel_measure/autosave.json)
"""

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np



# Bilinear interpolation


def bilinear_interpolate(image: np.ndarray, x: float, y: float) -> float:
    """
    Bilinearinė interpoliacija vieno kanalo paveiksliuke.

    Args:
        image: 2D numpy masyvas (H, W)
        x: X koordinatė (float, sub-pixel)
        y: Y koordinatė (float, sub-pixel)

    Returns:
        Interpoliuota intensyvumo reikšmė
    """
    h, w = image.shape[:2]
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))

    fx = x - math.floor(x)
    fy = y - math.floor(y)

    return float(
        image[y0, x0] * (1 - fx) * (1 - fy) +
        image[y0, x1] * fx * (1 - fy) +
        image[y1, x0] * (1 - fx) * fy +
        image[y1, x1] * fx * fy
    )



# Action (undo/redo)

# Veiksmų tipai
ACTION_ADD = 'add'
ACTION_DELETE = 'delete'
ACTION_EDIT_TYPE = 'edit_type'
ACTION_EDIT_VESSEL_ID = 'edit_vessel_id'
ACTION_EDIT_LINE = 'edit_line'
ACTION_EDIT_EDGES = 'edit_edges'

# Veiksmų aprašymai (UI)
_UNDO_DESCRIPTIONS = {
    ACTION_ADD: "Undo: remove measurement",
    ACTION_DELETE: "Undo: restore measurement",
    ACTION_EDIT_TYPE: "Undo: type change",
    ACTION_EDIT_VESSEL_ID: "Undo: vessel ID change",
    ACTION_EDIT_LINE: "Undo: line redraw",
    ACTION_EDIT_EDGES: "Undo: edge adjustment",
}

_REDO_DESCRIPTIONS = {
    ACTION_ADD: "Redo: add measurement",
    ACTION_DELETE: "Redo: delete measurement",
    ACTION_EDIT_TYPE: "Redo: type change",
    ACTION_EDIT_VESSEL_ID: "Redo: vessel ID change",
    ACTION_EDIT_LINE: "Redo: line redraw",
    ACTION_EDIT_EDGES: "Redo: edge adjustment",
}


def _create_action(action_type: str, image_name: str,
                   measurement_id: int,
                   old_state: Optional[dict],
                   new_state: Optional[dict]) -> dict:
    """Sukuria veiksmo struktūrą undo/redo stack'ui."""
    return {
        'type': action_type,
        'image_name': image_name,
        'measurement_id': measurement_id,
        'old_state': old_state,
        'new_state': new_state,
    }



# Vessel type konstantos

VESSEL_UNKNOWN = 0
VESSEL_ARTERY = 1
VESSEL_VEIN = 2



# MeasurementManager

class MeasurementManager:
    """
    Centrinė matavimų valdymo klasė.

    Atsakinga už:
    - Matavimų pridėjimą / šalinimą / redagavimą
    - Profilio ištraukimą iš žalio kanalo
    - Pločio skaičiavimą HHFW metodu
    - UNDO / REDO su pilna būsenos kopija
    - Auto-save į JSON
    """

    MAX_UNDO = 100
    AUTOSAVE_DIR = os.path.join(Path.home(), '.manual_vessel_measure')
    AUTOSAVE_FILE = 'autosave.json'

    def __init__(self):
        # {image_name: [measurement_dict, ...]}
        self.measurements: Dict[str, List[dict]] = {}

        # Undo / Redo stacks
        self.undo_stack: List[dict] = []
        self.redo_stack: List[dict] = []

        # Sekantis measurement ID per vaizdą
        self._next_ids: Dict[str, int] = {}

        # Exclusion zone kompiuteris (nustatomas iš išorės)
        self.exclusion_computer = None

        # OD duomenys (nustatomi iš išorės per od_detector)
        self._od_data: Dict[str, Tuple[int, int, int]] = {}

        # Pilni vaizdo keliai (session atkūrimui)
        self._image_paths: Dict[str, str] = {}

        # Exclusion kaukių RLE duomenys per vaizdą
        self._exclusion_masks: Dict[str, dict] = {}

   
    # Vaizdo kelių registracija

    def register_image_path(self, image_name: str, full_path: str):
        """Registruoja pilną vaizdo kelią sesijos atkūrimui."""
        self._image_paths[image_name] = full_path

    def get_image_path(self, image_name: str) -> Optional[str]:
        """Grąžina pilną vaizdo kelią arba None."""
        return self._image_paths.get(image_name)

    def get_all_image_paths(self) -> Dict[str, str]:
        """Grąžina visus registruotus vaizdo kelius."""
        return dict(self._image_paths)

   
    # Exclusion kaukių saugojimas per vaizdą

    def store_exclusion_mask(self, image_name: str, rle_data: Optional[dict]):
        """Saugo exclusion kaukės RLE duomenis konkrečiam vaizdui."""
        if rle_data is not None:
            self._exclusion_masks[image_name] = rle_data
        elif image_name in self._exclusion_masks:
            del self._exclusion_masks[image_name]

    def get_exclusion_mask(self, image_name: str) -> Optional[dict]:
        """Grąžina exclusion kaukės RLE duomenis arba None."""
        return self._exclusion_masks.get(image_name)

   
    # ID generatoriai

    def _next_id(self, image_name: str) -> int:
        """Sekantis unikalus matavimo ID šiam vaizdui."""
        self._ensure_image(image_name)
        current = self._next_ids[image_name]
        self._next_ids[image_name] = current + 1
        return current

    def _next_vessel_id(self, image_name: str) -> int:
        """Mažiausias nenaudojamas vessel ID šiam vaizdui."""
        self._ensure_image(image_name)
        used = set()
        for m in self.measurements.get(image_name, []):
            used.add(m['vessel_id'])
        # Rasti mažiausią nenaudojamą ID (1, 2, 3, ...)
        vid = 1
        while vid in used:
            vid += 1
        return vid

    def _ensure_image(self, image_name: str):
        """Užtikrina, kad vaizdas turi sąrašą ir ID skaitliukus."""
        if image_name not in self.measurements:
            self.measurements[image_name] = []
            self._next_ids[image_name] = 1

    # OD duomenys
   

    def set_od_data(self, image_name: str, od_x: int, od_y: int, od_r: int):
        """Nustato OD duomenis konkrečiam vaizdui."""
        self._od_data[image_name] = (od_x, od_y, od_r)

    def get_od_data(self, image_name: str) -> Tuple[int, int, int]:
        """Grąžina OD duomenis arba (0, 0, 0)."""
        return self._od_data.get(image_name, (0, 0, 0))

   
    # Profilio ir pločio skaičiavimas
   

    PROFILE_PADDING = 0.3  # 30% padding kiekvienoje pusėje

    @staticmethod
    def extract_profile(green_channel: np.ndarray,
                        x1: float, y1: float,
                        x2: float, y2: float
                        ) -> Tuple[Optional[np.ndarray], float, float,
                                   float, float, float, float]:
        """
        Ištraukia intensyvumo profilį su padding'u abiejose pusėse.

        Profilis ištemptas 30% už linijos ribų kiekvienoje pusėje,
        kad medikas matytų kontekstą ir galėtų koreguoti kraštus.

        3x oversampling — kiekvienas pikselis samplinamas 3 kartus.

        Args:
            green_channel: Žalio kanalo 2D masyvas
            x1, y1: Linijos pradžia (vaizdo koordinatės, float)
            x2, y2: Linijos pabaiga

        Returns:
            (profile, length, spp, ext_x1, ext_y1, ext_x2, ext_y2):
            - profile: 1D numpy masyvas su intensyvumais (arba None)
            - length: Originalios linijos ilgis pikseliais
            - samples_per_pixel: Kiek sample'ų tenka vienam pikseliui
            - ext_x1..ext_y2: Išplėstos linijos koordinatės (su padding)
        """
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)

        if length < 1.0:
            return None, 0.0, 0.0, x1, y1, x2, y2

        # Krypties vektorius
        ux = dx / length
        uy = dy / length

        # Padding pikseliais
        pad_px = length * MeasurementManager.PROFILE_PADDING

        # Išplėstos linijos koordinatės
        ext_x1 = x1 - ux * pad_px
        ext_y1 = y1 - uy * pad_px
        ext_x2 = x2 + ux * pad_px
        ext_y2 = y2 + uy * pad_px

        # Clamp į vaizdo ribas
        h, w = green_channel.shape[:2]
        ext_x1 = max(0, min(w - 1, ext_x1))
        ext_y1 = max(0, min(h - 1, ext_y1))
        ext_x2 = max(0, min(w - 1, ext_x2))
        ext_y2 = max(0, min(h - 1, ext_y2))

        # Išplėstos linijos ilgis
        ext_dx = ext_x2 - ext_x1
        ext_dy = ext_y2 - ext_y1
        ext_length = math.sqrt(ext_dx * ext_dx + ext_dy * ext_dy)

        if ext_length < 1.0:
            return None, 0.0, 0.0, x1, y1, x2, y2

        num_samples = max(3, int(ext_length * 3))  # 3x oversampling
        spp = num_samples / ext_length  # samples per pixel

        profile = np.zeros(num_samples, dtype=np.float64)
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.5
            px = ext_x1 + t * ext_dx
            py = ext_y1 + t * ext_dy
            profile[i] = bilinear_interpolate(green_channel, px, py)

        # Normalizacija į 0–255 diapazoną
        p_min = profile.min()
        p_max = profile.max()
        if p_max - p_min > 1e-6:
            profile = (profile - p_min) / (p_max - p_min) * 255.0
        else:
            profile[:] = 128.0

        return profile, length, spp, ext_x1, ext_y1, ext_x2, ext_y2


   
    # Exclusion zone patikrinimas
   

    def _check_exclusion(self, measurement: dict, image_name: str):
        """
        Patikrina ar matavimo centras patenka į draudžiamą zoną.
        Užpildo: in_exclusion_zone, exclusion_reason.
        """
        measurement['in_exclusion_zone'] = False
        measurement['exclusion_reason'] = ""

        if self.exclusion_computer is None:
            return

        od_x, od_y, od_r = self.get_od_data(image_name)

        in_zone, reason = self.exclusion_computer.check_measurement(
            measurement['cx'], measurement['cy'],
            od_x, od_y, od_r
        )

        measurement['in_exclusion_zone'] = in_zone
        measurement['exclusion_reason'] = reason

   
    # Matavimo kūrimas (pilna struktūra)
   

    def _build_measurement(self, image_name: str,
                           x1: float, y1: float,
                           x2: float, y2: float,
                           green_channel: np.ndarray,
                           vessel_type: int = VESSEL_UNKNOWN,
                           measurement_id: Optional[int] = None,
                           vessel_id: Optional[int] = None
                           ) -> Optional[dict]:
        """
        Sukuria pilną matavimo struktūrą.

        Returns:
            Matavimo dict arba None jei linija per trumpa
        """
        result = self.extract_profile(
            green_channel, x1, y1, x2, y2
        )
        profile, length, spp, ext_x1, ext_y1, ext_x2, ext_y2 = result

        if profile is None:
            return None

        # OD atstumas
        od_x, od_y, od_r = self.get_od_data(image_name)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist_od = math.sqrt((cx - od_x) ** 2 + (cy - od_y) ** 2)
        zone_rod = dist_od / od_r if od_r > 0 else 0.0

        # Kampas
        dx = x2 - x1
        dy = y2 - y1
        angle_deg = math.degrees(math.atan2(dy, dx))

        m_id = measurement_id if measurement_id is not None else self._next_id(image_name)
        v_id = vessel_id if vessel_id is not None else self._next_vessel_id(image_name)

        measurement = {
            'id': m_id,
            'vessel_id': v_id,
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'cx': cx, 'cy': cy,
            # Išplėstos profilio linijos koordinatės (su padding)
            'ext_x1': ext_x1, 'ext_y1': ext_y1,
            'ext_x2': ext_x2, 'ext_y2': ext_y2,
            'width_px': 0.0,
            'width_manual': None,
            'vessel_type': vessel_type,
            'profile': profile,
            'profile_edges': None,
            'profile_edges_manual': None,
            'threshold': None,
            'samples_per_pixel': spp,
            'line_length': length,
            'angle_deg': angle_deg,
            'distance_from_od': dist_od,
            'zone_rod': zone_rod,
            'in_exclusion_zone': False,
            'exclusion_reason': "",
        }

        self._check_exclusion(measurement, image_name)
        return measurement

   
    # Edge → Image koordinačių konversija
   

    @staticmethod
    def edge_to_image_coords(measurement: dict,
                              edge_sample: float
                              ) -> Tuple[float, float]:
        """
        Konvertuoja profilio edge sample poziciją į vaizdo koordinates.

        Naudoja extended line (ext_x1..ext_y2) koordinates.

        Args:
            measurement: Matavimo dict su ext_x1..ext_y2 ir profile
            edge_sample: Edge pozicija profilio sample erdvėje

        Returns:
            (img_x, img_y) vaizdo koordinatės
        """
        profile = measurement.get('profile')
        if profile is None or len(profile) < 2:
            return measurement['cx'], measurement['cy']

        n = len(profile)
        t = edge_sample / (n - 1)

        ex1 = measurement.get('ext_x1', measurement['x1'])
        ey1 = measurement.get('ext_y1', measurement['y1'])
        ex2 = measurement.get('ext_x2', measurement['x2'])
        ey2 = measurement.get('ext_y2', measurement['y2'])

        img_x = ex1 + t * (ex2 - ex1)
        img_y = ey1 + t * (ey2 - ey1)
        return img_x, img_y

   
    # Pagrindinės operacijos
   

    def add_measurement(self, image_name: str,
                        x1: float, y1: float,
                        x2: float, y2: float,
                        green_channel: np.ndarray,
                        vessel_type: int = VESSEL_UNKNOWN,
                        vessel_id: Optional[int] = None
                        ) -> Optional[dict]:
        """
        Prideda naują matavimą.

        Args:
            image_name: Vaizdo pavadinimas
            x1, y1, x2, y2: Linijos koordinatės (vaizdo, float)
            green_channel: Žalias kanalas profiliu
            vessel_type: 0=unknown, 1=artery, 2=vein
            vessel_id: Kraujagyslės ID (None = automatinis)

        Returns:
            Matavimo dict arba None jei nepavyko
        """
        self._ensure_image(image_name)

        measurement = self._build_measurement(
            image_name, x1, y1, x2, y2, green_channel, vessel_type,
            vessel_id=vessel_id
        )

        if measurement is None:
            return None

        self.measurements[image_name].append(measurement)

        self._push_undo(_create_action(
            ACTION_ADD, image_name,
            measurement['id'],
            old_state=None,
            new_state=self._copy_measurement(measurement)
        ))

        self._auto_save()
        return measurement

    def delete_measurement(self, image_name: str,
                           measurement_id: int) -> bool:
        """
        Pašalina matavimą.

        Returns:
            True jei pavyko
        """
        m = self._find(image_name, measurement_id)
        if m is None:
            return False

        self._push_undo(_create_action(
            ACTION_DELETE, image_name,
            measurement_id,
            old_state=self._copy_measurement(m),
            new_state=None
        ))

        self.measurements[image_name].remove(m)
        self._auto_save()
        return True

    def update_type(self, image_name: str,
                    measurement_id: int,
                    new_type: int) -> bool:
        """
        Keičia kraujagyslės tipą (A/V/?).

        Returns:
            True jei pavyko
        """
        m = self._find(image_name, measurement_id)
        if m is None:
            return False

        old = self._copy_measurement(m)
        m['vessel_type'] = new_type

        self._push_undo(_create_action(
            ACTION_EDIT_TYPE, image_name,
            measurement_id,
            old_state=old,
            new_state=self._copy_measurement(m)
        ))

        self._auto_save()
        return True

    def update_vessel_id(self, image_name: str,
                         measurement_id: int,
                         new_vessel_id: int) -> bool:
        """
        Keičia vessel identitetą (grupavimas).

        Returns:
            True jei pavyko
        """
        m = self._find(image_name, measurement_id)
        if m is None:
            return False

        old = self._copy_measurement(m)
        m['vessel_id'] = new_vessel_id

        self._push_undo(_create_action(
            ACTION_EDIT_VESSEL_ID, image_name,
            measurement_id,
            old_state=old,
            new_state=self._copy_measurement(m)
        ))

        self._auto_save()
        return True

    def update_line(self, image_name: str,
                    measurement_id: int,
                    new_x1: float, new_y1: float,
                    new_x2: float, new_y2: float,
                    green_channel: np.ndarray) -> bool:
        """
        Perbrėžia matavimo liniją (Redraw Line).
        Profilis ir plotis perskaičiuojami, rankiniai kraštai resetinami.

        Returns:
            True jei pavyko
        """
        m = self._find(image_name, measurement_id)
        if m is None:
            return False

        old = self._copy_measurement(m)

        # Perskaičiuoti
        result = self.extract_profile(
            green_channel, new_x1, new_y1, new_x2, new_y2
        )
        profile, length, spp, ext_x1, ext_y1, ext_x2, ext_y2 = result
        if profile is None:
            return False

        # OD atstumas
        od_x, od_y, od_r = self.get_od_data(image_name)
        cx = (new_x1 + new_x2) / 2.0
        cy = (new_y1 + new_y2) / 2.0
        dist_od = math.sqrt((cx - od_x) ** 2 + (cy - od_y) ** 2)

        # Kampas
        dx = new_x2 - new_x1
        dy = new_y2 - new_y1
        angle_deg = math.degrees(math.atan2(dy, dx))

        # Atnaujinti
        m['x1'], m['y1'] = new_x1, new_y1
        m['x2'], m['y2'] = new_x2, new_y2
        m['cx'], m['cy'] = cx, cy
        m['ext_x1'], m['ext_y1'] = ext_x1, ext_y1
        m['ext_x2'], m['ext_y2'] = ext_x2, ext_y2
        m['profile'] = profile
        m['width_px'] = 0.0
        m['profile_edges'] = None
        m['profile_edges_manual'] = None
        m['width_manual'] = None
        m['threshold'] = None
        m['samples_per_pixel'] = spp
        m['line_length'] = length
        m['angle_deg'] = angle_deg
        m['distance_from_od'] = dist_od
        m['zone_rod'] = dist_od / od_r if od_r > 0 else 0.0

        self._check_exclusion(m, image_name)

        self._push_undo(_create_action(
            ACTION_EDIT_LINE, image_name,
            measurement_id,
            old_state=old,
            new_state=self._copy_measurement(m)
        ))

        self._auto_save()
        return True

    def update_edges(self, image_name: str,
                     measurement_id: int,
                     new_left: float,
                     new_right: float) -> bool:
        """
        Koreguoja kraštus profilyje (Adjust Edges).
        Atnaujina rankinį plotį.

        Returns:
            True jei pavyko
        """
        m = self._find(image_name, measurement_id)
        if m is None:
            return False

        old = self._copy_measurement(m)

        m['profile_edges_manual'] = (new_left, new_right)
        m['width_manual'] = abs(new_right - new_left) / m['samples_per_pixel']

        self._push_undo(_create_action(
            ACTION_EDIT_EDGES, image_name,
            measurement_id,
            old_state=old,
            new_state=self._copy_measurement(m)
        ))

        self._auto_save()
        return True

   
    # Paieška ir pagalbinės
   

    def _find(self, image_name: str, measurement_id: int) -> Optional[dict]:
        """Randa matavimą pagal ID."""
        if image_name not in self.measurements:
            return None
        for m in self.measurements[image_name]:
            if m['id'] == measurement_id:
                return m
        return None

    def get_measurements(self, image_name: str) -> List[dict]:
        """Grąžina matavimų sąrašą konkrečiam vaizdui."""
        return self.measurements.get(image_name, [])

    def get_measurement_count(self, image_name: str) -> int:
        """Matavimų skaičius vaizdui."""
        return len(self.measurements.get(image_name, []))

    def get_effective_width(self, measurement: dict) -> float:
        """
        Grąžina efektyvų plotį: rankinį jei koreguotas,
        kitaip automatinį.
        """
        if measurement.get('width_manual') is not None:
            return measurement['width_manual']
        return measurement['width_px']

    def get_vessel_ids_in_use(self, image_name: str) -> Dict[int, dict]:
        """
        Grąžina naudojamus vessel ID su info.
        {vessel_id: {'type': int, 'count': int}}
        """
        result = {}
        for m in self.get_measurements(image_name):
            vid = m['vessel_id']
            if vid not in result:
                result[vid] = {'type': m['vessel_type'], 'count': 0}
            result[vid]['count'] += 1
        return result

    def clear_all(self, image_name: str):
        """
        Pašalina visus matavimus (su undo — kiekvienas kaip atskiras delete).
        Kviečiama tik po patvirtinimo dialogo.
        """
        measurements = list(self.measurements.get(image_name, []))
        for m in reversed(measurements):
            self.delete_measurement(image_name, m['id'])

    def _sort(self, image_name: str):
        """Surikiuoja matavimus pagal ID."""
        if image_name in self.measurements:
            self.measurements[image_name].sort(key=lambda m: m['id'])

   
    # Matavimo kopijavimas (undo/redo)
   

    @staticmethod
    def _copy_measurement(m: dict) -> dict:
        """
        Giliai kopijuoja matavimo dict.
        numpy masyvai kopijuojami su .copy(), kiti su copy.deepcopy.
        """
        result = {}
        for key, val in m.items():
            if isinstance(val, np.ndarray):
                result[key] = val.copy()
            elif isinstance(val, (dict, list, tuple)):
                result[key] = copy.deepcopy(val)
            else:
                result[key] = val
        return result

    @staticmethod
    def _restore_measurement(target: dict, source: dict):
        """
        Atkuria matavimo būseną iš source į target (in-place).
        """
        for key, val in source.items():
            if isinstance(val, np.ndarray):
                target[key] = val.copy()
            elif isinstance(val, (dict, list, tuple)):
                target[key] = copy.deepcopy(val)
            else:
                target[key] = val

   
    # UNDO / REDO
   

    def _push_undo(self, action: dict):
        """Įrašo veiksmą į undo stack, išvalo redo."""
        self.undo_stack.append(action)
        self.redo_stack.clear()
        if len(self.undo_stack) > self.MAX_UNDO:
            self.undo_stack.pop(0)

    def undo(self) -> bool:
        """
        Atšaukia paskutinį veiksmą.

        Returns:
            True jei pavyko, False jei stack tuščias
        """
        if not self.undo_stack:
            return False

        action = self.undo_stack.pop()
        img = action['image_name']
        mid = action['measurement_id']

        if action['type'] == ACTION_ADD:
            # Buvo pridėtas → pašalinti
            self._remove_by_id(img, mid)

        elif action['type'] == ACTION_DELETE:
            # Buvo pašalintas → grąžinti
            self._ensure_image(img)
            self.measurements[img].append(
                self._copy_measurement(action['old_state'])
            )
            self._sort(img)

        elif action['type'] in (ACTION_EDIT_TYPE, ACTION_EDIT_VESSEL_ID,
                                ACTION_EDIT_LINE, ACTION_EDIT_EDGES):
            # Buvo redaguotas → grąžinti seną būseną
            m = self._find(img, mid)
            if m is not None:
                self._restore_measurement(m, action['old_state'])

        self.redo_stack.append(action)
        self._auto_save()
        return True

    def redo(self) -> bool:
        """
        Pakartoja atšauktą veiksmą.

        Returns:
            True jei pavyko, False jei stack tuščias
        """
        if not self.redo_stack:
            return False

        action = self.redo_stack.pop()
        img = action['image_name']
        mid = action['measurement_id']

        if action['type'] == ACTION_ADD:
            # Buvo pašalintas (undo add) → pridėti atgal
            self._ensure_image(img)
            self.measurements[img].append(
                self._copy_measurement(action['new_state'])
            )
            self._sort(img)

        elif action['type'] == ACTION_DELETE:
            # Buvo grąžintas (undo delete) → pašalinti vėl
            self._remove_by_id(img, mid)

        elif action['type'] in (ACTION_EDIT_TYPE, ACTION_EDIT_VESSEL_ID,
                                ACTION_EDIT_LINE, ACTION_EDIT_EDGES):
            # Buvo atstatytas (undo edit) → taikyti new_state
            m = self._find(img, mid)
            if m is not None:
                self._restore_measurement(m, action['new_state'])

        self.undo_stack.append(action)
        self._auto_save()
        return True

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def get_undo_description(self) -> str:
        """Undo veiksmo aprašymas (Edit meniu, status bar)."""
        if not self.undo_stack:
            return ""
        return _UNDO_DESCRIPTIONS.get(self.undo_stack[-1]['type'], "Undo")

    def get_redo_description(self) -> str:
        """Redo veiksmo aprašymas."""
        if not self.redo_stack:
            return ""
        return _REDO_DESCRIPTIONS.get(self.redo_stack[-1]['type'], "Redo")

    def _remove_by_id(self, image_name: str, measurement_id: int):
        """Pašalina matavimą pagal ID (vidinis)."""
        if image_name not in self.measurements:
            return
        self.measurements[image_name] = [
            m for m in self.measurements[image_name]
            if m['id'] != measurement_id
        ]

   
    # Auto-save
   

    def _auto_save(self):
        """Išsaugo dabartinę būseną į autosave JSON."""
        try:
            os.makedirs(self.AUTOSAVE_DIR, exist_ok=True)
            path = os.path.join(self.AUTOSAVE_DIR, self.AUTOSAVE_FILE)
            data = self._serialize()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except (OSError, TypeError) as e:
            # Auto-save neturėtų sugriauti programos
            print(f"[MeasurementManager] Auto-save klaida: {e}")

    def _serialize(self) -> dict:
        """Konvertuoja matavimus į JSON-serializuojamą dict."""
        data = {
            'version': '1.1',
            'timestamp': time.time(),
            'image_paths': dict(self._image_paths),
            'optic_disc': {},
            'measurements': {},
            'exclusion_masks': {},
        }

        # OD duomenys
        for img_name, (ox, oy, r) in self._od_data.items():
            data['optic_disc'][img_name] = {'x': ox, 'y': oy, 'r': r}

        # Matavimai
        for img_name, meas_list in self.measurements.items():
            data['measurements'][img_name] = []
            for m in meas_list:
                entry = {}
                for key, val in m.items():
                    if key == 'profile':
                        # numpy → list
                        entry[key] = val.tolist() if val is not None else None
                    elif isinstance(val, (np.integer, np.int64, np.int32)):
                        entry[key] = int(val)
                    elif isinstance(val, (np.floating, np.float64, np.float32)):
                        entry[key] = float(val)
                    else:
                        entry[key] = val
                data['measurements'][img_name].append(entry)

        # Exclusion kaukės (per image, RLE formatu)
        if self._exclusion_masks:
            data['exclusion_masks'] = dict(self._exclusion_masks)

        return data

   
    # Session load
   

    def load_session(self, path: str) -> bool:
        """
        Atkuria sesiją iš JSON failo.

        Args:
            path: Kelias į JSON failą

        Returns:
            True jei pavyko
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[MeasurementManager] Session load klaida: {e}")
            return False

        return self._deserialize(data)

    def load_autosave(self) -> bool:
        """Bando atkurti sesiją iš autosave."""
        path = os.path.join(self.AUTOSAVE_DIR, self.AUTOSAVE_FILE)
        if not os.path.exists(path):
            return False
        return self.load_session(path)

    def has_autosave(self) -> bool:
        """Ar egzistuoja autosave failas."""
        path = os.path.join(self.AUTOSAVE_DIR, self.AUTOSAVE_FILE)
        return os.path.exists(path)

    def _deserialize(self, data: dict) -> bool:
        """Atkuria matavimus iš JSON dict."""
        try:
            # Išvalyti esamus duomenis
            self.measurements.clear()
            self._next_ids.clear()
            self._image_paths.clear()
            self._od_data.clear()
            self._exclusion_masks.clear()
            self.undo_stack.clear()
            self.redo_stack.clear()

            # Vaizdo keliai
            self._image_paths.update(data.get('image_paths', {}))

            # Exclusion kaukės (RLE)
            self._exclusion_masks.update(data.get('exclusion_masks', {}))

            # OD duomenys
            for img_name, od in data.get('optic_disc', {}).items():
                self._od_data[img_name] = (od['x'], od['y'], od['r'])

            # Matavimai
            for img_name, meas_list in data.get('measurements', {}).items():
                self._ensure_image(img_name)
                for entry in meas_list:
                    # Atkurti numpy
                    if entry.get('profile') is not None:
                        entry['profile'] = np.array(entry['profile'],
                                                    dtype=np.float64)
                    # Atkurti tuple
                    if entry.get('profile_edges') is not None:
                        entry['profile_edges'] = tuple(entry['profile_edges'])
                    if entry.get('profile_edges_manual') is not None:
                        entry['profile_edges_manual'] = tuple(
                            entry['profile_edges_manual']
                        )

                    self.measurements[img_name].append(entry)

                    # Atnaujinti ID skaitliukus
                    if entry['id'] >= self._next_ids[img_name]:
                        self._next_ids[img_name] = entry['id'] + 1

            return True
        except (KeyError, TypeError, ValueError) as e:
            print(f"[MeasurementManager] Deserialize klaida: {e}")
            return False

    def save_session(self, path: str) -> bool:
        """
        Išsaugo sesiją į nurodytą JSON failą.

        Args:
            path: Kelias į JSON failą

        Returns:
            True jei pavyko
        """
        try:
            data = self._serialize()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except (OSError, TypeError) as e:
            print(f"[MeasurementManager] Session save klaida: {e}")
            return False