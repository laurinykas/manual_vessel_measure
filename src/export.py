"""
export.py - Matavimų eksportas

Formatai:
    1. VICAVR formatas — suderinamas su VICAVR rinkinio struktūra
    2. Detalus CSV — visi matavimo laukai tyrimui
    3. Session JSON — pilna sesijos būsena (auto-save / Load Session)

VICAVR formatas:
    #Image no,vessel no,coord x,coord y,caliber,vessel type
    1,0,409,337,8.88897,2

Detalus CSV stulpeliai:
    image, vessel_id, measurement_no, cx, cy, x1, y1, x2, y2,
    width, type, line_length, angle_deg,
    distance_from_od, zone_rod,
"""

import csv
import json
import math
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np



# VICAVR formatas


def export_vicavr_format(measurements: Dict[str, List[dict]],
                         output_path: str,
                         image_number_map: Optional[Dict[str, int]] = None
                         ) -> bool:
    """
    Eksportuoja matavimus VICAVR formatu.

    Formatas:
        #Image no,vessel no,coord x,coord y,caliber,vessel type
        1,0,409,337,8.88897,2

    caliber = width_manual jei rankiniu būdu pakoreguotas,
              kitaip width_px

    vessel type: 0=unclassified, 1=artery, 2=vein

    Args:
        measurements: {image_name: [measurement_dict, ...]}
        output_path: Kelias į išvesties failą (.txt arba .csv)
        image_number_map: {image_name: int} — vaizdo numeriai.
            Jei None, numeruojama automatiškai nuo 1.

    Returns:
        True jei pavyko
    """
    try:
        # Automatinis numeravimas jei nenurodyta
        if image_number_map is None:
            image_number_map = {}
            for idx, name in enumerate(sorted(measurements.keys()), start=1):
                image_number_map[name] = idx

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # Header
            f.write("#Image no,vessel no,coord x,coord y,caliber,vessel type\n")

            for image_name in sorted(measurements.keys()):
                img_no = image_number_map.get(image_name, 0)
                meas_list = measurements[image_name]

                for m in meas_list:
                    vessel_no = m.get('vessel_id', 0)
                    cx = m.get('cx', 0.0)
                    cy = m.get('cy', 0.0)

                    # Efektyvus plotis: rankinis jei koreguotas
                    if m.get('width_manual') is not None:
                        caliber = m['width_manual']
                    else:
                        caliber = m.get('width_px', 0.0)

                    vessel_type = m.get('vessel_type', 0)

                    f.write(
                        f"{img_no},{vessel_no},"
                        f"{cx:.0f},{cy:.0f},"
                        f"{caliber:.5f},{vessel_type}\n"
                    )

        return True

    except OSError as e:
        print(f"[export] VICAVR eksporto klaida: {e}")
        return False



# Detalus CSV


# CSV stulpelių tvarka
_CSV_COLUMNS = [
    'image',
    'vessel_id',
    'measurement_no',
    'cx', 'cy',
    'x1', 'y1', 'x2', 'y2',
    'width',
    'type',
    'line_length',
    'angle_deg',
    'distance_from_od',
    'zone_rod',
    'in_exclusion_zone',
    'exclusion_reason',
]

# Vessel type žymėjimai CSV eksportui
_TYPE_LABELS = {0: 'U', 1: 'A', 2: 'V'}


def export_detailed_csv(measurements: Dict[str, List[dict]],
                        output_path: str) -> bool:
    """
    Eksportuoja matavimus į detalų CSV su visais laukais.

    Skirtas tyrimo duomenų analizei — visi matavimo parametrai,
    exclusion zone informacija, rankiniai koregavimai.

    Args:
        measurements: {image_name: [measurement_dict, ...]}
        output_path: Kelias į .csv failą

    Returns:
        True jei pavyko
    """
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_COLUMNS)

            for image_name in sorted(measurements.keys()):
                for m in measurements[image_name]:
                    # Tikrasis plotis: rankinis jei koreguotas, kitu atveju auto
                    w_auto = m.get('width_px', 0.0)
                    w_manual = m.get('width_manual')
                    w_final = w_manual if w_manual is not None else w_auto

                    row = [
                        image_name,
                        m.get('vessel_id', 0),
                        m.get('id', 0),
                        f"{m.get('cx', 0.0):.6f}",
                        f"{m.get('cy', 0.0):.6f}",
                        f"{m.get('x1', 0.0):.6f}",
                        f"{m.get('y1', 0.0):.6f}",
                        f"{m.get('x2', 0.0):.6f}",
                        f"{m.get('y2', 0.0):.6f}",
                        f"{w_final:.6f}",
                        _TYPE_LABELS.get(m.get('vessel_type', 0), 'U'),
                        f"{m.get('line_length', 0.0):.6f}",
                        f"{m.get('angle_deg', 0.0):.6f}",
                        f"{m.get('distance_from_od', 0.0):.6f}",
                        f"{m.get('zone_rod', 0.0):.6f}",
                        m.get('in_exclusion_zone', False),
                        m.get('exclusion_reason', ''),
                    ]
                    writer.writerow(row)

        return True

    except OSError as e:
        print(f"[export] CSV eksporto klaida: {e}")
        return False




# Pagalbinės konversijos


def _measurement_to_json(m: dict) -> dict:
    """
    Konvertuoja vieną matavimo dict į JSON-serializuojamą formatą.
    numpy masyvai → list, numpy int/float → Python int/float.
    """
    entry = {}
    for key, val in m.items():
        if key == 'profile':
            entry[key] = val.tolist() if isinstance(val, np.ndarray) else val
        elif isinstance(val, (np.integer,)):
            entry[key] = int(val)
        elif isinstance(val, (np.floating,)):
            entry[key] = float(val)
        elif isinstance(val, tuple):
            entry[key] = list(val)
        else:
            entry[key] = val
    return entry




# Eksperto matavimų importas


_TYPE_FROM_LABEL = {'U': 0, 'A': 1, 'V': 2, '0': 0, '1': 1, '2': 2}


def import_reference_csv(path: str) -> Dict[str, List[dict]]:
    """
    Importuoja eksperto matavimus iš detailed CSV.

    Grąžina {image_name: [ref_dict, ...]} kur ref_dict turi:
    x1, y1, x2, y2, width_px, vessel_type, vessel_id

    Args:
        path: Kelias į CSV failą

    Returns:
        {image_name: [ref_measurement, ...]}
    """
    result: Dict[str, List[dict]] = {}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row.get('image', '').strip()
                if not img:
                    continue

                try:
                    ref = {
                        'x1': float(row.get('x1', 0)),
                        'y1': float(row.get('y1', 0)),
                        'x2': float(row.get('x2', 0)),
                        'y2': float(row.get('y2', 0)),
                        'width_px': float(row.get('width_auto', 0) or 0),
                        'width_manual': (
                            float(row['width_manual'])
                            if row.get('width_manual') else None
                        ),
                        'vessel_type': _TYPE_FROM_LABEL.get(
                            row.get('type', 'U'), 0
                        ),
                        'vessel_id': int(
                            row.get('vessel_id', 0) or 0
                        ),
                        'id': int(
                            row.get('measurement_no',
                                    row.get('vessel_no', 0)) or 0
                        ),
                    }
                except (ValueError, KeyError):
                    continue

                if img not in result:
                    result[img] = []
                result[img].append(ref)

    except (OSError, csv.Error) as e:
        print(f"[Import] CSV import error: {e}")

    return result


def import_reference_session(path: str) -> Dict[str, List[dict]]:
    """
    Importuoja eksperto matavimus iš session JSON.

    Grąžina {image_name: [ref_dict, ...]} su pilnomis koordinatėmis.

    Args:
        path: Kelias į session JSON failą

    Returns:
        {image_name: [ref_measurement, ...]}
    """
    result: Dict[str, List[dict]] = {}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for img_name, meas_list in data.get('measurements', {}).items():
            refs = []
            for m in meas_list:
                ref = {
                    'x1': float(m.get('x1', 0)),
                    'y1': float(m.get('y1', 0)),
                    'x2': float(m.get('x2', 0)),
                    'y2': float(m.get('y2', 0)),
                    'ext_x1': float(m.get('ext_x1', m.get('x1', 0))),
                    'ext_y1': float(m.get('ext_y1', m.get('y1', 0))),
                    'ext_x2': float(m.get('ext_x2', m.get('x2', 0))),
                    'ext_y2': float(m.get('ext_y2', m.get('y2', 0))),
                    'width_px': float(m.get('width_px', 0)),
                    'width_manual': m.get('width_manual'),
                    'vessel_type': int(m.get('vessel_type', 0)),
                    'vessel_id': int(m.get('vessel_id', 0)),
                    'id': int(m.get('id', 0)),
                    'profile': (
                        np.array(m['profile'], dtype=np.float64)
                        if m.get('profile') else None
                    ),
                    'profile_edges': (
                        tuple(m['profile_edges'])
                        if m.get('profile_edges') else None
                    ),
                    'profile_edges_manual': (
                        tuple(m['profile_edges_manual'])
                        if m.get('profile_edges_manual') else None
                    ),
                    'samples_per_pixel': float(
                        m.get('samples_per_pixel', 3.0)
                    ),
                }
                refs.append(ref)

            if refs:
                result[img_name] = refs

    except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[Import] Session import error: {e}")

    return result



# Anotuoto vaizdo generavimas


# Vessel type spalvos (BGR — OpenCV tvarka)
_TYPE_COLORS_BGR = {
    0: (180, 180, 180),  # Unknown — pilka
    1: (80, 80, 255),    # Artery — raudona
    2: (255, 100, 60),   # Vein — mėlyna
}

# Reference spalvos (BGR)
_REF_COLORS_BGR = {
    0: (60, 200, 255),   # Unknown — geltona
    1: (255, 220, 60),   # Artery — žydra
    2: (60, 140, 255),   # Vein — oranžinė
}


def render_annotated_image(
    image_bgr: np.ndarray,
    measurements: List[dict],
    exclusion_mask: Optional[np.ndarray] = None,
    od_x: int = 0, od_y: int = 0, od_r: int = 0,
    reference_measurements: Optional[List[dict]] = None,
    od_multipliers: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Sukuria anotuotą vaizdą su matavimais, exclusion zonomis ir OD.

    Returns:
        Anotuotas BGR vaizdas
    """
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    # --- 1. Exclusion overlay (tiesioginė alpha be addWeighted) ---
    if exclusion_mask is not None:
        mask_bool = exclusion_mask > 0
        if np.any(mask_bool):
            alpha = 0.3
            tint = np.array([60, 60, 255], dtype=np.float32)  # BGR
            pixels = canvas[mask_bool].astype(np.float32)
            blended = pixels * (1.0 - alpha) + tint * alpha
            canvas[mask_bool] = np.clip(blended, 0, 255).astype(np.uint8)

    # --- 2. OD apskritimai ---
    if od_r > 0:
        mults = od_multipliers or [0.5, 1.0, 2.0, 3.0]
        for mult in mults:
            r = int(od_r * mult)
            cv2.circle(canvas, (od_x, od_y), r,
                       (180, 180, 180), 1, cv2.LINE_AA)

    # --- 3. Reference matavimai (dashed) ---
    if reference_measurements:
        for m in reference_measurements:
            color = _REF_COLORS_BGR.get(m.get('vessel_type', 0),
                                         _REF_COLORS_BGR[0])
            x1 = m.get('ext_x1', m.get('x1', 0))
            y1 = m.get('ext_y1', m.get('y1', 0))
            x2 = m.get('ext_x2', m.get('x2', 0))
            y2 = m.get('ext_y2', m.get('y2', 0))

            _draw_dashed_line(canvas, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 1)

            wp = m.get('width_manual') or m.get('width_px', 0)
            mx = int((m.get('x1', 0) + m.get('x2', 0)) / 2)
            my = int((m.get('y1', 0) + m.get('y2', 0)) / 2) - 5
            cv2.putText(canvas, f"R:{wp:.1f}", (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                        cv2.LINE_AA)

    # --- 4. Pagrindiniai matavimai ---
    for m in measurements:
        vtype = m.get('vessel_type', 0)
        color = _TYPE_COLORS_BGR.get(vtype, _TYPE_COLORS_BGR[0])

        ex1 = int(m.get('ext_x1', m.get('x1', 0)))
        ey1 = int(m.get('ext_y1', m.get('y1', 0)))
        ex2 = int(m.get('ext_x2', m.get('x2', 0)))
        ey2 = int(m.get('ext_y2', m.get('y2', 0)))

        cv2.line(canvas, (ex1, ey1), (ex2, ey2), color, 2, cv2.LINE_AA)

        # Edge tick-marks
        edges = m.get('profile_edges_manual') or m.get('profile_edges')
        if edges is not None:
            dx = ex2 - ex1
            dy = ey2 - ey1
            elen = math.sqrt(dx * dx + dy * dy)
            if elen > 1e-6:
                perp_x = -dy / elen
                perp_y = dx / elen
                tick = max(4, int(min(h, w) * 0.004))

                for edge_s in edges:
                    coords = _edge_to_image_coords(m, edge_s)
                    if coords is None:
                        continue
                    ix, iy = coords
                    t1 = (int(ix + perp_x * tick), int(iy + perp_y * tick))
                    t2 = (int(ix - perp_x * tick), int(iy - perp_y * tick))
                    cv2.line(canvas, t1, t2, color, 2, cv2.LINE_AA)

        wp = m.get('width_manual') or m.get('width_px', 0)
        type_char = {0: 'U', 1: 'A', 2: 'V'}.get(vtype, '?')
        label = f"{type_char}:{wp:.1f}"
        mx = int((m.get('x1', 0) + m.get('x2', 0)) / 2)
        my = int((m.get('y1', 0) + m.get('y2', 0)) / 2) - 6
        cv2.putText(canvas, label, (mx, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    cv2.LINE_AA)

    return canvas


def _edge_to_image_coords(m: dict, edge_sample: float
                           ) -> Optional[Tuple[float, float]]:
    """Edge sample → vaizdo koordinatės (inlined, be circular import)."""
    profile = m.get('profile')
    if profile is None:
        return None
    n = len(profile) if hasattr(profile, '__len__') else 0
    if n < 2:
        return None

    t = edge_sample / (n - 1)
    ex1 = m.get('ext_x1', m.get('x1', 0))
    ey1 = m.get('ext_y1', m.get('y1', 0))
    ex2 = m.get('ext_x2', m.get('x2', 0))
    ey2 = m.get('ext_y2', m.get('y2', 0))
    return ex1 + t * (ex2 - ex1), ey1 + t * (ey2 - ey1)


def _draw_dashed_line(img: np.ndarray, pt1: tuple, pt2: tuple,
                      color: tuple, thickness: int = 1,
                      dash_len: int = 8, gap_len: int = 6):
    """Piešia brūkšninę liniją ant vaizdo."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return

    ux = dx / length
    uy = dy / length
    pos = 0.0
    drawing = True

    while pos < length:
        seg = dash_len if drawing else gap_len
        end = min(pos + seg, length)
        if drawing:
            p1 = (int(pt1[0] + ux * pos), int(pt1[1] + uy * pos))
            p2 = (int(pt1[0] + ux * end), int(pt1[1] + uy * end))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
        pos = end
        drawing = not drawing



# Saugus vaizdo rašymas (apsauga nuo stack overflow su dideliais vaizdais)


def _safe_imwrite(path: str, image: np.ndarray) -> bool:
    """
    Saugus cv2.imwrite — jei PNG crashina (didelis vaizdas),
    bando JPEG arba TIFF fallback.

    Returns:
        True jei pavyko
    """
    try:
        ok = cv2.imwrite(path, image)
        if ok:
            return True
    except Exception as e:
        print(f"[Export] imwrite failed for {path}: {e}")

    # Fallback: pakeisti plėtinį į .jpg (spalvoti) arba .tif (kaukės)
    base, ext = os.path.splitext(path)
    if len(image.shape) == 2:
        # Grayscale kaukė — TIFF
        fallback = base + '.tif'
    else:
        # Spalvotas — JPEG
        fallback = base + '.jpg'

    try:
        params = []
        if fallback.endswith('.jpg'):
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        ok = cv2.imwrite(fallback, image, params)
        if ok:
            print(f"[Export] Fallback saved: {fallback}")
            return True
    except Exception as e2:
        print(f"[Export] Fallback also failed: {e2}")

    return False



# Unified eksportas — vienas folderis su viskuo


def export_image_results(
    output_dir: str,
    image_name: str,
    image_bgr: np.ndarray,
    measurements: List[dict],
    od_data: Tuple[int, int, int] = (0, 0, 0),
    exclusion_mask: Optional[np.ndarray] = None,
    original_path: Optional[str] = None,
    reference_measurements: Optional[List[dict]] = None,
    od_multipliers: Optional[List[float]] = None,
    all_measurements: Optional[Dict[str, List[dict]]] = None,
    all_od_data: Optional[dict] = None,
    exclusion_masks_rle: Optional[dict] = None,
    image_paths: Optional[dict] = None,
) -> str:
    """
    Eksportuoja vieno vaizdo rezultatus į folderį.

    Sukuria:
        {output_dir}/{base_name}/
            {base_name}_original.{ext}
            {base_name}_annotated.png
            {base_name}_exclusion.png      (jei yra)
            {base_name}_measurements.csv
            {base_name}_session.json

    Returns:
        Sukurto folderio kelias
    """
    base = os.path.splitext(image_name)[0]
    folder = os.path.join(output_dir, base)
    os.makedirs(folder, exist_ok=True)

    ext = os.path.splitext(image_name)[1] or '.tif'

    # 1. Originali nuotrauka (kopija)
    if original_path and os.path.exists(original_path):
        dst_orig = os.path.join(folder, f"{base}_original{ext}")
        try:
            if os.path.abspath(original_path) != os.path.abspath(dst_orig):
                shutil.copy2(original_path, dst_orig)
        except OSError as e:
            print(f"[Export] Copy original failed: {e}")
    else:
        _safe_imwrite(
            os.path.join(folder, f"{base}_original.png"), image_bgr
        )

    # 2. Anotuotas vaizdas
    annotated = render_annotated_image(
        image_bgr, measurements,
        exclusion_mask=exclusion_mask,
        od_x=od_data[0], od_y=od_data[1], od_r=od_data[2],
        reference_measurements=reference_measurements,
        od_multipliers=od_multipliers,
    )
    _safe_imwrite(
        os.path.join(folder, f"{base}_annotated.png"), annotated
    )

    # 3. Exclusion kaukė
    if exclusion_mask is not None and np.any(exclusion_mask > 0):
        _safe_imwrite(
            os.path.join(folder, f"{base}_exclusion.png"), exclusion_mask
        )

    # 4. Matavimai CSV
    if measurements:
        dst_csv = os.path.join(folder, f"{base}_measurements.csv")
        export_detailed_csv({image_name: measurements}, dst_csv)

    # 5. Session JSON (atkūrimui)
    session_data = {
        'version': '1.1',
        'timestamp': time.time(),
        'image_paths': image_paths or {},
        'optic_disc': {},
        'measurements': {},
        'exclusion_masks': exclusion_masks_rle or {},
    }

    if all_od_data:
        for img, (ox, oy, r) in all_od_data.items():
            session_data['optic_disc'][img] = {'x': ox, 'y': oy, 'r': r}
    elif od_data[2] > 0:
        session_data['optic_disc'][image_name] = {
            'x': od_data[0], 'y': od_data[1], 'r': od_data[2]
        }

    m_data = all_measurements or {image_name: measurements}
    for img, mlist in m_data.items():
        session_data['measurements'][img] = []
        for m in mlist:
            session_data['measurements'][img].append(
                _measurement_to_json(m)
            )

    dst_json = os.path.join(folder, f"{base}_session.json")
    try:
        with open(dst_json, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[Export] Session JSON error: {e}")

    return folder