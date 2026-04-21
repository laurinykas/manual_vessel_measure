"""
masking.py - Akies dugno kaukės kūrimas

- norming() -> norming()
- normingMask() -> normingMask()
- createMask() -> createMask()
- propCoef() -> propCoef()
"""

import cv2
import numpy as np
from typing import Tuple


def norming(im: np.ndarray) -> np.ndarray:
    """
    Normalizuoja paveikslėlį į 0-255 intervalą.

        cv::minMaxLoc(im, &min_val, &max_val);
        im -= min_val;
        cv::minMaxLoc(im, &min_val, &max_val);
        im = (im / max_val * 255);

    Args:
        im: Įvesties paveikslėlis (modifikuojamas vietoje)

    Returns:
        Normalizuotas paveikslėlis
    """
    # Konvertuoti į float kad išvengti perpildymo
    im_float = im.astype(np.float64)

    min_val = im_float.min()
    max_val = im_float.max()

    if max_val == min_val:
        return np.zeros_like(im, dtype=np.uint8)

    im_float = im_float - min_val
    max_val = im_float.max()

    if max_val > 0:
        im_float = (im_float / max_val * 255)

    return im_float.astype(np.uint8)


def normingMask(im: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Normalizuoja paveikslėlį tik kaukės srityje.

    C++ originale:
        cv::Mat im2 = (im | mask_inv);
        cv::minMaxLoc(im2, &min_val, &max_val);  // min iš kaukės srities
        im -= min_val;                             // uint8 saturuoja: <0 -> 0
        cv::minMaxLoc(im, &min_val, &max_val);
        im = (im / max_val * 255);

    Args:
        im: Įvesties paveikslėlis
        mask: Binari kaukė (255 = validus regionas)

    Returns:
        Normalizuotas paveikslėlis
    """
    # Sukurti inversą kaukę
    mask_inv = cv2.bitwise_not(mask)

    # Užpildyti ne-kaukės regioną maksimalia reikšme
    # kad minMaxLoc rastų tik kaukės srities min
    im2 = cv2.bitwise_or(im, mask_inv)

    # Rasti min kaukės srityje
    min_val = float(im2.min())

    # Konvertuoti į float
    im_float = im.astype(np.float64)

    # Atimti minimum
    im_float = im_float - min_val

    #uint8 saturation: neigiamos reikšmės -> 0
    im_float = np.clip(im_float, 0, None)

    # Rasti naują maksimumą
    max_val = im_float.max()

    # Normalizuoti į 0-255
    if max_val > 0:
        im_float = (im_float / max_val * 255)

    return im_float.astype(np.uint8)


def createMask(img_green: np.ndarray) -> np.ndarray:
    """
    Sukuria akies dugno kaukę.

    Algoritmas:
    1. Normalizuoti žalią kanalą
    2. Taikyti slenkstį (threshold = 30)
    3. Morfologinis uždarymas (close)
    4. Flood fill iš kampų pašalinti foną
    5. Erozija pašalinti krašto artefaktus

    Args:
        img_green: Žalias kanalas arba pilkas paveikslėlis

    Returns:
        Binari kaukė (255 = akies dugno regionas)
    """
    # Apskaičiuoti proporciją pagal plotį
    prop = round(img_green.shape[1] / 500.0)
    prop = max(1, prop)  # Minimaliai 1

    # Normalizuoti
    img_norm = norming(img_green.copy())

    # Taikyti slenkstį
    _, img_mask = cv2.threshold(img_norm, 30, 255, cv2.THRESH_BINARY)

    # Morfologinis uždarymas
    Bcl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, Bcl, iterations=5)

    # Paruošti flood fill
    im_floodfill = img_mask.copy()

    rect_size = int(round(50 * prop))

    lastr = img_green.shape[0] - 1
    lastc = img_green.shape[1] - 1

    # Nupiešti juodus stačiakampius kampuose
    # Viršutinis kairys
    cv2.rectangle(im_floodfill, (0, 0), (rect_size, rect_size), 0, cv2.FILLED)
    # Viršutinis dešinys
    cv2.rectangle(im_floodfill, (lastc, 0), (lastc - rect_size, rect_size), 0, cv2.FILLED)
    # Apatinis kairys
    cv2.rectangle(im_floodfill, (0, lastr), (rect_size, lastr - rect_size), 0, cv2.FILLED)
    # Apatinis dešinys
    cv2.rectangle(im_floodfill, (lastc, lastr), (lastc - rect_size, lastr - rect_size), 0, cv2.FILLED)

    # Flood fill iš kiekvieno kampo
    h, w = im_floodfill.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, flood_mask, (0, 0), 255)
    cv2.floodFill(im_floodfill, flood_mask, (lastc, 0), 255)
    cv2.floodFill(im_floodfill, flood_mask, (0, lastr), 255)
    cv2.floodFill(im_floodfill, flood_mask, (lastc, lastr), 255)

    # Invertuoti flood fill rezultatą
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Sujungti su originalia kauke
    img_mask = cv2.bitwise_or(img_mask, im_floodfill_inv)

    # Erozija pašalinti krašto artefaktus
    mask_border = int(round(15 * prop))
    if mask_border < 3:
        mask_border = 3
    # Užtikrinti nelyginį dydį (reikalinga cv2.getStructuringElement)
    if mask_border % 2 == 0:
        mask_border += 1

    Bbrd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_border, mask_border))
    img_mask = cv2.erode(img_mask, Bbrd)

    return img_mask


def propCoef(img_mask: np.ndarray) -> float:
    """
    Apskaičiuoja proporcingumo koeficientą pagal kaukės plotį.

    Algoritmas skaičiuoja kaukės plotį (nuo pirmo iki paskutinio
    balto pikselio horizontaliai) ir grąžina santykį su 500.

    Args:
        img_mask: Binari kaukė

    Returns:
        Proporcingumo koeficientas (kaukės_plotis / 500)
    """
    min_m = img_mask.shape[1]  # Pradinis minimum = plotis
    max_m = 1

    # Iteruoti per eilutes
    for i in range(1, img_mask.shape[0]):
        lp = 0  # Ankstesnis pikselis

        for j in range(1, img_mask.shape[1]):
            cp = img_mask[i, j]  # Dabartinis pikselis

            # Rasti kairįjį kraštą (perėjimas iš 0 į 255)
            if lp == 0 and cp == 255 and min_m > j:
                min_m = j
            # Rasti dešinįjį kraštą (perėjimas iš 255 į 0)
            elif lp == 255 and cp == 0 and max_m < j:
                max_m = j

            lp = cp

    # Apskaičiuoti plotį
    wd = max_m - min_m

    if wd <= 0:
        return 1.0

    # Proporcingumo koeficientas
    prop = float(wd) / 500.0

    return prop


def propCoef_fast(img_mask: np.ndarray) -> float:
    """
    Greita proporcingumo koeficiento versija naudojant numpy.

    Args:
        img_mask: Binari kaukė

    Returns:
        Proporcingumo koeficientas
    """
    # Rasti visas eilutes su bent vienu baltu pikseliu
    row_has_content = np.any(img_mask > 0, axis=1)

    if not np.any(row_has_content):
        return 1.0

    # Rasti kairįjį ir dešinįjį kraštus
    left_edges = []
    right_edges = []

    for row_idx in np.where(row_has_content)[0]:
        row = img_mask[row_idx]
        nonzero = np.where(row > 0)[0]

        if len(nonzero) > 0:
            left_edges.append(nonzero[0])
            right_edges.append(nonzero[-1])

    if not left_edges:
        return 1.0

    # Plotis
    min_m = min(left_edges)
    max_m = max(right_edges)
    wd = max_m - min_m

    if wd <= 0:
        return 1.0

    return float(wd) / 500.0



# PAGRINDINĖ FUNKCIJA

def create_fundus_mask(img_green: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Sukuria akies dugno kaukę ir apskaičiuoja skalės koeficientą.

    Args:
        img_green: Žalias kanalas arba pilkas paveikslėlis

    Returns:
        tuple: (img_mask, sc)
            - img_mask: Binari kaukė
            - sc: Skalės koeficientas
    """
    # Sukurti kaukę
    img_mask = createMask(img_green)

    # Apskaičiuoti skalės koeficientą
    sc = propCoef(img_mask)

    # Užtikrinti minimalų koeficientą
    sc = max(0.5, sc)

    return img_mask, sc