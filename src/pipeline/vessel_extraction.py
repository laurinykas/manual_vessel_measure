"""
vessel_extraction.py - Kraujagyslių išskyrimas (OPTIMIZUOTA VERSIJA)

"""

import cv2
import numpy as np
from typing import Tuple


def thinning(im: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen ploninimo algoritmas.

    Args:
        im: Binarinis paveikslėlis (0-255)

    Returns:
        Suplonintas paveikslėlis (0-255)
    """
    # 1 prioritetas: OpenCV ximgproc
    try:
        return cv2.ximgproc.thinning(
            im, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
    except (cv2.error, AttributeError):
        pass

    # 2 prioritetas: vektorizuotas Python Zhang-Suen
    return _thinning_python(im)


def _thinning_python(im: np.ndarray) -> np.ndarray:
    """
    Optimizuota Python Zhang-Suen implementacija.
    Naudoja numpy operacijas vietoj ciklų.
    """
    # Konvertuoti į 0-1
    im = (im // 255).astype(np.uint8)

    prev = np.zeros(im.shape, dtype=np.uint8)
    max_iterations = 1000
    iteration = 0

    while iteration < max_iterations:
        im = _thinning_iteration_vectorized(im, 0)
        im = _thinning_iteration_vectorized(im, 1)

        diff = cv2.absdiff(im, prev)
        if cv2.countNonZero(diff) == 0:
            break

        prev = im.copy()
        iteration += 1

    return im * 255


def _thinning_iteration_vectorized(im: np.ndarray, iter_type: int) -> np.ndarray:
    """
    Vektorizuota Zhang-Suen iteracija naudojant numpy.
    """
    # Gauti kaimynus naudojant slicing (daug greičiau nei ciklai)
    p2 = np.pad(im, 1)[:-2, 1:-1]   # viršuje
    p3 = np.pad(im, 1)[:-2, 2:]     # viršuje dešinėje
    p4 = np.pad(im, 1)[1:-1, 2:]    # dešinėje
    p5 = np.pad(im, 1)[2:, 2:]      # apačioje dešinėje
    p6 = np.pad(im, 1)[2:, 1:-1]    # apačioje
    p7 = np.pad(im, 1)[2:, :-2]     # apačioje kairėje
    p8 = np.pad(im, 1)[1:-1, :-2]   # kairėje
    p9 = np.pad(im, 1)[:-2, :-2]    # viršuje kairėje

    # A - perėjimų skaičius
    A = ((p2 == 0) & (p3 == 1)).astype(np.int32) + \
        ((p3 == 0) & (p4 == 1)).astype(np.int32) + \
        ((p4 == 0) & (p5 == 1)).astype(np.int32) + \
        ((p5 == 0) & (p6 == 1)).astype(np.int32) + \
        ((p6 == 0) & (p7 == 1)).astype(np.int32) + \
        ((p7 == 0) & (p8 == 1)).astype(np.int32) + \
        ((p8 == 0) & (p9 == 1)).astype(np.int32) + \
        ((p9 == 0) & (p2 == 1)).astype(np.int32)

    # B - kaimynų suma
    B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

    # m1, m2 sąlygos
    if iter_type == 0:
        m1 = p2 * p4 * p6
        m2 = p4 * p6 * p8
    else:
        m1 = p2 * p4 * p8
        m2 = p2 * p6 * p8

    # Zhang-Suen sąlyga
    condition = (A == 1) & (B >= 2) & (B <= 6) & (m1 == 0) & (m2 == 0)

    # Pašalinti pažymėtus pikselius
    result = im.copy()
    result[condition] = 0

    return result


def cleareIsolated(im: np.ndarray) -> np.ndarray:
    """
    Pašalina izoliuotus pikselius.

    Args:
        im: Binarinis paveikslėlis (0-255)

    Returns:
        Išvalytas paveikslėlis (0-255)
    """
    # Konvertuoti į 0-1
    binary = (im // 255).astype(np.uint8)

    # Kaimynų skaičiavimo kernelis
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Suskaičiuoti kaimynus naudojant konvoliuciją
    # borderType=BORDER_CONSTANT: kraštai = 0, kad kraštiniai pikseliai
    # turėtų teisingą kaimynų skaičių
    neighbor_count = cv2.filter2D(binary, -1, kernel,
                                  borderType=cv2.BORDER_CONSTANT)

    # Pašalinti TIK vidinius pikselius be kaimynų
    interior = np.zeros_like(binary, dtype=bool)
    interior[1:-1, 1:-1] = True

    binary[(binary == 1) & (neighbor_count == 0) & interior] = 0

    return binary * 255


def cleareJoints(im: np.ndarray) -> np.ndarray:
    """
    Pašalina susikirtimų taškus.

C++ palikta kaip užkomentuota



    Args:
        im: Binarinis paveikslėlis (0-255)

    Returns:
        Išvalytas paveikslėlis (0-255)
    """
    im_bin = (im // 255).astype(np.uint8)

    for i in range(1, im_bin.shape[0] - 1):
        for j in range(1, im_bin.shape[1] - 1):
            if im_bin[i, j] != 1:
                continue

            found = False

            p2 = im_bin[i - 1, j]
            p3 = im_bin[i - 1, j + 1]
            p4 = im_bin[i, j + 1]
            p5 = im_bin[i + 1, j + 1]
            p6 = im_bin[i + 1, j]
            p7 = im_bin[i + 1, j - 1]
            p8 = im_bin[i, j - 1]
            p9 = im_bin[i - 1, j - 1]

            # Kryžminių kaimynų suma (+ 1 už centrą)
            s = 1 + p2 + p4 + p6 + p8
            if s > 3:
                found = True

            # Diagonaliniai
            if p9 == 1 and p6 == 1 and p4 == 1:
                found = True
            if p7 == 1 and p2 == 1 and p4 == 1:
                found = True
            if p3 == 1 and p6 == 1 and p8 == 1:
                found = True
            if p5 == 1 and p2 == 1 and p8 == 1:
                found = True

            if found:
                im_bin[i, j] = 0
                im_bin[i - 1, j] = 0
                im_bin[i - 1, j + 1] = 0
                im_bin[i, j + 1] = 0
                im_bin[i + 1, j + 1] = 0
                im_bin[i + 1, j] = 0
                im_bin[i + 1, j - 1] = 0
                im_bin[i, j - 1] = 0
                im_bin[i - 1, j - 1] = 0

    return im_bin * 255


def bwe1(img_green: np.ndarray, img_mask: np.ndarray, prop: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kraujagyslių išskyrimas (Blood Vessel Extraction).
    Pridėti saugikliai.

    Args:
        img_green: Žalias kanalas (apdorotas)
        img_mask: Binari kaukė
        prop: Proporcingumo koeficientas (sc)

    Returns:
        tuple: (img_proc_t, img_proc_thn)
    """
    # Struktūriniai elementai
    b1 = int(2 * prop) * 2 + 1
    b2 = int(4 * prop) * 2 + 1

    B1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (b1, b1))
    B2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (b2, b2))

    # Closing su B2
    img_proc1 = cv2.dilate(img_green, B2)
    img_proc1 = cv2.erode(img_proc1, B2)

    # Closing su B1
    img_proc2 = cv2.dilate(img_green, B1)
    img_proc2 = cv2.erode(img_proc2, B1)

    # Skirtumas
    img_proc = cv2.subtract(img_proc1, img_proc2)


    # kaukė bus pritaikyta vėliau prieš thinning

    # Binarizavimas
    min_val, max_val, _, _ = cv2.minMaxLoc(img_proc)
    thresh = max_val * 0.1
    _, img_proc_t = cv2.threshold(img_proc, thresh, 255, cv2.THRESH_BINARY)
    img_proc_t = img_proc_t.astype(np.uint8)

    # Median blur #C nera saugikliu
    median_size = int(2 * prop) * 2 + 1
    # Minimalus OpenCV reikalavimas: >= 3 ir nelyginis
    if median_size < 3:
        median_size = 3
    if median_size % 2 == 0:
        median_size += 1
    img_proc_t = cv2.medianBlur(img_proc_t, median_size)

    # Ploninimas
    img_proc_thn = img_proc_t.copy()
    img_proc_thn = cv2.bitwise_and(img_proc_thn, img_proc_thn, mask=img_mask)
    img_proc_thn = thinning(img_proc_thn)

    # Pašalinti izoliuotus pikselius
    img_proc_thn = cleareIsolated(img_proc_thn)

    return img_proc_t, img_proc_thn