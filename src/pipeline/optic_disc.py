"""
optic_disc.py - Optinio disko aptikimas

- calcNorm() - normaliojo skirstinio skaičiavimas
- checkValidOD() - patikrinti ar OD validus
- opticDiscInit1() - preliminarus optinio disko aptikimas
- opticDisc2() - tikslus optinio disko aptikimas
"""

import cv2
import numpy as np
import math
from typing import Tuple

from .masking import normingMask, norming


def calcNorm(x: int, miu: float, sigma: float, mult: float) -> float:
    """
    Normaliojo skirstinio reikšmės skaičiavimas.

    Args:
        x: Įvesties reikšmė
        miu: Vidurkis (mean)
        sigma: Dispersija (variance)
        mult: Dauginimo koeficientas

    Returns:
        Normaliojo skirstinio reikšmė
    """
    return (mult / math.sqrt(2.0 * math.pi * sigma)) * math.exp(-math.pow(x - miu, 2) / (2.0 * sigma))


def checkValidOD(img: np.ndarray, center: Tuple[int, int], radius: float) -> bool:
    """
    Patikrina ar apskritimas gali būti tinkamas optinis diskas.

    Args:
        img: Pilkas paveikslėlis
        center: Centro koordinatės (x, y)
        radius: Spindulys

    Returns:
        True jei validus OD
    """
    # Sukurti kaukes
    mask1 = np.zeros(img.shape[:2], dtype=np.uint8)
    mask2 = np.zeros(img.shape[:2], dtype=np.uint8)

    # Vidinė kaukė (OD sritis)
    cv2.circle(mask1, center, int(radius), 255, -1)

    # Išorinė kaukė (žiedas aplink OD)
    cv2.circle(mask2, center, int(radius * 1.5), 255, -1)
    cv2.circle(mask2, center, int(radius), 0, -1)

    # Apskaičiuoti vidurkius
    mean1 = cv2.mean(img, mask=mask1)[0]  # Vidinis regionas (OD)
    mean2 = cv2.mean(img, mask=mask2)[0]  # Išorinis žiedas

    # Originalaus C++ kodo klaida: return(mean1[0] <= mean2[0]);
    # turėtų būti optinio disko vidus šviesesnis už išorę
    return mean1 >= mean2 * 0.9


def opticDiscInit1(img_green: np.ndarray, img_proc_thn: np.ndarray,
                   img_mask: np.ndarray, sc: float,
                   btsz_from: int = 80) -> Tuple[int, int, int]:
    """
    Preliminarus optinio disko aptikimas.

    Algoritmas:
    1. Hough linijų aptikimas ant suplonintos kraujagyslių nuotraukos
    2. Linijų susikirtimų radimas (OD centre susikerta daugiausiai kraujagyslių)
    3. Šviesumo tikimybės žemėlapio sudarymas
    4. Gradiento triukšmo pašalinimas
    5. Optimalaus OD dydžio paieška konvoliuojant elipsės filtrus
    6. Validavimas - ar yra kraujagyslių susikirtimų

    Args:
        img_green: Žalias kanalas
        img_proc_thn: Suplonintas kraujagyslių paveikslėlis
        img_mask: Akies dugno kaukė
        sc: Skalės koeficientas
        btsz_from: Pradinis OD dydžio paieškos taškas (numatytasis 80)

    Returns:
        tuple: (cx_i, cy_i, radius_i) - centro x, y ir spindulys
    """
    # Sukurti linijų susikirtimų paveikslėlį
    img_lint = np.zeros_like(img_proc_thn)

    # Hough linijų aptikimas
    threshold = int(round(20 * sc))
    lines = cv2.HoughLines(img_proc_thn, 1, np.pi / 180, threshold)

    lines_p1 = []  # Linijos su kampu < PI/4
    lines_p2 = []  # Linijos su kampu > 3*PI/4

    if lines is not None:
        all_lines = lines[:, 0, :]  # (N, 2)
        thetas = all_lines[:, 1]
        mask1 = thetas < np.pi / 4
        mask2 = thetas > 3 * np.pi / 4
        lines_p1 = [(r, t) for r, t in all_lines[mask1]]
        lines_p2 = [(r, t) for r, t in all_lines[mask2]]

    # Rasti susikirtimus tarp statmenų linijų (VEKTORIZUOTA)
    if lines_p1 and lines_p2:
        arr1 = np.array(lines_p1)  # (N, 2) - rho, theta
        arr2 = np.array(lines_p2)  # (M, 2)

        sin1 = np.sin(arr1[:, 1])
        cos1 = np.cos(arr1[:, 1])
        sin2 = np.sin(arr2[:, 1])
        cos2 = np.cos(arr2[:, 1])

        # Pašalinti sin==0 atvejus
        valid1 = sin1 != 0
        valid2 = sin2 != 0

        if np.any(valid1) and np.any(valid2):
            a = -cos1[valid1] / sin1[valid1]  # (N',)
            c = arr1[valid1, 0] / sin1[valid1]
            b = -cos2[valid2] / sin2[valid2]  # (M',)
            d = arr2[valid2, 0] / sin2[valid2]

            theta1_v = arr1[valid1, 1]
            theta2_v = arr2[valid2, 1]

            # Outer products: (N', M')
            A, B = np.meshgrid(a, b, indexing='ij')
            C, D = np.meshgrid(c, d, indexing='ij')
            T1, T2 = np.meshgrid(theta1_v, theta2_v, indexing='ij')

            denom = A - B
            slope_diff = np.abs(T1 - T2)

            valid_mask = (denom != 0) & (slope_diff > 1)

            if np.any(valid_mask):
                x_vals = np.round((D[valid_mask] - C[valid_mask]) / denom[valid_mask]).astype(int)
                y_vals = np.round(A[valid_mask] * ((D[valid_mask] - C[valid_mask]) / denom[valid_mask]) + C[valid_mask]).astype(int)

                # Filtruoti pagal ribas
                in_bounds = ((x_vals > 0) & (x_vals < img_lint.shape[1]) &
                             (y_vals > 0) & (y_vals < img_lint.shape[0]))

                x_valid = x_vals[in_bounds]
                y_valid = y_vals[in_bounds]

                if len(x_valid) > 0:
                    img_lint[y_valid, x_valid] = 255

    # Išplėsti susikirtimų taškus
    b1 = int(round(5 * sc))
    if b1 % 2 == 0:
        b1 += 1
    B1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (b1, b1))
    img_lint = cv2.dilate(img_lint, B1)

    # Suliejimas - sukurti tikimybės žemėlapį
    blrsz = int(100 * sc)
    if blrsz % 2 == 0:
        blrsz += 1
    img_lint = cv2.GaussianBlur(img_lint, (blrsz, blrsz), blrsz)

    # Pritaikyti kaukę
    img_lint = cv2.bitwise_and(img_lint, img_lint, mask=img_mask)

    # Šviesumo žemėlapis (img_J)
    img_J = cv2.bitwise_and(img_green, img_green, mask=img_mask)

    # Pašalinti kraujagyslių įtaką morfologiniu uždarymu
    blrsz = int(15 * sc)
    if blrsz % 2 == 0:
        blrsz += 1
    B_cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blrsz, blrsz))
    img_J = cv2.dilate(img_J, B_cl)
    img_J = cv2.erode(img_J, B_cl)

    # Papildomas suliejimas
    blrsz = int(4 * sc)
    if blrsz % 2 == 0:
        blrsz += 1
    if blrsz < 3:
        blrsz = 3
    img_J = cv2.GaussianBlur(img_J, (blrsz, blrsz), blrsz)

    # Tikimybė, kad OD arčiau centro (Gauso skirstinys) - VEKTORIZUOTA
    gr = img_green.shape[0]
    gr_m = gr // 2
    sigma = math.pow(gr_m / 3.0, 2)
    mult = 100.0 * gr
    r_vals = np.arange(gr, dtype=np.float64)
    cr_vals = (mult / math.sqrt(2.0 * math.pi * sigma)) * np.exp(-(r_vals - gr_m) ** 2 / (2.0 * sigma))
    cr_vals = cr_vals.astype(np.int64).astype(np.float64)
    grad = np.broadcast_to(cr_vals.reshape(-1, 1), (gr, img_green.shape[1])).copy()

    # Pašalinti gradientinį triukšmą - vertikalus (VEKTORIZUOTA)
    row_mean = cv2.reduce(img_green, 1, cv2.REDUCE_AVG)  # shape (H, 1)
    grad_noise = np.broadcast_to(row_mean.astype(np.float64), (img_green.shape[0], img_green.shape[1])).copy()

    blrsz = int(100 * sc)
    if blrsz % 2 == 0:
        blrsz += 1
    grad_noise = cv2.GaussianBlur(grad_noise, (blrsz, blrsz), blrsz)

    # Pašalinti gradientinį triukšmą - horizontalus (VEKTORIZUOTA)
    col_mean = cv2.reduce(img_green, 0, cv2.REDUCE_AVG)  # shape (1, W)
    grad_noise_v = np.broadcast_to(col_mean.astype(np.float64), (img_green.shape[0], img_green.shape[1])).copy()

    grad_noise_v = cv2.GaussianBlur(grad_noise_v, (blrsz, blrsz), blrsz)

    # Sujungti visus žemėlapius
    img_J = img_J.astype(np.float64)
    img_lint = img_lint.astype(np.float64)

    # img_J = 1.0*img_J + 0.2*grad + 0.2*img_lint - 0.4*grad_noise - 0.3*grad_noise_v
    img_J = 1.0 * img_J + 0.2 * grad + 0.2 * img_lint - 0.4 * grad_noise - 0.3 * grad_noise_v

    # Normalizuoti kaukės srityje
    img_J = np.clip(img_J, 0, 255).astype(np.uint8)
    img_J = normingMask(img_J, img_mask)

    # Paruošti OD dydžio paieškai
    imgbtsz_init = img_J.copy().astype(np.uint8)

    btsz_max = 20
    btsz_max_val = 0.0
    btsz_max_loc = (0, 0)
    btsz_wd = 0
    btsz_to = 200

    # Kopijuoti kaukę (kad nekeistume originalios)
    img_mask_work = img_mask.copy()

    # OPTIMIZACIJA: downsampling koeficientas filter2D ciklui
    # Mažesnis paveikslėlis → ~ds² kartų greičiau, lokacija vis tiek tiksli
    ds = 4  # downsampling faktorius
    h_orig, w_orig = imgbtsz_init.shape[:2]
    h_ds, w_ds = max(1, h_orig // ds), max(1, w_orig // ds)

    goodOD = False
    max_iterations = 10
    itr = 0

    while not goodOD and itr < max_iterations:
        btsz_max = 20
        btsz_max_val = 0.0

        # Sumažinti paveikslėlį prieš filter2D ciklą
        imgbtsz_ds = cv2.resize(imgbtsz_init, (w_ds, h_ds), interpolation=cv2.INTER_AREA)

        # Ieškoti optimalaus OD dydžio (ant sumažinto paveiksliuko)
        for btsz in range(btsz_from, btsz_to + 1, 5):
            btszc = int(btsz * sc)
            if btszc < 3:
                btszc = 3

            # Kernelio dydis sumažintam paveiksliukui
            btszc_ds = max(3, btszc // ds)
            if btszc_ds % 2 == 0:
                btszc_ds += 1

            # Sukurti elipsės filtrą (mažesnį)
            Bt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (btszc_ds, btszc_ds))
            Bt = Bt.astype(np.float64)
            s = np.sum(Bt)
            if s > 0:
                Bt = Bt / s

            # Konvoliuoti (ant mažo paveiksliuko - DAUG greičiau)
            imgbtsz = cv2.filter2D(imgbtsz_ds.astype(np.float64), cv2.CV_64F, Bt)

            # Rasti maksimumą
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgbtsz)

            if btsz_max_val < max_val:
                btsz_max_val = max_val
                btsz_max = btsz
                # Perskaičiuoti lokaciją į originalias koordinates
                btsz_max_loc = (max_loc[0] * ds, max_loc[1] * ds)
                btsz_wd = int(btszc / 2)

        # Patikrinti ar OD kertasi su kraujagyslėmis (originaliose koordinatėse)
        GTestM = np.zeros_like(img_proc_thn)
        cv2.circle(GTestM, btsz_max_loc, btsz_wd * 2, 255, -1)
        GTest = cv2.bitwise_and(img_proc_thn, GTestM)

        if cv2.countNonZero(GTest) > 1:
            goodOD = True
        else:
            # Pašalinti šią sritį ir bandyti vėl (originaliame paveiksliuke)
            GTestM = np.zeros_like(img_proc_thn)
            cv2.circle(GTestM, btsz_max_loc, btsz_wd * 2, 255, -1)
            GTestM_inv = cv2.bitwise_not(GTestM)
            imgbtsz_init = cv2.bitwise_and(imgbtsz_init, imgbtsz_init, mask=GTestM_inv)
            img_mask_work = cv2.bitwise_and(img_mask_work, GTestM_inv)
            itr += 1

    cx_i = btsz_max_loc[0]
    cy_i = btsz_max_loc[1]
    radius_i = btsz_wd

    return cx_i, cy_i, radius_i


def opticDisc2(cx_i: int, cy_i: int, radius_i: int,
               img_green: np.ndarray, img_mask: np.ndarray,
               sc: float) -> Tuple[int, int, int]:
    """
    Tikslus optinio disko aptikimas naudojant Hough apskritimus.

    Algoritmas:
    1. Sukurti paieškos sritį aplink preliminarų OD
    2. Morfologinis uždarymas
    3. Sobel gradientai (Dx, Dy)
    4. Binarizavimas
    5. Hough apskritimų aptikimas
    6. Geriausio apskritimo parinkimas pagal gradiento sutapimą

    Args:
        cx_i, cy_i, radius_i: Preliminaraus OD koordinatės
        img_green: Žalias kanalas
        img_mask: Akies dugno kaukė
        sc: Skalės koeficientas

    Returns:
        tuple: (cx, cy, radius) - tikslios OD koordinatės
    """
    # Sukurti paieškos kaukę
    subImg_mask = np.zeros_like(img_mask)
    cv2.circle(subImg_mask, (cx_i, cy_i), int(radius_i * 2.5), 255, -1)
    subImg_mask = cv2.bitwise_and(img_mask, subImg_mask)

    # Sukurti eroduotą kaukę
    subImg_mask2 = cv2.bitwise_and(subImg_mask, img_mask)
    b1 = int(30 * sc)
    if b1 % 2 == 0:
        b1 += 1
    if b1 < 3:
        b1 = 3
    B1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (b1, b1))
    subImg_mask2 = cv2.erode(subImg_mask2, B1)

    # Paruošti paveikslėlį
    img_J = img_green.copy()
    subImg = cv2.bitwise_and(img_J, img_J, mask=subImg_mask)

    # Morfologinis uždarymas
    b2 = int(20 * sc)
    if b2 % 2 == 0:
        b2 += 1
    if b2 < 3:
        b2 = 3
    B2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (b2, b2))
    subImg = cv2.dilate(subImg, B2)
    subImg = cv2.erode(subImg, B2)

    # Gaussian blur
    gbsz = int(10 * sc)
    if gbsz < 3:
        gbsz = 3
    if gbsz % 2 == 0:
        gbsz += 1
    subImg = cv2.GaussianBlur(subImg, (gbsz, gbsz), gbsz)

    # Sobel gradientai
    Dx = cv2.Sobel(subImg, cv2.CV_64F, 1, 0, ksize=3)
    Dx = cv2.convertScaleAbs(Dx)

    Dy = cv2.Sobel(subImg, cv2.CV_64F, 0, 1, ksize=3)
    Dy = cv2.convertScaleAbs(Dy)

    # Sujungti gradientus
    subImg_dst = cv2.addWeighted(Dx, 0.5, Dy, 0.5, 0)
    subImg_dst = cv2.convertScaleAbs(subImg_dst)

    # Pritaikyti kaukę ir normalizuoti
    subImg_dst2 = cv2.bitwise_and(subImg_dst, subImg_dst, mask=subImg_mask2)
    subImg_dst2 = norming(subImg_dst2)

    # Binarizavimas
    _, subImg_dst2 = cv2.threshold(subImg_dst2, 32, 255, cv2.THRESH_BINARY)

    # Morfologinės operacijos - open ir close
    b3 = int(3 * sc)
    if b3 % 2 == 0:
        b3 += 1
    if b3 < 3:
        b3 = 3
    B3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (b3, b3))

    # Open
    subImg_dst2 = cv2.erode(subImg_dst2, B3)
    subImg_dst2 = cv2.dilate(subImg_dst2, B3)

    # Close
    subImg_dst2 = cv2.dilate(subImg_dst2, B3)
    subImg_dst2 = cv2.erode(subImg_dst2, B3)

    # Nustatyti spindulio paieškos ribas
    rad_from = int(50 * sc)
    rad_to = int(100 * sc)

    # Koreguoti pagal preliminarų spindulį
    df = 0
    if radius_i < rad_from:
        df = int(radius_i - rad_from - 5 * sc)
    elif radius_i > rad_to:
        df = int(radius_i - rad_to + 5 * sc)

    rad_from += df
    rad_to += df

    # Užtikrinti teigiamas reikšmes
    rad_from = max(1, rad_from)
    rad_to = max(rad_from + 1, rad_to)

    # Hough apskritimų aptikimas
    param1 = 20  # Canny slenkstis
    param2 = int(5 * sc)  # Centro slenkstis
    param2 = max(1, param2)

    circles = cv2.HoughCircles(
        subImg_dst2,
        cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=4.0 * sc,
        param1=param1,
        param2=param2,
        minRadius=rad_from,
        maxRadius=rad_to
    )

    # Inicializuoti rezultatus
    cx = 0
    cy = 0
    radius = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))

        best_circle_idx = 0
        max_overlap = 0

        # Ieškoti geriausio apskritimo
        for i in range(min(len(circles[0]), 10)):
            center = (int(circles[0][i][0]), int(circles[0][i][1]))
            rad = int(circles[0][i][2])

            # Patikrinti ar validus OD
            if checkValidOD(img_green, center, rad):
                # Apskaičiuoti sutapimą su gradiento kraštais
                circleTest = np.zeros_like(subImg_dst2)
                cv2.circle(circleTest, center, rad, 255, 10)
                circleTest = cv2.bitwise_and(subImg_dst2, circleTest)
                overlap = cv2.countNonZero(circleTest)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_circle_idx = i

        # Grąžinti geriausią apskritimą
        if max_overlap > 0 or len(circles[0]) > 0:
            cx = int(circles[0][best_circle_idx][0])
            cy = int(circles[0][best_circle_idx][1])
            radius = int(circles[0][best_circle_idx][2])

    return cx, cy, radius


def detect_optic_disc(img_green: np.ndarray, img_proc_thn: np.ndarray,
                      img_mask: np.ndarray, sc: float) -> Tuple[int, int, int]:
    """
    Pilnas optinio disko aptikimas.

    C++ algoritmas:
    1. opticDiscInit1() - preliminarus aptikimas pagal linijų susikirtimus
    2. opticDisc2() - tikslus aptikimas su Hough apskritimais
    3. FOV FIX: jei od_r == 0, bandyti su btsz_from=50

    Args:
        img_green: Žalias kanalas (apdorotas)
        img_proc_thn: Suplonintas kraujagyslių paveikslėlis
        img_mask: Akies dugno kaukė
        sc: Skalės koeficientas

    Returns:
        tuple: (od_x, od_y, od_r) - optinio disko koordinatės ir spindulys
    """
    # 1. Preliminarus aptikimas (C++ opticDiscInit1)
    cx_i, cy_i, radius_i = opticDiscInit1(img_green, img_proc_thn, img_mask, sc, btsz_from=80)

    # Jei nepavyko rasti - grąžinti nulius
    if cx_i == 0 and cy_i == 0:
        return 0, 0, 0

    # 2. Tikslus aptikimas (C++ opticDisc2)
    od_x, od_y, od_r = opticDisc2(cx_i, cy_i, radius_i, img_green, img_mask, sc)

   
    # FOV FIX
    # Jei od_r == 0, bandyti su sumažintu btsz_from=50
    if od_r == 0:
        # Pakartotinis bandymas su mažesniu btsz_from
        cx_i, cy_i, radius_i = opticDiscInit1(img_green, img_proc_thn, img_mask, sc, btsz_from=50)
        if cx_i > 0 or cy_i > 0:
            od_x, od_y, od_r = opticDisc2(cx_i, cy_i, radius_i, img_green, img_mask, sc)

    # Jei vis dar nepavyko - grąžinti preliminarų rezultatą
    if od_x == 0 and od_y == 0:
        return cx_i, cy_i, radius_i

    return od_x, od_y, od_r