"""
preprocessing.py - Paveikslėlių išankstinis apdorojimas

- preprocessing1() - skirtas kraujagyslėms
- preprocessing2() - skirtas OD (optiniam diskui)
- preprocessing3() - skirtas matavimams
- preprocessing4() - skirtas matavimams (išplėstinis)
"""

import cv2
import numpy as np

from .masking import normingMask


def preprocessing1(img: np.ndarray, img_mask: np.ndarray, sc: float) -> np.ndarray:
    """
    Išankstinis apdorojimas kraujagyslių išskyrimui.

    Algoritmas:
    1. CLAHE (clipLimit=2) ant žalio kanalo x2
    2. Normalizacija kaukės srityje
    3. Gaussian blur (dydis priklauso nuo sc)
    4. Non-local means denoising

    Args:
        img: BGR paveikslėlis
        img_mask: Binari kaukė
        sc: Skalės koeficientas

    Returns:
        Apdorotas BGR paveikslėlis
    """
    # Išskaidyti kanalus (B, G, R)
    ch = list(cv2.split(img))

    # CLAHE su clipLimit=2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Taikyti CLAHE žaliam kanalui du kartus
    ch[1] = clahe.apply(ch[1])
    ch[1] = clahe.apply(ch[1])

    # Normalizuoti kaukės srityje
    ch[1] = normingMask(ch[1], img_mask)

    # Gaussian blur - dydis priklauso nuo skalės
    blur_size = int(2 * sc) * 2 + 1
    if blur_size < 3:
        blur_size = 3
    ch[1] = cv2.GaussianBlur(ch[1], (blur_size, blur_size), 1)

    # Non-local means denoising
    # h=7, templateWindowSize=7, searchWindowSize=21
    ch[1] = cv2.fastNlMeansDenoising(ch[1], None, h=7,
                                      templateWindowSize=7,
                                      searchWindowSize=21)

    # Sujungti kanalus
    result = cv2.merge(ch)

    return result


def preprocessing2(img: np.ndarray) -> np.ndarray:
    """
    Išankstinis apdorojimas optinio disko aptikimui.

    Algoritmas:
    1. Konvertuoti į HSV
    2. CLAHE (clipLimit=2) ant V kanalo
    3. Konvertuoti atgal į BGR
    4. Mėlyno kanalo apdorojimas (x0.5, median, gaussian)

    Args:
        img: BGR paveikslėlis

    Returns:
        Apdorotas BGR paveikslėlis
    """
    # Konvertuoti į HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Išskaidyti kanalus (H, S, V)
    ch_hsv = list(cv2.split(img_hsv))

    # CLAHE ant V (value) kanalo
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch_hsv[2] = clahe.apply(ch_hsv[2])

    # Sujungti ir konvertuoti atgal į BGR
    img_hsv = cv2.merge(ch_hsv)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Išskaidyti BGR kanalus
    ch = list(cv2.split(img_bgr))

    # Mėlynas kanalas * 0.5
    ch[0] = (ch[0].astype(np.float32) * 0.5).astype(np.uint8)

    # Median blur ant mėlyno kanalo
    ch[0] = cv2.medianBlur(ch[0], 3)

    # Gaussian blur ant mėlyno kanalo
    ch[0] = cv2.GaussianBlur(ch[0], (3, 3), 1)

    # Sujungti kanalus
    result = cv2.merge(ch)

    return result


def preprocessing3(img: np.ndarray, img_mask: np.ndarray) -> np.ndarray:
    """
    Išankstinis apdorojimas matavimams.

    Algoritmas:
    1. CLAHE (clipLimit=1) ant žalio ir raudono kanalų
    2. Non-local means denoising ant žalio ir raudono
    3. Normalizacija žalio kanalo kaukės srityje

    Args:
        img: BGR paveikslėlis
        img_mask: Binari kaukė

    Returns:
        Apdorotas BGR paveikslėlis
    """
    # Išskaidyti kanalus (B, G, R)
    ch = list(cv2.split(img))

    # CLAHE su clipLimit=1
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

    # Taikyti CLAHE žaliam ir raudonam kanalams
    ch[1] = clahe.apply(ch[1])  # Green
    ch[2] = clahe.apply(ch[2])  # Red

    # Non-local means denoising
    # h=3, templateWindowSize=3, searchWindowSize=11
    ch[1] = cv2.fastNlMeansDenoising(ch[1], None, h=3,
                                      templateWindowSize=3,
                                      searchWindowSize=11)
    ch[2] = cv2.fastNlMeansDenoising(ch[2], None, h=3,
                                      templateWindowSize=3,
                                      searchWindowSize=11)

    # Normalizuoti žalią kanalą kaukės srityje
    ch[1] = normingMask(ch[1], img_mask)

    # Sujungti kanalus
    result = cv2.merge(ch)

    return result


def preprocessing4(img: np.ndarray, img_mask: np.ndarray) -> np.ndarray:
    """
    Išankstinis apdorojimas matavimams (išplėstinis).

    Algoritmas:
    1. CLAHE (clipLimit=1) ant žalio ir raudono kanalų
    2. Non-local means denoising ant visų kanalų
    3. Normalizacija visų kanalų kaukės srityje

    Args:
        img: BGR paveikslėlis
        img_mask: Binari kaukė

    Returns:
        Apdorotas BGR paveikslėlis
    """
    # Išskaidyti kanalus (B, G, R)
    ch = list(cv2.split(img))

    # CLAHE su clipLimit=1
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

    # Taikyti CLAHE žaliam ir raudonam kanalams
    # ch[0] = clahe.apply(ch[0])
    ch[1] = clahe.apply(ch[1])  # Green
    ch[2] = clahe.apply(ch[2])  # Red

    # Non-local means denoising ant visų kanalų
    # h=5, templateWindowSize=5, searchWindowSize=11
    ch[0] = cv2.fastNlMeansDenoising(ch[0], None, h=5,
                                      templateWindowSize=5,
                                      searchWindowSize=11)
    ch[1] = cv2.fastNlMeansDenoising(ch[1], None, h=5,
                                      templateWindowSize=5,
                                      searchWindowSize=11)
    ch[2] = cv2.fastNlMeansDenoising(ch[2], None, h=5,
                                      templateWindowSize=5,
                                      searchWindowSize=11)

    # Normalizuoti visus kanalus kaukės srityje
    ch[0] = normingMask(ch[0], img_mask)
    ch[1] = normingMask(ch[1], img_mask)
    ch[2] = normingMask(ch[2], img_mask)

    # Sujungti kanalus
    result = cv2.merge(ch)
    
    return result