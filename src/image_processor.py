"""
image_processor.py - Vaizdo apdorojimas atvaizdavimui

Funkcijos skirtos tik DISPLAY tikslams (CLAHE toggle, žalias kanalas).
Pilnas preprocessing pipeline'as OD aptikimui naudoja preprocessing.py.
"""

import cv2
import numpy as np


def apply_clahe_for_display(image_bgr: np.ndarray) -> np.ndarray:
    """
    CLAHE apdorojimas atvaizdavimui (toggle su C klavišu).

    Taikoma CLAHE (clipLimit=2) ant žalio kanalo du kartus.
    Mėlynas ir raudonas kanalai paliekami nepakeisti.

    Args:
        image_bgr: Originalus BGR paveikslėlis

    Returns:
        BGR paveikslėlis su CLAHE ant žalio kanalo
    """
    ch = list(cv2.split(image_bgr))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch[1] = clahe.apply(ch[1])
    ch[1] = clahe.apply(ch[1])
    return cv2.merge(ch)


def get_green_channel(image_bgr: np.ndarray) -> np.ndarray:
    """
    Ištraukia žalią kanalą iš BGR paveikslėlio.

    Žalias kanalas naudojamas:
    - Profilio intensyvumo matavimams (measurement_manager.py)
    - Kraujagyslių vizualizacijai (toggle su G klavišu)

    Args:
        image_bgr: BGR paveikslėlis

    Returns:
        Žalias kanalas (uint8, vienas kanalas)
    """
    return image_bgr[:, :, 1]