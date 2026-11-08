"""Histogram and image metadata."""
import cv2
import numpy as np


def rgb_histograms(image: np.ndarray) -> dict[str, np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB input required")
    return {
        label: cv2.calcHist([image], [idx], None, [256], [0, 256]).ravel()
        for idx, label in enumerate(("R", "G", "B"))
    }


def image_details(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    return {"width": w, "height": h, "channels": 1 if image.ndim == 2 else image.shape[2], "mean": float(np.mean(image))}
