"""Image filtering functions (RGB uint8 arrays)."""
import cv2
import numpy as np


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def detect_edges(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    if not 0 <= low < high <= 255:
        raise ValueError("Thresholds must satisfy 0 <= low < high <= 255")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, low, high)
