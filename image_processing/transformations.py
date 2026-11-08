"""Basic image transformations. All input images are RGB uint8 arrays."""
import cv2
import numpy as np


def grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def binary_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    if not 0 <= threshold <= 255:
        raise ValueError("Threshold must be in [0, 255]")
    _, result = cv2.threshold(grayscale(image), threshold, 255, cv2.THRESH_BINARY)
    return result


def adjust_brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
    if not -255 <= brightness <= 255 or contrast < 0:
        raise ValueError("Invalid brightness or contrast")
    # float operations avoid uint8 overflow before clipping
    return np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if width < 1:
        raise ValueError("Width must be >= 1")
    h, w = image.shape[:2]
    height = max(1, round(h * width / w))
    method = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=method)


def rotate_right_angle(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return image.copy()
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Degrees must be 0, 90, 180 or 270")


def flip_image(image: np.ndarray, direction: str) -> np.ndarray:
    if direction == "None":
        return image.copy()
    if direction == "Horizontal":
        return cv2.flip(image, 1)
    if direction == "Vertical":
        return cv2.flip(image, 0)
    raise ValueError("Unsupported flip direction")
