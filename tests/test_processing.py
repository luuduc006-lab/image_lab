import numpy as np
import pytest

from image_processing.analysis import image_details, rgb_histograms
from image_processing.filters import gaussian_blur, sharpen, detect_edges
from image_processing.transformations import (
    grayscale, binary_threshold, adjust_brightness_contrast,
    resize_to_width, rotate_right_angle, flip_image,
)

@pytest.fixture
def sample():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[10:30, 15:45] = (200, 120, 50)
    return image


def test_grayscale(sample):
    result = grayscale(sample)
    assert result.shape == (40, 60) and result.dtype == np.uint8


def test_binary_values(sample):
    assert set(np.unique(binary_threshold(sample, 100))).issubset({0, 255})


def test_blur_and_sharpen_shape(sample):
    assert gaussian_blur(sample, 5).shape == sample.shape
    assert sharpen(sample).shape == sample.shape


def test_invalid_kernel(sample):
    with pytest.raises(ValueError):
        gaussian_blur(sample, 4)


def test_edge_shape(sample):
    assert detect_edges(sample, 40, 150).shape == (40, 60)


def test_brightness_clips(sample):
    brighter = adjust_brightness_contrast(sample, 100, 2)
    assert brighter.dtype == np.uint8
    assert brighter.max() == 255


def test_resize_preserves_ratio(sample):
    assert resize_to_width(sample, 120).shape == (80, 120, 3)


def test_rotate(sample):
    assert rotate_right_angle(sample, 90).shape == (60, 40, 3)
    assert np.array_equal(rotate_right_angle(sample, 0), sample)


def test_flip(sample):
    assert np.array_equal(flip_image(sample, "Horizontal")[:, 0], sample[:, -1])


def test_histogram_counts(sample):
    for hist in rgb_histograms(sample).values():
        assert hist.shape == (256,)
        assert hist.sum() == 40 * 60


def test_metadata(sample):
    info = image_details(sample)
    assert (info["width"], info["height"], info["channels"]) == (60, 40, 3)
