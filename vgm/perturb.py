"""Image-space perturbations for the Task 4/5 robustness study.

These approximate illumination and sensor changes on rendered frames. They are
labelled as image-space perturbations in all outputs; scene-level failure cases
(distractors, occlusion, adjacency) are produced in the simulator instead.
"""
from __future__ import annotations

import cv2
import numpy as np


def brightness(bgr, gain: float):
    return np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def gamma(bgr, g: float):
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def color_cast(bgr, bgr_gains=(1.0, 1.0, 1.0)):
    return np.clip(bgr.astype(np.float32) * np.asarray(bgr_gains, np.float32), 0, 255).astype(np.uint8)


def gaussian_noise(bgr, sigma: float, rng: np.random.Generator):
    noisy = bgr.astype(np.float32) + rng.normal(0.0, sigma, bgr.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def depth_noise(depth_m, sigma_m: float, rng: np.random.Generator):
    return depth_m + rng.normal(0.0, sigma_m, depth_m.shape).astype(depth_m.dtype)


# name -> (rgb transform, depth transform); each takes (array, rng).
IMAGE_SCENARIOS = {
    'nominal': (None, None),
    'dim_0.5x': (lambda im, r: brightness(im, 0.5), None),
    'dark_0.3x': (lambda im, r: brightness(im, 0.3), None),
    'bright_1.6x': (lambda im, r: brightness(im, 1.6), None),
    'gamma_2.0': (lambda im, r: gamma(im, 2.0), None),
    'warm_cast': (lambda im, r: color_cast(im, (0.75, 0.95, 1.25)), None),
    'rgb_noise_s15': (lambda im, r: gaussian_noise(im, 15.0, r), None),
    'depth_noise_2mm': (None, lambda d, r: depth_noise(d, 0.002, r)),
    'depth_noise_5mm': (None, lambda d, r: depth_noise(d, 0.005, r)),
}
