"""Checkerboard pose generators shared by capture and the synthetic self-test.

'repo'  : the original profile (board 0.86-0.98 m from the camera, tilts <= 27 deg).
'wide'  : strong tilts (25-55 deg about a random in-plane axis) and a 0.45-0.95 m
          depth range, which is what Zhang's method needs to separate focal length
          from board distance (Zhang 2000; Bouguet's toolbox guidance).
The camera sits at world z = 1.0 m looking straight down.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

CAMERA_Z = 1.0
HALF_FOV_X = np.radians(30.0)   # nominal 60 deg on the 960 px axis
ASPECT = 720 / 960


def repo_pose(rng: np.random.Generator, attempt: int) -> list[float]:
    if attempt == 0:
        return [0.0, 0.0, 0.07, 0.0, 0.0, 0.0, 1.0]
    xyz = [float(rng.uniform(-0.16, 0.16)), float(rng.uniform(-0.11, 0.11)),
           float(rng.uniform(0.02, 0.14))]
    rpy = [float(rng.uniform(-27, 27)), float(rng.uniform(-27, 27)), float(rng.uniform(-32, 32))]
    return xyz + Rotation.from_euler('xyz', rpy, degrees=True).as_quat().tolist()


def wide_pose(rng: np.random.Generator, attempt: int) -> list[float]:
    if attempt == 0:
        return [0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0]
    z = float(rng.uniform(0.05, 0.55))
    dist = CAMERA_Z - z
    # Keep the board centre well inside the frustum at this depth.
    half_x = max(0.0, 0.55 * dist * np.tan(HALF_FOV_X) - 0.06)
    half_y = max(0.0, 0.55 * dist * np.tan(HALF_FOV_X) * ASPECT - 0.06)
    xy = [float(rng.uniform(-half_x, half_x)), float(rng.uniform(-half_y, half_y))]
    tilt = float(rng.uniform(25, 55)) * (1 if rng.random() < .5 else -1)
    axis_angle = float(rng.uniform(0, 2 * np.pi))
    axis = np.array([np.cos(axis_angle), np.sin(axis_angle), 0.0])
    R = Rotation.from_rotvec(np.radians(tilt) * axis) * Rotation.from_euler('z', rng.uniform(-45, 45), degrees=True)
    return xy + [z] + R.as_quat().tolist()


PROFILES = {'repo': repo_pose, 'wide': wide_pose}
