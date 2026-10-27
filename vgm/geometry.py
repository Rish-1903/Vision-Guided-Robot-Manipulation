"""Rigid-body helpers shared by perception, planning and evaluation."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


def obj_pose_matrix(pose) -> np.ndarray:
    """CoppeliaSim pose [x, y, z, qx, qy, qz, qw] -> 4x4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(pose[3:], dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(pose[:3], float)
    return T


def T_to_pose(T) -> list[float]:
    """4x4 homogeneous matrix -> CoppeliaSim pose list."""
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()].tolist()


def norm_yaw_pi(angle: float) -> float:
    """Canonical yaw in [-pi/2, pi/2); a cuboid's yaw is ambiguous modulo pi."""
    return (float(angle) + np.pi / 2) % np.pi - np.pi / 2


def yaw_of(T) -> float:
    return math.atan2(float(T[1, 0]), float(T[0, 0]))


def symmetric_yaw_error_deg(yaw_a: float, yaw_b: float) -> float:
    """Absolute yaw difference for a shape with 180-degree symmetry, in degrees."""
    return abs(math.degrees(norm_yaw_pi(yaw_a - yaw_b)))


def translation_rotation_errors(estimated, truth):
    """Return (translation mm, raw rotation deg, symmetry-aware rotation deg)."""
    p_err_mm = 1000.0 * float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]))
    raw = math.degrees(float(Rotation.from_matrix(estimated[:3, :3].T @ truth[:3, :3]).magnitude()))
    flip = Rotation.from_euler('z', math.pi).as_matrix()
    flipped = math.degrees(float(Rotation.from_matrix(
        estimated[:3, :3].T @ truth[:3, :3] @ flip).magnitude()))
    return p_err_mm, raw, min(raw, flipped)


def top_down_R(yaw: float) -> np.ndarray:
    """Tool frame with z pointing down and x along ``yaw`` in the base XY plane."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]], dtype=float)


def make_tcp_goal(center_B, yaw: float, tcp_above_center: float) -> np.ndarray:
    """Desired TCP pose above an object centre, tool z pointing down."""
    T = np.eye(4)
    T[:3, :3] = top_down_R(yaw)
    T[:3, 3] = np.asarray(center_B, float) + np.array([0.0, 0.0, tcp_above_center])
    return T


def way_above(T, meters: float) -> np.ndarray:
    out = np.array(T, dtype=float, copy=True)
    out[2, 3] += meters
    return out


def perturb_transform(T, rot_deg=(0.0, 0.0, 0.0), trans_mm=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Right-multiply a small error expressed in T's own (child) frame.

    Used to inject a deliberate camera-extrinsic calibration error for the
    open-loop versus closed-loop experiment. ``rot_deg`` is XYZ Euler.
    """
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_euler('xyz', rot_deg, degrees=True).as_matrix()
    delta[:3, 3] = np.asarray(trans_mm, float) / 1000.0
    return np.asarray(T, float) @ delta


def invert(T) -> np.ndarray:
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out
