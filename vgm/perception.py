"""Image-only perception: HSV localization (Task 4) and RGB-D pose (Task 5).

Nothing in this module reads simulator state. Inputs are images, depth maps,
and the calibration estimated in Tasks 2-3.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .geometry import norm_yaw_pi

# Rendered material colours and OpenCV hue windows (0..179).
COLORS = {
    'red': {'rgb': (1.0, 0.06, 0.06), 'alias': '/T4_Red', 'hue_ranges': [(0, 12), (170, 179)]},
    'green': {'rgb': (0.04, 1.0, 0.06), 'alias': '/T4_Green', 'hue_ranges': [(40, 86)]},
    'blue': {'rgb': (0.06, 0.12, 1.0), 'alias': '/T4_Blue', 'hue_ranges': [(95, 139)]},
}
# Flat yellow placement pads used as VISUAL targets in the closed-loop experiment.
PAD_SPEC = {'rgb': (1.0, 0.80, 0.0), 'hue_ranges': [(18, 34)]}

BLOCK_SIZE = np.array([0.082, 0.060, 0.038], dtype=float)   # metres
PAD_SIZE = np.array([0.140, 0.110, 0.002], dtype=float)     # metres
TABLE_SIZE = [0.70, 0.58, 0.014]
TABLE_Z = 0.025
TABLE_TOP_Z = TABLE_Z + TABLE_SIZE[2] / 2
BLOCK_Z = TABLE_TOP_Z + BLOCK_SIZE[2] / 2 + 0.001
PAD_Z = TABLE_TOP_Z + PAD_SIZE[2] / 2 + 0.0002

_MORPH_KERNEL = np.ones((3, 3), np.uint8)


def load_calibration(path: Path):
    """Load K, distortion and B_from_Ccv (OpenCV camera -> robot base)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing calibration file: {path}. Expected the Task 3 output '
                                'runs/<id>/task3/recovered/camera_to_robot_transform_recovered.npz')
    with np.load(path) as cal:
        for key in ('K', 'B_from_Ccv'):
            if key not in cal:
                raise KeyError(f'Missing key {key} in {path}')
        K = np.array(cal['K'], dtype=float)
        B_from_C = np.array(cal['B_from_Ccv'], dtype=float)
        dist = np.asarray(cal['dist'] if 'dist' in cal else np.zeros(5), dtype=float)
    if K.shape != (3, 3) or B_from_C.shape != (4, 4):
        raise ValueError('Invalid calibration matrix shape')
    return K, dist, B_from_C


def hue_mask(hsv: np.ndarray, hue_ranges, s_min: int = 80, v_min: int = 45) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for low, high in hue_ranges:
        mask |= cv2.inRange(hsv, (low, s_min, v_min), (high, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)


def _largest_blob(mask: np.ndarray, min_area: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None, 0
    return max(contours, key=cv2.contourArea), len(contours)


def detect_from_rgb(bgr: np.ndarray, min_area: int = 120, specs: dict | None = None):
    """HSV segmentation + largest contour per colour.

    Returns ``(detections, masks)``. ``masks[name]`` contains ONLY the selected
    (largest) blob, so a same-coloured distractor elsewhere in the image does not
    leak into the RGB-D pose fit. ``detections[name]['n_blobs']`` reports how many
    candidate blobs were seen, which flags ambiguous scenes.
    """
    specs = COLORS if specs is None else specs
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    detections, masks = {}, {}
    for name, spec in specs.items():
        full = hue_mask(hsv, spec['hue_ranges'])
        contour, n_blobs = _largest_blob(full, min_area)
        if contour is None:
            detections[name], masks[name] = None, np.zeros_like(full)
            continue
        moments = cv2.moments(contour)
        if abs(moments['m00']) < 1e-10:
            detections[name], masks[name] = None, np.zeros_like(full)
            continue
        selected = np.zeros_like(full)
        cv2.drawContours(selected, [contour], -1, 255, thickness=cv2.FILLED)
        masks[name] = cv2.bitwise_and(full, selected)
        x, y, w, h = cv2.boundingRect(contour)
        detections[name] = {
            'centroid': [float(moments['m10'] / moments['m00']),
                         float(moments['m01'] / moments['m00'])],
            'bbox_xyxy': [int(x), int(y), int(x + w - 1), int(y + h - 1)],
            'area_pixels': float(cv2.contourArea(contour)),
            'image_rect_angle_deg': float(cv2.minAreaRect(contour)[-1]),
            'n_blobs': int(n_blobs),
        }
    return detections, masks


def overlay_image(bgr, detections, ground_truth=None):
    vis = bgr.copy()
    palette = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 60, 0)}
    for name, d in detections.items():
        if d is None:
            continue
        box = d['bbox_xyxy']
        u, v = (int(round(z)) for z in d['centroid'])
        colour = palette.get(name, (0, 220, 255))
        cv2.rectangle(vis, tuple(box[:2]), tuple(box[2:]), colour, 2)
        cv2.circle(vis, (u, v), 4, (255, 255, 255), -1)
        cv2.putText(vis, name, (box[0], max(20, box[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, colour, 2, cv2.LINE_AA)
    for name, gt in (ground_truth or {}).items():
        if gt is None:
            continue
        x1, y1, x2, y2 = gt['bbox_xyxy']
        cv2.rectangle(vis, (round(x1), round(y1)), (round(x2), round(y2)), (245, 245, 245), 1)
        u, v = gt['top_center_uv']
        cv2.drawMarker(vis, (round(u), round(v)), (255, 255, 255), cv2.MARKER_CROSS, 11, 1)
    return vis


def local_corners(size):
    half = np.array(size, dtype=float) / 2
    return np.asarray([[a * half[0], b * half[1], c * half[2]]
                       for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)


def project_box(B_from_O, size, K, dist, B_from_Ccv):
    """Project a box pose (robot base frame) into the image. EVALUATION helper."""
    C_from_B = np.linalg.inv(B_from_Ccv)
    verts_B = local_corners(size) @ B_from_O[:3, :3].T + B_from_O[:3, 3]
    top_B = np.array([0.0, 0.0, size[2] / 2]) @ B_from_O[:3, :3].T + B_from_O[:3, 3]
    xyz_B = np.concatenate([verts_B, top_B[None, :]], axis=0)
    xyz_C = xyz_B @ C_from_B[:3, :3].T + C_from_B[:3, 3]
    if np.any(xyz_C[:, 2] <= 0):
        return None
    uv, _ = cv2.projectPoints(np.ascontiguousarray(xyz_C), np.zeros(3), np.zeros(3), K, dist)
    uv = uv.reshape(-1, 2)
    return {'bbox_xyxy': [float(uv[:8, 0].min()), float(uv[:8, 1].min()),
                          float(uv[:8, 0].max()), float(uv[:8, 1].max())],
            'top_center_uv': uv[8].tolist()}


def box_iou(box_a, box_b) -> float:
    xa, ya, xb, yb = map(float, box_a)
    xc, yc, xd, yd = map(float, box_b)
    inter = max(0., min(xb, xd) - max(xa, xc)) * max(0., min(yb, yd) - max(ya, yc))
    a1, a2 = max(0., xb - xa) * max(0., yb - ya), max(0., xd - xc) * max(0., yd - yc)
    return float(inter / (a1 + a2 - inter)) if a1 + a2 > inter else 0.0


def backproject_pixels(u, v, z, K, dist, depth_mode: str = 'z'):
    """Pixels + depth -> OpenCV camera-frame points (metres).

    depth_mode='z': depth along the optical axis; 'range': Euclidean ray length.
    """
    pixels = np.column_stack((u, v)).astype(np.float64).reshape(-1, 1, 2)
    xy = cv2.undistortPoints(pixels, K, dist).reshape(-1, 2)
    if depth_mode == 'range':
        z = z / np.sqrt(1.0 + (xy ** 2).sum(axis=1))
    return np.column_stack((xy[:, 0] * z, xy[:, 1] * z, z))


def estimate_pose_from_mask(depth_m, mask, K, dist, B_from_C, block_size=BLOCK_SIZE,
                            min_top_points: int = 100, depth_mode: str = 'z',
                            top_tol_m: float = 0.0025):
    """Constrained 6D pose of an upright cuboid from its colour mask and depth.

    1. Erode the mask and back-project valid depth pixels.
    2. Transform to the robot base with the estimated extrinsics.
    3. Keep the top face (points near the 90th-percentile height).
    4. Fit the top-plane normal with SVD and the footprint with minAreaRect.
    5. Offset the top-face centre by half the known height.

    Returns ``(B_from_O, diagnostics)`` or ``(None, {'reason': ...})``.
    """
    if mask is None or depth_m is None:
        return None, {'reason': 'no RGB mask or depth'}
    clean = cv2.erode(mask, _MORPH_KERNEL, iterations=1)
    valid = (clean > 0) & np.isfinite(depth_m) & (depth_m > 0.08) & (depth_m < 5.0)
    vv, uu = np.nonzero(valid)
    if len(uu) < min_top_points:
        return None, {'reason': f'too few colored depth pixels: {len(uu)}'}
    z = depth_m[vv, uu].astype(np.float64)
    cloud_c = backproject_pixels(uu, vv, z, K, dist, depth_mode=depth_mode)
    cloud_b = cloud_c @ B_from_C[:3, :3].T + B_from_C[:3, 3]
    z90 = float(np.percentile(cloud_b[:, 2], 90))
    top = cloud_b[np.abs(cloud_b[:, 2] - z90) <= top_tol_m]
    if len(top) < min_top_points:
        return None, {'reason': f'too few top-face points: {len(top)}, all={len(cloud_b)}'}
    centre = top.mean(axis=0)
    _, _, vt = np.linalg.svd(top - centre, full_matrices=False)
    n = vt[-1].copy()
    if n[2] < 0:
        n *= -1
    tilt = math.degrees(math.acos(float(np.clip(n[2], -1, 1))))
    if tilt > 20:
        return None, {'reason': f'top plane tilt implausible: {tilt:.1f} deg'}
    rect = cv2.minAreaRect(np.asarray(top[:, :2] * 1000, dtype=np.float32))
    corners_mm = cv2.boxPoints(rect)
    e1, e2 = corners_mm[1] - corners_mm[0], corners_mm[2] - corners_mm[1]
    long_edge = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    yaw = norm_yaw_pi(math.atan2(float(long_edge[1]), float(long_edge[0])))
    x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    x = x - n * np.dot(n, x)
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, n)
    top_center = np.array([rect[0][0] * 1e-3, rect[0][1] * 1e-3, np.median(top[:, 2])])
    T = np.eye(4)
    T[:3, :3] = np.column_stack((x, y, n))
    T[:3, 3] = top_center - n * (float(block_size[2]) / 2)
    footprint = sorted((float(rect[1][0]), float(rect[1][1])), reverse=True)
    expected = sorted((1000 * float(block_size[0]), 1000 * float(block_size[1])), reverse=True)
    return T, {'top_points': int(len(top)), 'color_depth_points': int(len(cloud_b)),
               'yaw_deg_canonical': math.degrees(yaw), 'top_normal_tilt_deg': tilt,
               'footprint_long_short_mm': footprint,
               'footprint_deficit_mm': [expected[0] - footprint[0], expected[1] - footprint[1]],
               'top_face_base_z_m': float(top_center[2])}


def estimate_pad_pose(depth_m, pad_mask, K, dist, B_from_C):
    """Pose of a flat placement pad from its visible outline.

    The pad is larger than a block, so the minimum-area rectangle of the visible
    ring still recovers the pad outline when a block sits inside it.
    """
    return estimate_pose_from_mask(depth_m, pad_mask, K, dist, B_from_C, block_size=PAD_SIZE,
                                   min_top_points=150, top_tol_m=0.004)


def detect_pads(bgr, depth_m, K, dist, B_from_C, expected: int = 3, min_area: int = 400):
    """Return a list of pad poses (robot base frame) sorted by base-frame y."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    full = hue_mask(hsv, PAD_SPEC['hue_ranges'])
    contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted([c for c in contours if cv2.contourArea(c) >= min_area],
                      key=cv2.contourArea, reverse=True)[:expected]
    poses = []
    for contour in contours:
        selected = np.zeros_like(full)
        cv2.drawContours(selected, [contour], -1, 255, thickness=cv2.FILLED)
        T, info = estimate_pad_pose(depth_m, cv2.bitwise_and(full, selected), K, dist, B_from_C)
        if T is not None:
            poses.append((T, info))
    return sorted(poses, key=lambda item: float(item[0][1, 3]))


def draw_axes(image_bgr, B_from_O, B_from_C, K, dist, axis_length: float = 0.055):
    C_from_O = np.linalg.inv(B_from_C) @ B_from_O
    if C_from_O[2, 3] <= 0.01:
        return image_bgr
    rvec, _ = cv2.Rodrigues(C_from_O[:3, :3].copy())
    cv2.drawFrameAxes(image_bgr, K, dist, rvec, C_from_O[:3, 3].copy(), axis_length, 2)
    return image_bgr
