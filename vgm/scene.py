"""CoppeliaSim I/O and scene staging.

Functions that READ object poses (``object_pose_B``) are for scene staging and
evaluation only; they are never called by perception or planning code.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import obj_pose_matrix
from .perception import (BLOCK_SIZE, BLOCK_Z, COLORS, PAD_SIZE, PAD_SPEC, PAD_Z,
                         TABLE_SIZE, TABLE_Z)

PARKED_POSE = [2.0, 2.0, 0.1, 0.0, 0.0, 0.0, 1.0]


def connect():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client = RemoteAPIClient()
    return client, client.require('sim')


def get_maybe(sim, alias):
    try:
        return sim.getObject(alias)
    except Exception:
        return None


def read_bgr(sim, camera) -> np.ndarray:
    raw, resolution = sim.getVisionSensorImg(camera)
    w, h = int(resolution[0]), int(resolution[1])
    buf = np.frombuffer(raw, dtype=np.uint8)
    if buf.size != w * h * 3:
        raise ValueError(f'Expected RGB {w}x{h} image, got {buf.size} bytes')
    # CoppeliaSim's image origin is bottom-left; OpenCV's is top-left.
    return cv2.cvtColor(np.flipud(buf.reshape(h, w, 3)).copy(), cv2.COLOR_RGB2BGR)


def read_depth_m(sim, camera, image_hw) -> np.ndarray:
    """Metric depth (option bit 0 = 1), flipped to OpenCV image order."""
    buf, resolution = sim.getVisionSensorDepth(camera, 1)
    width, height = map(int, resolution)
    depth = np.frombuffer(buf, dtype='<f4')
    if depth.size != width * height:
        raise ValueError(f'Expected {width}x{height} metric depth, got {depth.size} values')
    depth = np.flipud(depth.reshape(height, width)).copy()
    if depth.shape != tuple(image_hw):
        raise ValueError(f'RGB and depth resolution mismatch: {image_hw} vs {depth.shape}')
    return depth


def object_pose_B(sim, handle, base) -> np.ndarray:
    """Simulator ground truth. EVALUATION / STAGING ONLY."""
    return obj_pose_matrix(sim.getObjectPose(handle, base))


def W_from_base(sim, base) -> np.ndarray:
    """Known workcell geometry: where the robot base is bolted in the world."""
    return obj_pose_matrix(sim.getObjectPose(base, -1))


def world_to_base(sim, base, xyz_world) -> np.ndarray:
    p = np.asarray([*xyz_world[:3], 1.0], dtype=float)
    return (np.linalg.inv(W_from_base(sim, base)) @ p)[:3]


def world_yaw_to_base(sim, base, yaw_world_rad: float) -> float:
    R = W_from_base(sim, base)[:3, :3]
    base_yaw_in_world = float(np.arctan2(R[1, 0], R[0, 0]))
    return yaw_world_rad - base_yaw_in_world


def park(sim, alias):
    handle = get_maybe(sim, alias)
    if handle is not None:
        sim.setObjectPose(handle, PARKED_POSE, -1)


def move_old_targets_outside_view(sim):
    # The Task 3 rig and the Task 2 board would otherwise cause false detections.
    for alias in ('/Task3CalibrationRig', '/Task2Checkerboard'):
        park(sim, alias)


def make_block(sim, alias, size, color, emission: float = 0.30):
    handle = get_maybe(sim, alias)
    if handle is None:
        handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, list(map(float, size)), 2)
        sim.setObjectAlias(handle, alias.lstrip('/'))
    sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)
    sim.setShapeColor(handle, '', sim.colorcomponent_ambient_diffuse, list(color))
    sim.setShapeColor(handle, '', sim.colorcomponent_emission, [float(v) * emission for v in color])
    return handle


def prepare_scene(sim) -> dict:
    move_old_targets_outside_view(sim)
    table = make_block(sim, '/T4_Table', TABLE_SIZE, (.25, .25, .25))
    sim.setObjectPose(table, [0., 0., TABLE_Z, 0., 0., 0., 1.], -1)
    return {name: make_block(sim, info['alias'], BLOCK_SIZE, info['rgb'])
            for name, info in COLORS.items()}


def set_block_world(sim, handle, x, y, yaw_deg, z=BLOCK_Z):
    q = Rotation.from_euler('z', yaw_deg, degrees=True).as_quat().tolist()
    sim.setObjectPose(handle, [float(x), float(y), float(z), *q], -1)


def place_pads(sim, destinations: dict) -> dict:
    """Create/position flat yellow pads at the destinations (scene staging)."""
    pads = {}
    for color, dest in destinations.items():
        alias = f'/T6_Pad_{color}'
        handle = make_block(sim, alias, PAD_SIZE, PAD_SPEC['rgb'], emission=0.25)
        yaw = float(dest[3]) if len(dest) > 3 else 0.0
        set_block_world(sim, handle, dest[0], dest[1], yaw, z=PAD_Z)
        pads[color] = handle
    return pads


def remove_pads(sim):
    for color in COLORS:
        park(sim, f'/T6_Pad_{color}')


def random_layout(rng, n: int = 3, minimum_separation: float = 0.17,
                  x_range=(-0.215, 0.215), y_range=(-0.155, 0.155), accept=None):
    """Sample ``n`` well-separated XY positions; ``accept(xy)`` adds constraints."""
    positions = []
    for _ in range(n):
        for _try in range(2000):
            xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)], dtype=float)
            if accept is not None and not accept(xy):
                continue
            if all(np.linalg.norm(xy - p) > minimum_separation for p in positions):
                positions.append(xy)
                break
        else:
            raise RuntimeError('Could not sample distinct object locations; relax the constraints.')
    return positions


def apply_layout(sim, objects, rng, trial_idx):
    """Task 4/5 dataset layouts. Trial 0 is a fixed, easy layout."""
    if trial_idx == 0:
        poses = [(-.19, -.11, 0.), (.19, -.10, 40.), (0., .135, -35.)]
    else:
        poses = [(p[0], p[1], float(rng.uniform(-85, 85))) for p in random_layout(rng)]
    for (name, handle), (x, y, yaw) in zip(objects.items(), poses):
        set_block_world(sim, handle, x, y, yaw)
    return poses
