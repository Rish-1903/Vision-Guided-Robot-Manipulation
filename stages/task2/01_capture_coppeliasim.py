"""Task 2 (step 1): synthesize a checkerboard workcell and capture images.

Instructions:
  - Open CoppeliaSim with a NEW EMPTY scene and the ZMQ Remote API add-on enabled.
  - Do NOT start the simulation yourself; the script starts/stops it in stepping mode.
  - Run: python 01_capture_coppeliasim.py --views 30 [--profile wide|repo]

The code estimates NO intrinsic parameters and does not read the camera's ground-truth
pose or intrinsics for calibration. The known nominal field of view is saved only for
later VALIDATION of the estimated matrix.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from _corner_utils import detect_corners
from pose_profiles import PROFILES
from make_checkerboard import (
    INNER_CORNERS,
    SQUARE_METERS,
    BORDER_SQUARES,
    generate,
)

ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = (960, 720)  # width, height
CAMERA_FOV_DEG = 60.0
CAMERA_POSITION = [0.0, 0.0, 1.0]


def read_camera_bgr(sim, camera_handle: int) -> np.ndarray:
    raw, resolution = sim.getVisionSensorImg(camera_handle)
    w, h = (int(resolution[0]), int(resolution[1]))
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != w * h * 3:
        raise RuntimeError(f'Unexpected vision buffer: {arr.size} bytes for {w}x{h}.')
    rgb_bottom_up = arr.reshape(h, w, 3)
    # CoppeliaSim delivers bottom-up RGB; OpenCV expects top-down BGR.
    return cv2.cvtColor(cv2.flip(rgb_bottom_up, 0), cv2.COLOR_RGB2BGR)


def create_workcell(sim, texture_file: Path):
    w, h = IMAGE_SIZE
    floats = [
        0.01, 5.0, math.radians(CAMERA_FOV_DEG), 0.01,
        0.0, 0.0, 0.45, 0.45, 0.45, 0.0, 0.0,
    ]
    # perspective, hide the view frustum, no fog, fixed grey background
    try:
        camera = sim.getObject('/Task2Camera')
        print('Using existing /Task2Camera; its pose and FoV will NOT be changed.')
    except Exception:
        camera = sim.createVisionSensor(2 | 4 | 64 | 128, [w, h, 0, 0], floats)
        sim.setObjectAlias(camera, 'Task2Camera')
        # CoppeliaSim cameras look along local +Z.
        sim.setObjectPose(camera, [*CAMERA_POSITION, 1.0, 0.0, 0.0, 0.0])
        print('Created /Task2Camera at its default pose.')

    num_squares_x = INNER_CORNERS[0] + 1 + 2 * BORDER_SQUARES
    num_squares_y = INNER_CORNERS[1] + 1 + 2 * BORDER_SQUARES
    board_width = num_squares_x * SQUARE_METERS
    board_height = num_squares_y * SQUARE_METERS
    # Texture-plane import is built into the simulator: avoids manual UV mapping.
    try:
        board = sim.getObject('/Task2Checkerboard')
        print('Using existing /Task2Checkerboard.')
    except Exception:
        board, texture_id, resolution = sim.createTexture(
            str(texture_file.resolve()), 1 | 2, [board_width, board_height]
        )
        sim.setObjectAlias(board, 'Task2Checkerboard')
    sim.setObjectPose(board, [0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 1.0])
    print(f'Camera: {w} x {h}, nominal FoV: {CAMERA_FOV_DEG} degrees')
    print(f'Checkerboard plane: {board_width:.3f} x {board_height:.3f} m')
    return camera, board


def pose_for_attempt(rng: np.random.Generator, attempt: int, profile: str = 'wide') -> list[float]:
    """Board pose for one capture attempt; see pose_profiles.py for the profiles."""
    return PROFILES[profile](rng, attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--views', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--profile', choices=list(PROFILES), default='wide',
                        help="'wide' (default): 25-55 deg tilts, 0.45-0.95 m depth. "
                             "'repo': the original <=27 deg, ~0.9 m profile")
    parser.add_argument('--output', type=Path, default=ROOT / 'dataset')
    args = parser.parse_args()
    if args.views < 15:
        raise SystemExit('Request at least 15 accepted images; 25-40 is recommended.')

    dest = args.output.resolve()
    images = dest / 'images'
    detected = dest / 'detected_corners'
    images.mkdir(parents=True, exist_ok=True)
    detected.mkdir(parents=True, exist_ok=True)
    texture = ROOT / 'checkerboard.png'
    generate(texture)

    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client = RemoteAPIClient()
        sim = client.require('sim')
    except Exception as exc:
        raise SystemExit(
            'Could not connect to CoppeliaSim. Open CoppeliaSim with the ZMQ Remote API '
            f'add-on running. Original error: {exc}'
        ) from exc

    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError('Stop CoppeliaSim simulation before capturing Task 2 views.')
    # Hide any existing colored blocks/landmark rig during checkerboard captures.
    for name in ['/T4_Red', '/T4_Green', '/T4_Blue', '/Task3CalibrationRig']:
        try:
            h = sim.getObject(name)
            sim.setObjectPose(h, [2., 2., 0.1, 0., 0., 0., 1.], -1)
        except Exception:
            pass
    camera, board = create_workcell(sim, texture)
    # Synchronous stepping avoids capturing the previous frame after repositioning.
    sim.setStepping(True)
    sim.startSimulation()
    rng = np.random.default_rng(args.seed)
    saved = []
    try:
        for attempt in range(max(100, args.views * 6)):
            if len(saved) >= args.views:
                break
            pose = pose_for_attempt(rng, attempt, args.profile)
            sim.setObjectPose(board, pose)
            sim.step()
            sim.step()
            frame = read_camera_bgr(sim, camera)
            corners = detect_corners(frame)
            if corners is None:
                print(f'Attempt {attempt+1:03d}: checkerboard not detected; changing pose.')
                continue
            idx = len(saved)
            image_path = images / f'view_{idx:03d}.png'
            cv2.imwrite(str(image_path), frame)
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, INNER_CORNERS, corners, True)
            cv2.imwrite(str(detected / image_path.name), vis)
            saved.append({'file': image_path.name, 'camera_to_board': 'not used',
                          'simulation_board_pose_for_data_generation_only': pose})
            print(f'Captured {len(saved):02d}/{args.views} from attempt {attempt+1}')
    finally:
        sim.stopSimulation()

    manifest = {
        'description': 'CoppeliaSim perspective camera; intrinsics must be estimated from images',
        'resolution': list(IMAGE_SIZE),
        'nominal_field_of_view_deg_for_validation_only': CAMERA_FOV_DEG,
        'internal_corners': list(INNER_CORNERS),
        'square_size_m': SQUARE_METERS,
        'pose_profile': args.profile,
        'board_generation_poses_not_for_calibration': saved,
    }
    with (dest / 'manifest.json').open('w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nCaptured {len(saved)} images in {images}')
    if len(saved) < 15:
        raise SystemExit('Not enough accepted checkerboard views. Check image visibility and lighting.')
    print('Next: python 02_calibrate_camera.py --dataset dataset')
    print('Save the CoppeliaSim scene (*.ttt) if you want to keep the camera/workcell.')


if __name__ == '__main__':
    main()
