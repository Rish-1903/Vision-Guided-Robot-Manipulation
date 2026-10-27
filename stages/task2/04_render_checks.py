#!/usr/bin/env python3
"""Task 2 diagnostic: measure the renderer's focal length directly.

DIAGNOSTIC ONLY. This script reads the camera and board poses from the
simulator to know the exact board distance; it never feeds calibration.

The board is placed fronto-parallel at several known distances. For a
fronto-parallel board the square pitch in pixels is p = f * s / Z, so
  fx = p_x * Z / s   and   fy = p_y * Z / s
are measured independently of Zhang's method. This separates three causes of
the Task 2 mismatch:
  * fx ~= fy ~= nominal (831 px)  -> renderer is fine; the checkerboard fit is at
    fault (corner detection, view selection, distortion terms).
  * fx ~= fy but != nominal       -> the FoV convention differs (e.g. the angle
    applies to the other image axis: 623 px) or the camera FoV was changed.
  * fx != fy (square aspect != 1)  -> texture/plane stretch in the render.
Optionally compare against Task 2 and Task 3 estimates with --task2 / --task3.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _corner_utils import detect_corners  # noqa: E402
from make_checkerboard import INNER_CORNERS, SQUARE_METERS  # noqa: E402


def read_bgr(sim, camera):
    raw, res = sim.getVisionSensorImg(camera)
    w, h = int(res[0]), int(res[1])
    img = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
    return cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_RGB2BGR)


def pitch_xy(corners):
    """Mean square pitch along the board's columns and rows, in pixels.

    Corners are robustly fitted with an affine map grid->image, whose columns give
    the pixel step per square along each board axis.
    """
    cols, rows = INNER_CORNERS
    uv = corners.reshape(-1, 2).astype(np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64)
    A = np.column_stack([grid, np.ones(len(grid))])
    coef, *_ = np.linalg.lstsq(A, uv, rcond=None)
    step_i, step_j = coef[0], coef[1]       # image displacement per board column / row
    residual = float(np.sqrt(np.mean(np.sum((A @ coef - uv) ** 2, axis=1))))
    return step_i, step_j, residual


def k_from(path, key_options):
    with np.load(path) as z:
        for k in key_options:
            if k in z:
                return np.asarray(z[k], float)
    raise KeyError(f'{path}: none of {key_options}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--camera', default='/Task2Camera')
    p.add_argument('--board', default='/Task2Checkerboard')
    p.add_argument('--distances', type=float, nargs='+', default=[0.55, 0.75, 0.93])
    p.add_argument('--yaws', type=float, nargs='+', default=[0.0, 90.0],
                   help='Board yaw (deg); 90 swaps which board axis lies along image x')
    p.add_argument('--task2', type=Path, default=None, help='camera_calibration.npz from Task 2')
    p.add_argument('--task3', type=Path, default=None, help='camera_calibration_recovered.npz from Task 3')
    p.add_argument('--output', type=Path, default=HERE / 'render_checks')
    args = p.parse_args()
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from scipy.spatial.transform import Rotation
    sim = RemoteAPIClient().require('sim')
    if sim.getSimulationState() != sim.simulation_stopped:
        raise SystemExit('Stop the simulation first.')
    camera, board = sim.getObject(args.camera), sim.getObject(args.board)
    cam_pose = sim.getObjectPose(camera, -1)
    cam_z = float(cam_pose[2])
    res = sim.getVisionSensorRes(camera)
    W, H = int(res[0]), int(res[1])
    try:
        angle = float(sim.getObjectFloatParam(camera, sim.visionfloatparam_perspective_angle))
    except Exception:
        angle = math.radians(60.0)
    f_long = (max(W, H) / 2) / math.tan(angle / 2)
    f_short = (min(W, H) / 2) / math.tan(angle / 2)
    original_board_pose = sim.getObjectPose(board, -1)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    sim.setStepping(True)
    sim.startSimulation()
    try:
        for Z in args.distances:
            for yaw in args.yaws:
                q = Rotation.from_euler('z', yaw, degrees=True).as_quat().tolist()
                sim.setObjectPose(board, [float(cam_pose[0]), float(cam_pose[1]), cam_z - Z, *q], -1)
                sim.step()
                sim.step()
                img = read_bgr(sim, camera)
                corners = detect_corners(img)
                if corners is None:
                    print(f'Z={Z:.2f} yaw={yaw:.0f}: board not detected (too close or out of frame?)')
                    continue
                si, sj, resid = pitch_xy(corners)
                # Pixel pitch along image x and image y, whichever board axis lies there.
                steps = np.vstack([si, sj])
                px = float(np.max(np.abs(steps[:, 0])))
                py = float(np.max(np.abs(steps[:, 1])))
                Zb = float(cam_z - sim.getObjectPose(board, -1)[2])   # board-plane distance
                rows.append({'distance_m': Zb, 'board_yaw_deg': yaw, 'pitch_x_px': px, 'pitch_y_px': py,
                             'fx_measured_px': px * Zb / SQUARE_METERS,
                             'fy_measured_px': py * Zb / SQUARE_METERS,
                             'square_aspect_px_x_over_y': px / py, 'affine_fit_residual_px': resid})
                cv2.imwrite(str(args.output / f'frontal_Z{Z:.2f}_yaw{yaw:.0f}.png'), img)
                r = rows[-1]
                print(f"Z={Zb:.3f} yaw={yaw:>3.0f}: fx={r['fx_measured_px']:.1f} fy={r['fy_measured_px']:.1f} "
                      f"aspect={r['square_aspect_px_x_over_y']:.4f} (affine residual {resid:.3f} px)")
    finally:
        sim.stopSimulation()
        sim.setObjectPose(board, original_board_pose, -1)
    if not rows:
        raise SystemExit('No fronto-parallel view detected.')
    fx = float(np.mean([r['fx_measured_px'] for r in rows]))
    fy = float(np.mean([r['fy_measured_px'] for r in rows]))
    report = {'resolution': [W, H], 'perspective_angle_deg': math.degrees(angle),
              'f_if_angle_on_long_axis_px': f_long, 'f_if_angle_on_short_axis_px': f_short,
              'fx_measured_mean_px': fx, 'fy_measured_mean_px': fy, 'rows': rows}
    for name, path, keys in (('task2', args.task2, ('K', 'K_constrained')),
                             ('task3', args.task3, ('K',))):
        if path:
            K = k_from(path, keys)
            report[f'{name}_fx_px'] = float(K[0, 0])
            report[f'{name}_fy_px'] = float(K[1, 1])
    aspect = fx / fy
    if abs(aspect - 1) > 0.01:
        verdict = f'Non-square pixels / stretched texture in render (fx/fy = {aspect:.4f}).'
    elif abs(fx - f_long) / f_long < 0.01:
        verdict = 'Renderer matches the nominal long-axis FoV model; the Task 2 fit itself is at fault.'
    elif abs(fx - f_short) / f_short < 0.01:
        verdict = 'The FoV angle applies to the SHORT image axis; fix the nominal reference.'
    else:
        verdict = 'Renderer focal length matches neither convention; check the camera FoV setting.'
    report['verdict'] = verdict
    (args.output / 'render_checks.json').write_text(json.dumps(report, indent=2))
    print(f'\nMeasured fx={fx:.1f} px, fy={fy:.1f} px | nominal long-axis {f_long:.1f}, short-axis {f_short:.1f}')
    for k in ('task2_fx_px', 'task3_fx_px'):
        if k in report:
            print(f'{k}: {report[k]:.1f}')
    print('VERDICT:', verdict)


if __name__ == '__main__':
    main()
