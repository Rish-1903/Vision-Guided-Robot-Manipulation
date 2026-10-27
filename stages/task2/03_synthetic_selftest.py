#!/usr/bin/env python3
"""Task 2 diagnostic: can our calibration pipeline recover a KNOWN camera?

No simulator needed. Corners of the 9x6, 30 mm board are projected through a
known ideal pinhole camera (960x720, 60 deg FoV) using the same board-pose
generators as the capture script, corrupted with Gaussian pixel noise, and fed to
the SAME fit_camera() used by 02_calibrate_camera.py. Repeated over seeds, this
measures the bias and spread of the focal-length estimate caused purely by view
geometry and corner noise.

Interpretation: if the 'repo' profile already shows multi-percent focal error
here, weak view geometry alone explains the Task 2 mismatch and the fix is the
'wide' capture profile. If the synthetic error is small but real renders still
disagree, look at the renderer (texture scaling, FoV convention): see
04_render_checks.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
from make_checkerboard import INNER_CORNERS, SQUARE_METERS  # noqa: E402
from pose_profiles import CAMERA_Z, PROFILES  # noqa: E402

_spec = importlib.util.spec_from_file_location('calib02', HERE / '02_calibrate_camera.py')
calib02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calib02)

W, H = 960, 720
F_TRUE = (W / 2) / math.tan(math.radians(30))
K_TRUE = np.array([[F_TRUE, 0, W / 2], [0, F_TRUE, H / 2], [0, 0, 1.0]])
# Camera at (0, 0, CAMERA_Z) looking down world -Z (OpenCV axes).
R_CW = np.diag([1.0, -1.0, -1.0])
C_W = np.array([0.0, 0.0, CAMERA_Z])


def board_points_local():
    obj = calib02.board_object_points().astype(np.float64)
    centre = np.array([(INNER_CORNERS[0] - 1) * SQUARE_METERS / 2,
                       (INNER_CORNERS[1] - 1) * SQUARE_METERS / 2, 0.0])
    return obj, obj - centre


def synth_views(profile, n_views, noise_px, rng):
    obj, local = board_points_local()
    objs, imgs = [], []
    attempt = 0
    while len(objs) < n_views and attempt < 50 * n_views:
        pose = PROFILES[profile](rng, attempt)
        attempt += 1
        R = Rotation.from_quat(pose[3:]).as_matrix()
        pw = local @ R.T + np.asarray(pose[:3])
        pc = (pw - C_W) @ R_CW          # R_CW is symmetric and its own inverse
        if np.any(pc[:, 2] < 0.1):
            continue
        normal_c = R_CW @ R[:, 2]
        if normal_c[2] > -0.17:          # board must face the camera (< ~80 deg)
            continue
        uv = (pc[:, :2] / pc[:, 2:3]) * F_TRUE + np.array([W / 2, H / 2])
        if np.any(uv < 8) or np.any(uv[:, 0] > W - 8) or np.any(uv[:, 1] > H - 8):
            continue
        uv = uv + rng.normal(0, noise_px, uv.shape)
        objs.append(obj.astype(np.float32))
        imgs.append(uv.reshape(-1, 1, 2).astype(np.float32))
    return objs, imgs


def run(profile, n_views, noise_px, repeats, seed):
    out = {'full': [], 'pinhole': [], 'constrained': []}
    for r in range(repeats):
        rng = np.random.default_rng(seed + 1000 * r)
        objs, imgs = synth_views(profile, n_views, noise_px, rng)
        if len(objs) < 12:
            raise RuntimeError(f'{profile}: only {len(objs)} valid synthetic views')
        for key, kw in (('full', {}), ('pinhole', {'zero_distortion': True}),
                        ('constrained', {'constrained_pinhole': True})):
            _, K, _, _ = calib02.fit_camera(objs, imgs, (W, H), **kw)
            out[key].append(100 * (K[0, 0] - F_TRUE) / F_TRUE)
    return {k: {'mean_fx_error_pct': float(np.mean(v)), 'std_fx_error_pct': float(np.std(v)),
                'max_abs_fx_error_pct': float(np.max(np.abs(v)))} for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--views', type=int, default=24)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--noise-px', type=float, nargs='+', default=[0.1, 0.3])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, default=HERE / 'selftest_results.json')
    args = p.parse_args()
    results = {'true_fx_px': F_TRUE, 'views': args.views, 'repeats': args.repeats, 'rows': []}
    print(f'True fx = {F_TRUE:.2f} px; {args.views} views x {args.repeats} repeats\n')
    print(f"{'profile':<8}{'noise px':>9}  {'model':<12}{'mean err %':>11}{'std %':>8}{'max |err| %':>12}")
    for profile in PROFILES:
        for noise in args.noise_px:
            res = run(profile, args.views, noise, args.repeats, args.seed)
            for model, s in res.items():
                print(f"{profile:<8}{noise:>9.2f}  {model:<12}{s['mean_fx_error_pct']:>11.3f}"
                      f"{s['std_fx_error_pct']:>8.3f}{s['max_abs_fx_error_pct']:>12.3f}")
                results['rows'].append({'profile': profile, 'noise_px': noise, 'model': model, **s})
    args.output.write_text(json.dumps(results, indent=2))
    print('\nSaved', args.output)


if __name__ == '__main__':
    main()
