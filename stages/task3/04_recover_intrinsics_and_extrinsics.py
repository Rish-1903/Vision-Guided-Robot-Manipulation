#!/usr/bin/env python3
"""Recover camera intrinsics and fixed camera-to-robot pose from Task 3 images.

Only uses 3D coordinates of the eight calibration landmarks, target poses known
relative to the robot base, and the measured 2D image landmark coordinates.
The simulator ground truth in the JSON is read ONLY AFTER fitting, for validation.

Assumes an undistorted pinhole camera, square pixels, centered principal point.
For a physical camera these constraints must be re-evaluated. Use the image
resolution to fix the principal point, not a simulator-provided camera pose/FOV.

Usage:
  python 04_recover_intrinsics_and_extrinsics.py \\
    --results task3_results/camera_robot_calibration.json \\
    --output task3_results/recovered
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Exactly the same landmark coordinates as 03_calibrate_camera_robot.py (metres).
LANDMARKS = {
    'red': (-.145, -.085, .028),
    'orange': (.135, -.083, .042),
    'yellow': (.125, .090, .038),
    'green': (-.140, .087, .034),
    'cyan': (-.050, -.035, .125),
    'blue': (.048, .044, .115),
    'violet': (-.052, .054, .072),
    'magenta': (.085, -.035, .077),
}
SIM_CAMERA_FROM_CV = np.eye(4)
SIM_CAMERA_FROM_CV[:3, :3] = np.diag([-1., -1., 1.])


def from_pose(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def observations(records):
    """3D landmarks in base coords + pixel detections for each target pose."""
    all_X, all_uv = [], []
    for record in records:
        B_from_Q = from_pose(record['target_pose_in_robot_base'])
        P3, uv = [], []
        for name, xyz in LANDMARKS.items():
            if name not in record['detections']:
                continue
            P3.append(B_from_Q[:3, :3] @ xyz + B_from_Q[:3, 3])
            uv.append(record['detections'][name])
        if len(P3) < 6:
            raise ValueError('Encountered a record with fewer than 6 landmarks')
        all_X.append(np.array(P3, np.float64))
        all_uv.append(np.array(uv, np.float64))
    return all_X, all_uv


def parameter_to_camera(params, width, height):
    f = float(np.exp(params[0]))
    K = np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1.]], dtype=float)
    R = Rotation.from_rotvec(params[1:4]).as_matrix()
    t = params[4:7]
    C_from_B = np.eye(4)
    C_from_B[:3, :3] = R
    C_from_B[:3, 3] = t
    return K, C_from_B


def reprojection_residual(params, X, uv, width, height):
    K, C_from_B = parameter_to_camera(params, width, height)
    cam_xyz = X @ C_from_B[:3, :3].T + C_from_B[:3, 3]
    z = cam_xyz[:, 2]
    z_safe = np.where(np.abs(z) < 1e-5, np.where(z >= 0, 1e-5, -1e-5), z)
    predicted = cam_xyz[:, :2] / z_safe[:, None] * K[0, 0] + K[:2, 2]
    return (predicted - uv).ravel()


def candidate_initializations(X, uv, width, height):
    """Image-only PnP initializations over plausible, unconstrained focal guesses.

    Focal guesses are numeric starting values, NOT the scene's FOV or true K.
    """
    inits = []
    for f in (350., 550., 750., 950., 1250., 1700., 2400., 3400., 5000.):
        K0 = np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1.]], dtype=float)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                np.ascontiguousarray(X), np.ascontiguousarray(uv),
                K0, None, flags=cv2.SOLVEPNP_ITERATIVE
            )
        except cv2.error:
            continue
        if not ok:
            continue
        p0 = np.r_[np.log(f), rvec.reshape(3), tvec.reshape(3)]
        depth = (X @ Rotation.from_rotvec(p0[1:4]).as_matrix().T + p0[4:])[..., 2]
        if np.mean(depth > 0) < .95:
            continue
        err = np.sqrt(np.mean(reprojection_residual(p0, X, uv, width, height)**2))
        inits.append((float(err), p0))
    if not inits:
        raise RuntimeError('No positive-depth initial estimate from image correspondences')
    return sorted(inits, key=lambda item: item[0])


def fit_fixed_camera(X, uv, width, height):
    initial = candidate_initializations(X, uv, width, height)
    best = None
    for _, p0 in initial[:4]:
        res = least_squares(
            reprojection_residual, p0, args=(X, uv, width, height),
            loss='soft_l1', f_scale=1.5, method='trf', x_scale='jac',
            max_nfev=1400,
            bounds=(np.array([np.log(100.), *([-np.inf]*6)]),
                    np.array([np.log(20000.), *([np.inf]*6)]))
        )
        d = reprojection_residual(res.x, X, uv, width, height)
        score = float(np.sqrt(np.mean(d**2)))
        if best is None or score < best[0]:
            best = (score, res)
    if best is None:
        raise RuntimeError('Optimization failed')
    return best[1]


def point_rms(params, X, uv, w, h):
    d = reprojection_residual(params, X, uv, w, h).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d*d, axis=1))))


def rotation_error_deg(A, B):
    return float(np.rad2deg((Rotation.from_matrix(A).inv()*Rotation.from_matrix(B)).magnitude()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--seed', type=int, default=2403)
    args = parser.parse_args()
    with args.results.open() as handle:
        previous = json.load(handle)
    records = previous['records']
    if len(records) < 12:
        sys.exit('Need at least 12 accepted target views')
    X_sets, uv_sets = observations(records)
    shuffled = np.random.default_rng(args.seed).permutation(len(records))
    n_val = max(3, round(len(records) * .25))
    validation = shuffled[:n_val]
    training = shuffled[n_val:]
    Xtr = np.concatenate([X_sets[i] for i in training])
    UVtr = np.concatenate([uv_sets[i] for i in training])
    result = fit_fixed_camera(Xtr, UVtr, args.width, args.height)
    K, C_from_B = parameter_to_camera(result.x, args.width, args.height)
    B_from_Ccv = np.linalg.inv(C_from_B)
    B_from_Csim = B_from_Ccv @ SIM_CAMERA_FROM_CV
    train_rms = point_rms(result.x, Xtr, UVtr, args.width, args.height)
    val_rms = [point_rms(result.x, X_sets[i], uv_sets[i], args.width, args.height)
               for i in validation]
    print('=== Recovered intrinsics from 3D-2D correspondences ===')
    print(K)
    print(f'Training landmark projection RMS: {train_rms:.3f} px')
    print(f'Validation landmark RMS by held-out view: {np.round(val_rms, 3)} px')
    print(f'Validation mean RMS: {np.mean(val_rms):.3f} px')
    print('=== Recovered robot_base_T_OpenCV_camera ===')
    print(B_from_Ccv)
    truth = previous.get('simulator_ground_truth_USED_ONLY_FOR_VALIDATION', {})
    true_B_Csim = truth.get('B_from_Csim_ground_truth')
    validation_info = {}
    if true_B_Csim is not None:
        true_B_Csim = np.array(true_B_Csim, dtype=float)
        trans_err = 1000*np.linalg.norm(B_from_Csim[:3, 3]-true_B_Csim[:3, 3])
        rot_err = rotation_error_deg(B_from_Csim[:3, :3], true_B_Csim[:3, :3])
        validation_info = {'translation_error_mm': float(trans_err),
                           'rotation_error_deg': rot_err}
        print('=== GROUND TRUTH VALIDATION ONLY ===')
        print(f'Translation error: {trans_err:.3f} mm')
        print(f'Rotation error: {rot_err:.4f} degrees')
    else:
        print('No simulator truth in file; report projection consistency only.')
    print(f'Optimization status: {result.message}')
    out = args.output or args.results.parent/'recovered'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'camera_calibration_recovered.npz', K=K,
             K_pinhole=K, dist=np.zeros((5, 1)), dist_pinhole=np.zeros((5, 1)),
             width=args.width, height=args.height)
    np.savez(out/'camera_to_robot_transform_recovered.npz',
             B_from_Ccv=B_from_Ccv, B_from_Csim=B_from_Csim,
             K=K, dist=np.zeros((5, 1)))
    report = {
        'method': 'joint 3D-2D camera intrinsic and eye-to-hand extrinsic calibration',
        'data_sources_for_estimation': 'known target landmark geometry, known target poses in robot base, 2D observed landmark centers',
        'camera_model_assumptions': 'zero distortion, fx=fy, cx=w/2, cy=h/2',
        'K': K.tolist(),
        'B_from_Ccv': B_from_Ccv.tolist(),
        'B_from_Csim': B_from_Csim.tolist(),
        'training_view_ids': [int(i) for i in training],
        'validation_view_ids': [int(i) for i in validation],
        'train_projection_rms_px': train_rms,
        'validation_projection_rms_px_each': val_rms,
        'validation_projection_mean_rms_px': float(np.mean(val_rms)),
        'simulator_ground_truth_validation_only': validation_info,
        'note': 'Target motion was confined to small area; assess uncertainty/observability when using outside this workcell.'
    }
    (out/'recovery_results.json').write_text(json.dumps(report, indent=2))
    print('Outputs saved to:', out)

if __name__ == '__main__':
    main()
