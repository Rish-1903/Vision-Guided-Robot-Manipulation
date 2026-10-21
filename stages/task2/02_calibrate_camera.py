"""Task 2 (step 2): estimate camera intrinsics from rendered checkerboard images.

The simulator's field-of-view metadata is consulted ONLY AFTER calibration, for
validation of our image-estimated K. Camera pose and board scene coordinates are
NEVER used for the estimates.

Usage: python 02_calibrate_camera.py --dataset dataset
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from make_checkerboard import INNER_CORNERS, SQUARE_METERS
from _corner_utils import detect_corners  # shared helper for consistent image detection

ROOT = Path(__file__).resolve().parent


def board_object_points() -> np.ndarray:
    cols, rows = INNER_CORNERS
    coords = np.zeros((cols * rows, 3), dtype=np.float32)
    coords[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_METERS
    return coords


def projection_rms(obj, observed, rvec, tvec, K, dist):
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    residual = projected.reshape(-1, 2) - observed.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))


def fit_camera(object_points, image_points, size, zero_distortion=False,
               constrained_pinhole=False):
    flags = 0
    K_init = None
    dist_init = None
    if zero_distortion or constrained_pinhole:
        flags |= (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 |
                  cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                  cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    if constrained_pinhole:
        # Initial focal guess depends ONLY on the observed resolution, not
        # the simulator FoV or ground-truth intrinsics. Focal length is fitted.
        f0 = float(max(size))
        K_init = np.array([[f0,0.,size[0]/2],
                           [0.,f0,size[1]/2], [0.,0.,1.]], float)
        dist_init = np.zeros((5,1),float)
        flags |= (cv2.CALIB_USE_INTRINSIC_GUESS |
                  cv2.CALIB_FIX_PRINCIPAL_POINT |
                  cv2.CALIB_FIX_ASPECT_RATIO)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, size, K_init, dist_init, flags=flags
    )
    per_view_rms = [
        projection_rms(a, b, r, t, K, dist)
        for a, b, r, t in zip(object_points, image_points, rvecs, tvecs)
    ]
    return float(rms), K, dist, per_view_rms


def holdout_corner_error(obj, detected_corners, K, dist):
    """Estimate board pose from half its corners; test on never-used corners.

    This does not train on held-out images, and prediction errors are computed on
    corners omitted from the per-image PnP estimation.
    """
    img = detected_corners.reshape(-1, 2)
    n = len(obj)
    inds = np.arange(n)
    train_mask = (inds % 2 == 0)
    test_mask = ~train_mask
    ok, rvec, tvec = cv2.solvePnP(
        obj[train_mask], img[train_mask], K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise ValueError('PnP failed on hold-out image')
    return projection_rms(obj[test_mask], img[test_mask], rvec, tvec, K, dist)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = dataset / 'calibration_results'
    output.mkdir(parents=True, exist_ok=True)
    with (dataset / 'manifest.json').open() as f:
        meta = json.load(f)
    size = tuple(meta['resolution'])
    cols, rows = meta['internal_corners']
    square = float(meta['square_size_m'])
    if (cols, rows) != INNER_CORNERS or square != SQUARE_METERS:
        raise ValueError('Mismatch between manifest and checkerboard constants')
    object_template = board_object_points()
    object_points = []
    image_points = []
    accepted_files = []
    for image_path in sorted((dataset / 'images').glob('*.png')):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'WARNING: Could not read {image_path}')
            continue
        if (image.shape[1], image.shape[0]) != size:
            print(f'WARNING: Image size mismatch in {image_path}')
            continue
        corners = detect_corners(image)
        if corners is not None:
            object_points.append(object_template.copy())
            image_points.append(corners)
            accepted_files.append(image_path.name)
    n = len(accepted_files)
    if n < 15:
        raise SystemExit(f'Only {n} usable views. Capture at least 15 diverse views.')
    print(f'Found {n} valid checkerboard views')

    rng = np.random.default_rng(args.seed)
    inds = rng.permutation(n)
    n_val = max(3, int(round(n * .20)))
    val_idx = inds[:n_val]
    train_idx = inds[n_val:]
    objects_train = [object_points[i] for i in train_idx]
    corners_train = [image_points[i] for i in train_idx]

    rms, K, dist, per_train = fit_camera(objects_train, corners_train, size)
    rms_pinhole, K_pinhole, dist_pinhole, per_pinhole = fit_camera(
        objects_train, corners_train, size, zero_distortion=True
    )
    rms_constrained, K_constrained, dist_constrained, per_constrained = fit_camera(
        objects_train, corners_train, size, constrained_pinhole=True
    )
    val_errors = [holdout_corner_error(object_points[i], image_points[i], K, dist)
                  for i in val_idx]
    val_errors_pinhole = [
        holdout_corner_error(object_points[i], image_points[i], K_pinhole, dist_pinhole)
        for i in val_idx
    ]
    val_errors_constrained = [
        holdout_corner_error(object_points[i], image_points[i], K_constrained, dist_constrained)
        for i in val_idx
    ]
    fov = float(meta['nominal_field_of_view_deg_for_validation_only'])
    # CoppeliaSim assigns the field-of-view angle to the longest image dimension.
    f_expected = max(size) / (2.0 * math.tan(math.radians(fov) / 2.0))
    nominal_K = np.array([[f_expected, 0., size[0] / 2.],
                          [0., f_expected, size[1] / 2.],
                          [0., 0., 1.]])
    summary = {
        'calibration_image_count': len(train_idx),
        'heldout_image_count': len(val_idx),
        'calibration_files': [accepted_files[i] for i in train_idx],
        'heldout_files': [accepted_files[i] for i in val_idx],
        'checkerboard_inner_corners': list(INNER_CORNERS),
        'square_size_m': SQUARE_METERS,
        'image_size_px': list(size),
        'unconstrained_distortion_model': {
            'K_estimated_from_images': K.tolist(),
            'distortion_coefficients': dist.reshape(-1).tolist(),
            'train_opencv_rms_px': rms,
            'train_per_view_reprojection_rms_px': per_train,
            'heldout_corner_rms_px_per_image': val_errors,
            'heldout_mean_corner_rms_px': float(np.mean(val_errors)),
        },
        'ideal_pinhole_model': {
            'K_estimated_from_images': K_pinhole.tolist(),
            'distortion_coefficients': dist_pinhole.reshape(-1).tolist(),
            'train_opencv_rms_px': rms_pinhole,
            'train_per_view_reprojection_rms_px': per_pinhole,
            'heldout_corner_rms_px_per_image': val_errors_pinhole,
            'heldout_mean_corner_rms_px': float(np.mean(val_errors_pinhole)),
        },
        'centered_square_pixel_zero_distortion_model': {
            'K_estimated_from_checkerboard_images': K_constrained.tolist(),
            'assumptions': 'fx=fy, cx=w/2, cy=h/2, zero distortion; focal length fitted from images',
            'train_opencv_rms_px': rms_constrained,
            'train_per_view_reprojection_rms_px': per_constrained,
            'heldout_mean_corner_rms_px': float(np.mean(val_errors_constrained)),
        },
        'simulator_reference_for_validation_only': {
            'nominal_fov_deg': fov,
            'K_ideal_reference': nominal_K.tolist(),
            'full_model_fx_error_percent': float(100 * abs(K[0, 0] - f_expected) / f_expected),
            'full_model_fy_error_percent': float(100 * abs(K[1, 1] - f_expected) / f_expected),
            'pinhole_fx_error_percent': float(100 * abs(K_pinhole[0, 0] - f_expected) / f_expected),
            'pinhole_fy_error_percent': float(100 * abs(K_pinhole[1, 1] - f_expected) / f_expected),
            'constrained_fx_error_percent': float(100 * abs(K_constrained[0, 0] - f_expected) / f_expected),
        },
    }
    with (output / 'calibration_summary.json').open('w') as f:
        json.dump(summary, f, indent=2)
    np.savez(output / 'camera_calibration.npz', K=K, dist=dist,
             K_pinhole=K_pinhole, dist_pinhole=dist_pinhole,
             K_constrained=K_constrained, dist_constrained=dist_constrained,
             K_sim_reference_for_validation_only=nominal_K,
             width=size[0], height=size[1])

    plt.figure(figsize=(9, 4.5))
    xs = np.arange(len(train_idx))
    plt.plot(xs, per_train, marker='o', label='5-coefficient distortion model')
    plt.plot(xs, per_pinhole, marker='s', label='zero-distortion pinhole model')
    plt.plot(xs, per_constrained, marker='^', label='centered square-pixel pinhole')
    plt.xlabel('Calibration image index')
    plt.ylabel('Reprojection RMS error (pixels)')
    plt.title('Per-image training reprojection error')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / 'training_reprojection_errors.png', dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    x = np.arange(n_val)
    plt.bar(x - 0.25, val_errors, width=.25, label='Distortion model')
    plt.bar(x, val_errors_pinhole, width=.25, label='Pinhole model')
    plt.bar(x + .25, val_errors_constrained, width=.25, label='Constrained pinhole')
    plt.xlabel('Held-out image index')
    plt.ylabel('Withheld-corner reprojection RMS (pixels)')
    plt.title('Validation on unseen views and withheld corners')
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / 'validation_corner_errors.png', dpi=180)
    plt.close()

    print('\n=== Estimated camera intrinsics (unconstrained lens model) ===')
    print(K)
    print('Distortion coefficients:', dist.flatten())
    print(f'Training RMS: {rms:.4f} px')
    print(f'Held-out withheld-corner mean RMS: {np.mean(val_errors):.4f} px')
    print('\n=== Zero-distortion pinhole calibration ===')
    print(K_pinhole)
    print(f'Training RMS: {rms_pinhole:.4f} px')
    print(f'Held-out withheld-corner mean RMS: {np.mean(val_errors_pinhole):.4f} px')
    print('\n=== Constrained checkerboard pinhole calibration (fitted image-only) ===')
    print(K_constrained)
    print(f'Training RMS: {rms_constrained:.4f} px')
    print(f'Held-out withheld-corner mean RMS: {np.mean(val_errors_constrained):.4f} px')
    print('\n=== Simulator reference (VALIDATION ONLY, not used to estimate K) ===')
    print(nominal_K)
    print(f'\nResults: {output}')
    print('Do not report any values as experimental results until you have run this on actual CoppeliaSim frames.')


if __name__ == '__main__':
    main()
