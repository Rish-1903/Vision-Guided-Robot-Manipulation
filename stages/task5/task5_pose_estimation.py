#!/usr/bin/env python3
"""Task 5 — vision-only RGB-D pose estimation for Task 4 CoppeliaSim blocks.

Run beside vision_calibration_task3/ and with saved scene containing /Franka,
/Task2Camera and optionally the three /T4_* blocks.

INFERENCE: RGB masks + metric depth + Task3 calibrated K and B_from_Ccv +
known block geometry; NEVER uses sim.getObjectPose(camera or objects).
GROUND TRUTH: sim.getObjectPose(block, robot) is read ONLY AFTER inference.

Physical assumptions: rigid cuboids with known dimensions and top surface
visible, lying upright on a horizontal tabletop. Relative 180-degree yaw is
unobservable for a plain symmetric box; report symmetry-aware orientation error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from task4_common import (BLOCK_SIZE, COLORS, load_calibration, read_bgr,
                          detect_from_rgb, obj_pose_matrix, prepare_scene,
                          apply_layout, connect)

HERE = Path(__file__).resolve().parent


def read_depth_m(sim, camera, image_hw):
    """CoppeliaSim metric depth (option bit 0 = 1), image origin flipped."""
    buf, resolution = sim.getVisionSensorDepth(camera, 1)
    width, height = map(int, resolution)
    depth = np.frombuffer(buf, dtype='<f4')
    if depth.size != width * height:
        raise ValueError(f'Expected {width}x{height} metric depth, got {depth.size} values')
    depth = np.flipud(depth.reshape(height, width)).copy()
    if depth.shape != tuple(image_hw):
        raise ValueError(f'RGB and depth resolution mismatch: {image_hw} vs {depth.shape}')
    return depth


def backproject_pixels(u, v, z, K, dist, depth_mode='z'):
    """Map masked depth pixels to camera coordinates in meters.

    depth_mode='z': depth is camera-axis depth (standard perspective Z).
    depth_mode='range': depth measures Euclidean distance along each ray.
    """
    pixels = np.column_stack((u, v)).astype(np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(pixels, K, dist).reshape(-1, 2)
    xy = rays
    if depth_mode == 'range':
        z = z / np.sqrt(1.0 + (xy**2).sum(axis=1))
    return np.column_stack((xy[:, 0]*z, xy[:, 1]*z, z))


def norm_yaw_pi(angle):
    """Canonical yaw in [-pi/2, pi/2); cuboid yaw is ambiguous modulo pi."""
    return (float(angle) + np.pi/2) % np.pi - np.pi/2


def estimate_pose_from_mask(depth_m, mask, K, dist, B_from_C, block_size=BLOCK_SIZE,
                            min_top_points=100, depth_mode='z', top_tol_m=0.0025):
    """Estimate B_from_O without simulator object poses.

    1. Erode RGB color mask and backproject valid depth pixels.
    2. Transform points using measured camera-to-base transform.
    3. Isolate visible top surface using its top-most height quantile.
    4. Fit top-plane normal with SVD; fit footprint via minAreaRect.
    5. Recover block center by offsetting half the known block height.

    Returns (4x4 B_from_O, diagnostics) or (None, diagnostic reason).
    """
    if mask is None or depth_m is None:
        return None, {'reason': 'no RGB mask or depth'}
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.erode(mask, kernel, iterations=1)
    valid = (clean > 0) & np.isfinite(depth_m) & (depth_m > 0.08) & (depth_m < 5.0)
    vv, uu = np.nonzero(valid)
    if len(uu) < min_top_points:
        return None, {'reason': f'too few colored depth pixels: {len(uu)}'}
    z = depth_m[vv, uu].astype(np.float64)
    cloud_c = backproject_pixels(uu, vv, z, K, dist, depth_mode=depth_mode)
    cloud_b = cloud_c @ B_from_C[:3, :3].T + B_from_C[:3, 3]
    # Top face is near the highest base-Z. Discard side faces by height.
    z90 = float(np.percentile(cloud_b[:, 2], 90))
    select = (cloud_b[:, 2] >= z90 - top_tol_m) & (cloud_b[:, 2] <= z90 + top_tol_m)
    top = cloud_b[select]
    if len(top) < min_top_points:
        return None, {'reason': f'too few top-face points: {len(top)}, all={len(cloud_b)}'}
    # Deprojected top plane: normal from SVD; normal upward in robot base frame.
    center_p = np.mean(top, axis=0)
    _, singular_vals, vt = np.linalg.svd(top - center_p, full_matrices=False)
    n = vt[-1].copy()
    if n[2] < 0:
        n *= -1
    tilt = math.degrees(math.acos(float(np.clip(n[2], -1, 1))))
    if tilt > 20:
        return None, {'reason': f'top plane tilt implausible: {tilt:.1f} deg'}
    # Rotated rectangle in base-XY; longer side of known 82x60mm box is x-axis.
    xy_mm = np.asarray(top[:, :2] * 1000, dtype=np.float32)
    rect = cv2.minAreaRect(xy_mm)
    corners_mm = cv2.boxPoints(rect)
    edge = corners_mm[1] - corners_mm[0]
    edge2 = corners_mm[2] - corners_mm[1]
    long_edge = edge if np.linalg.norm(edge) >= np.linalg.norm(edge2) else edge2
    yaw = norm_yaw_pi(math.atan2(float(long_edge[1]), float(long_edge[0])))
    x = np.array([math.cos(yaw), math.sin(yaw), 0.], dtype=float)
    # Build full 3D orientation with the fitted top normal.
    x = x - n * np.dot(n, x)
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, n)
    R = np.column_stack((x, y, n))
    # Top face center comes from footprint, not RGB silhouette center.
    top_center = np.array([rect[0][0]*0.001, rect[0][1]*0.001,
                           np.median(top[:, 2])], dtype=float)
    xyz = top_center - n * (float(block_size[2]) / 2)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    fitted_wh_mm = sorted((float(rect[1][0]), float(rect[1][1])), reverse=True)
    info = {'top_points': int(len(top)), 'color_depth_points': int(len(cloud_b)),
            'yaw_deg_canonical': math.degrees(yaw),
            'top_normal_tilt_deg': tilt,
            'footprint_long_short_mm': fitted_wh_mm,
            'top_face_base_z_m': float(top_center[2])}
    return T, info


def translation_rotation_errors(estimated, truth):
    p_err_mm = 1000. * float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]))
    raw_R_err = math.degrees(float(Rotation.from_matrix(estimated[:3, :3].T @ truth[:3, :3]).magnitude()))
    flip = Rotation.from_euler('z', math.pi).as_matrix()
    R_flip = truth[:3, :3] @ flip
    flip_err = math.degrees(float(Rotation.from_matrix(estimated[:3, :3].T @ R_flip).magnitude()))
    # Cuboid has a 180-degree symmetry about its local vertical axis.
    return p_err_mm, raw_R_err, min(raw_R_err, flip_err)


def draw_axes(image_bgr, B_from_O, B_from_C, K, dist, axis_length=0.055):
    C_from_O = np.linalg.inv(B_from_C) @ B_from_O
    R = C_from_O[:3, :3].copy()
    t = C_from_O[:3, 3].copy()
    if t[2] <= 0.01:
        return image_bgr
    rvec, _ = cv2.Rodrigues(R)
    cv2.drawFrameAxes(image_bgr, K, dist, rvec, t, axis_length, 2)
    return image_bgr


def plot_summary(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid = [x for x in rows if x['detected']]
    if not valid:
        return
    for field, title, filename in [
        ('translation_error_mm', '3D translation error by color', 'translation_error_by_color.png'),
        ('symmetry_rotation_error_deg', 'Symmetry-aware rotation error by color', 'rotation_error_by_color.png'),
    ]:
        groups = [[float(x[field]) for x in valid if x['color'] == name] for name in COLORS]
        if not all(groups):
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(groups, labels=list(COLORS))
        ax.set_title(title)
        ax.set_ylabel('Millimetres' if field.endswith('_mm') else 'Degrees')
        fig.tight_layout()
        fig.savefig(output/filename, dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', default='/Task2Camera')
    parser.add_argument('--robot-base', default='/Franka')
    parser.add_argument('--calibration', type=Path,
                        default=HERE.parent/'vision_calibration_task3'/'task3_results'/
                        'recovered'/'camera_to_robot_transform_recovered.npz')
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--depth-mode', choices=['z', 'range'], default='z',
                        help='z=optical-axis distance (default); range=ray distance')
    parser.add_argument('--output', type=Path, default=HERE/'task5_results')
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be at least 1')
    out = args.output.resolve()
    for sub in ['rgb', 'depth', 'overlays', 'masks']:
        (out/sub).mkdir(parents=True, exist_ok=True)
    K, dist, B_from_Ccv = load_calibration(args.calibration)
    print('Using recovered image-estimated intrinsics:\n', K)
    print('Using recovered camera-to-Franka-base transform:\n', B_from_Ccv)
    try:
        sim = connect()
        camera = sim.getObject(args.camera)
        base = sim.getObject(args.robot_base)
    except Exception as exc:
        sys.exit(f'Cannot access camera or robot base: {exc}')
    objects = prepare_scene(sim)
    rng = np.random.default_rng(args.seed)
    rows = []
    detailed = []
    last_predictions = {}
    sim.setStepping(True)
    sim.startSimulation()
    try:
        for trial in range(args.trials):
            apply_layout(sim, objects, rng, trial)
            sim.step()
            sim.step()
            rgb = read_bgr(sim, camera)
            try:
                depth = read_depth_m(sim, camera, rgb.shape[:2])
            except Exception as exc:
                raise RuntimeError('Depth capture failed. In /Task2Camera sensor properties, '
                                   'UNCHECK "Ignore depth info (faster)". Original: '+str(exc)) from exc
            if trial == 0:
                valid = depth[np.isfinite(depth)]
                if valid.size < 100 or np.ptp(valid) < 1e-5 or np.nanmedian(valid) < .05:
                    raise RuntimeError('Depth seems empty/uniform. UNCHECK the /Task2Camera sensor '
                                       'option "Ignore depth info (faster)"; then rerun.')
                print(f'Depth sanity: min={valid.min():.4f}m, '
                      f'median={np.median(valid):.4f}m, max={valid.max():.4f}m')
            detections, masks = detect_from_rgb(rgb)
            overlay = rgb.copy()
            predicted = {}
            for name, obj in objects.items():
                pose_B, diagnostic = estimate_pose_from_mask(depth, masks[name], K, dist,
                                                               B_from_Ccv, depth_mode=args.depth_mode)
                predicted[name] = (pose_B, diagnostic)
                if pose_B is not None:
                    draw_axes(overlay, pose_B, B_from_Ccv, K, dist)
                    xy = np.asarray(detections[name]['centroid'], dtype=int)
                    cv2.putText(overlay, f'{name} {diagnostic["yaw_deg_canonical"]:.0f}deg',
                                (int(xy[0])-45, int(xy[1])-30),
                                cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 2)
                else:
                    print(f'Trial {trial:02d} {name}: pose not available: {diagnostic["reason"]}')
            # IMPORTANT: ALL predictions above are complete before any GT pose is accessed.
            ground_truth_B = {name: obj_pose_matrix(sim.getObjectPose(handle, base))
                              for name, handle in objects.items()}
            count = 0
            for name in COLORS:
                pose_B, diagnostic = predicted[name]
                gt_B = ground_truth_B[name]
                if pose_B is None:
                    row = {'trial': trial, 'color': name, 'detected': False,
                           'reason': diagnostic['reason'], 'translation_error_mm': '',
                           'raw_rotation_error_deg': '', 'symmetry_rotation_error_deg': '',
                           'yaw_deg_canonical': ''}
                else:
                    count += 1
                    t_err, raw_rot, sym_rot = translation_rotation_errors(pose_B, gt_B)
                    row = {'trial': trial, 'color': name, 'detected': True,
                           'reason': '', 'translation_error_mm': t_err,
                           'raw_rotation_error_deg': raw_rot,
                           'symmetry_rotation_error_deg': sym_rot,
                           'yaw_deg_canonical': diagnostic['yaw_deg_canonical']}
                rows.append(row)
                detailed.append({**row,
                                 'B_from_O_est': None if pose_B is None else pose_B.tolist(),
                                 'B_from_O_gt_evaluation_only': gt_B.tolist(),
                                 'diagnostics': diagnostic})
            if trial == 0 or trial == args.trials-1:
                cv2.imwrite(str(out/'rgb'/f'rgb_{trial:03d}.png'), rgb)
                dep_vis = np.clip((depth - np.nanmin(depth)) /
                                  max(1e-6, np.nanmax(depth) - np.nanmin(depth))*255, 0, 255)
                dep_vis = np.nan_to_num(dep_vis, nan=0).astype(np.uint8)
                cv2.imwrite(str(out/'depth'/f'depth_{trial:03d}.png'), cv2.applyColorMap(dep_vis, cv2.COLORMAP_TURBO))
                for name in COLORS:
                    cv2.imwrite(str(out/'masks'/f'{name}_{trial:03d}.png'), masks[name])
            cv2.imwrite(str(out/'overlays'/f'pose_overlay_{trial:03d}.png'), overlay)
            if trial == args.trials-1:
                last_predictions = {name: (predicted[name][0].tolist() if predicted[name][0] is not None else None)
                                    for name in COLORS}
            print(f'Trial {trial+1:02d}/{args.trials}: {count}/3 poses | '
                  + ', '.join(f'{row["color"]}: {row["translation_error_mm"]:.2f} mm / '
                              f'{row["symmetry_rotation_error_deg"]:.2f} deg'
                              for row in rows[-3:] if row['detected']))
    finally:
        sim.stopSimulation()
    with (out/'pose_errors.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['trial', 'color', 'detected', 'reason',
                             'translation_error_mm', 'raw_rotation_error_deg',
                             'symmetry_rotation_error_deg', 'yaw_deg_canonical'])
        writer.writeheader()
        writer.writerows(rows)
    (out/'pose_details.json').write_text(json.dumps(detailed, indent=2))
    (out/'latest_estimated_poses.json').write_text(json.dumps({
        'frame': 'Franka base', 'method': 'RGB-D cuboid top plane + geometric rectangle fitting',
        'pose_matrices_B_from_O': last_predictions,
        'warning': 'Pose is physically constrained by upright cuboids on a table; '
                   '180-degree yaw ambiguity remains.'}, indent=2))
    hits = [r for r in rows if r['detected']]
    summary = {
        'n_trials': args.trials, 'n_expected_poses': len(rows), 'n_estimated_poses': len(hits),
        'pose_estimation_rate': len(hits)/len(rows), 'calibration_file': str(args.calibration),
        'camera': args.camera, 'robot_base': args.robot_base, 'depth_mode': args.depth_mode,
        'mean_translation_error_mm': float(np.mean([r['translation_error_mm'] for r in hits])) if hits else None,
        'median_translation_error_mm': float(np.median([r['translation_error_mm'] for r in hits])) if hits else None,
        'translation_rmse_mm': float(np.sqrt(np.mean([r['translation_error_mm']**2 for r in hits]))) if hits else None,
        'mean_symmetry_rotation_error_deg': float(np.mean([r['symmetry_rotation_error_deg'] for r in hits])) if hits else None,
        'per_color': {},
        'assumptions': ['fixed recovered extrinsics', 'known cuboid dimensions',
                        'visible top face', 'upright blocks on tabletop',
                        '180-degree yaw symmetry; report minimum rotation error over symmetry group'],
        'evaluation_ground_truth_policy': 'simulation block poses only read after the entire RGB-D estimation pass',
    }
    for color in COLORS:
        group = [r for r in hits if r['color']==color]
        summary['per_color'][color] = {
            'success': len(group), 'trials': args.trials,
            'mean_translation_error_mm': float(np.mean([r['translation_error_mm'] for r in group])) if group else None,
            'mean_symmetry_rotation_error_deg': float(np.mean([r['symmetry_rotation_error_deg'] for r in group])) if group else None,
        }
    (out/'metrics.json').write_text(json.dumps(summary, indent=2))
    plot_summary(rows, out)
    print('\n=== TASK 5 SUMMARY ===')
    print(json.dumps(summary, indent=2))
    print('Results directory:', out)
    print('Pose-estimated B_from_O matrices saved to latest_estimated_poses.json for Task 6.')
    print('Save the updated CoppeliaSim scene for Task 6.')


if __name__ == '__main__':
    main()
