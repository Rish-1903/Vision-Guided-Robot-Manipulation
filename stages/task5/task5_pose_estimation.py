#!/usr/bin/env python3
"""Task 5: RGB-D pose estimation of the Task 4 blocks.

Inference uses RGB masks, metric depth, the Task 3 calibration (K, B_from_Ccv)
and the known block size. It never reads simulator object or camera poses;
ground truth is read after the whole estimation pass, for evaluation only.

Assumptions: rigid cuboids of known size lying upright on the table with the
top face visible. Yaw is ambiguous by 180 degrees, so rotation errors are
reported as the minimum over the symmetry group.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vgm.config import latest_calibration  # noqa: E402
from vgm.geometry import translation_rotation_errors  # noqa: E402
from vgm.metrics import describe, rate  # noqa: E402
from vgm.perception import (COLORS, detect_from_rgb, draw_axes,  # noqa: E402
                            estimate_pose_from_mask, load_calibration)
from vgm.scene import (apply_layout, connect, object_pose_B, prepare_scene,  # noqa: E402
                       read_bgr, read_depth_m)


def estimate_all(rgb, depth, K, dist, B_from_C, depth_mode='z'):
    detections, masks = detect_from_rgb(rgb)
    return detections, masks, {name: estimate_pose_from_mask(depth, masks[name], K, dist, B_from_C,
                                                            depth_mode=depth_mode)
                               for name in COLORS}


def score_poses(predicted, truth_B, trial, extra=None):
    rows = []
    for name in COLORS:
        pose_B, diag = predicted[name]
        row = {'trial': trial, 'color': name, 'detected': pose_B is not None,
               'reason': '' if pose_B is not None else diag['reason'],
               'translation_error_mm': None, 'raw_rotation_error_deg': None,
               'symmetry_rotation_error_deg': None,
               'yaw_deg_canonical': None if pose_B is None else diag['yaw_deg_canonical'],
               **(extra or {})}
        if pose_B is not None:
            t, raw, sym = translation_rotation_errors(pose_B, truth_B[name])
            row.update(translation_error_mm=t, raw_rotation_error_deg=raw, symmetry_rotation_error_deg=sym)
        rows.append(row)
    return rows


def summarize(rows):
    hits = [r for r in rows if r['detected']]
    out = {'pose_estimation': rate(len(hits), len(rows)),
           'translation_error_mm': describe(r['translation_error_mm'] for r in hits),
           'translation_rmse_mm': float(np.sqrt(np.mean([r['translation_error_mm'] ** 2 for r in hits])))
           if hits else None,
           'symmetry_rotation_error_deg': describe(r['symmetry_rotation_error_deg'] for r in hits),
           'per_color': {}}
    for color in COLORS:
        g = [r for r in rows if r['color'] == color]
        gh = [r for r in g if r['detected']]
        out['per_color'][color] = {
            'pose_estimation': rate(len(gh), len(g)),
            'translation_error_mm': describe(r['translation_error_mm'] for r in gh),
            'symmetry_rotation_error_deg': describe(r['symmetry_rotation_error_deg'] for r in gh)}
    return out


def plot_summary(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid = [x for x in rows if x['detected']]
    for field, title, filename in [
            ('translation_error_mm', '3D translation error by colour', 'translation_error_by_color.png'),
            ('symmetry_rotation_error_deg', 'Symmetry-aware rotation error by colour',
             'rotation_error_by_color.png')]:
        groups = [[float(x[field]) for x in valid if x['color'] == name] for name in COLORS]
        if not all(groups):
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(groups, labels=list(COLORS))
        ax.set_title(title)
        ax.set_ylabel('Millimetres' if field.endswith('_mm') else 'Degrees')
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def check_depth(depth):
    valid = depth[np.isfinite(depth)]
    if valid.size < 100 or np.ptp(valid) < 1e-5 or np.nanmedian(valid) < .05:
        raise RuntimeError('Depth looks empty/uniform. In the /Task2Camera properties, UNCHECK '
                           '"Ignore depth info (faster)" and rerun.')
    print(f'Depth sanity: min={valid.min():.4f} m, median={np.median(valid):.4f} m, max={valid.max():.4f} m')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', default='/Task2Camera')
    parser.add_argument('--robot-base', default='/Franka')
    parser.add_argument('--calibration', type=Path, default=None,
                        help='Task 3 .npz (default: latest calibration from this repo)')
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--depth-mode', choices=['z', 'range'], default='z',
                        help='z = optical-axis depth (default); range = ray length')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'task5_standalone')
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be at least 1')
    out = args.output.resolve()
    for sub in ('rgb', 'depth', 'overlays', 'masks'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    calibration = args.calibration or latest_calibration()
    K, dist, B_from_Ccv = load_calibration(calibration)
    print('Recovered intrinsics:\n', K, '\nRecovered camera-to-base transform:\n', B_from_Ccv)
    try:
        _, sim = connect()
        camera, base = sim.getObject(args.camera), sim.getObject(args.robot_base)
    except Exception as exc:
        sys.exit(f'Cannot access camera or robot base: {exc}')
    objects = prepare_scene(sim)
    rng = np.random.default_rng(args.seed)
    rows, detailed = [], []
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
                raise RuntimeError('Depth capture failed. UNCHECK "Ignore depth info (faster)" on '
                                   f'/Task2Camera. Original: {exc}') from exc
            if trial == 0:
                check_depth(depth)
            detections, masks, predicted = estimate_all(rgb, depth, K, dist, B_from_Ccv, args.depth_mode)
            # All predictions are complete before any ground truth is accessed.
            truth_B = {name: object_pose_B(sim, h, base) for name, h in objects.items()}
            trial_rows = score_poses(predicted, truth_B, trial)
            rows.extend(trial_rows)
            overlay = rgb.copy()
            for name in COLORS:
                pose_B, diag = predicted[name]
                detailed.append({**trial_rows[list(COLORS).index(name)],
                                 'B_from_O_est': None if pose_B is None else pose_B.tolist(),
                                 'B_from_O_gt_evaluation_only': truth_B[name].tolist(),
                                 'diagnostics': diag})
                if pose_B is not None:
                    draw_axes(overlay, pose_B, B_from_Ccv, K, dist)
                    u, v = (int(c) for c in detections[name]['centroid'])
                    cv2.putText(overlay, f'{name} {diag["yaw_deg_canonical"]:.0f}deg', (u - 45, v - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 2)
            cv2.imwrite(str(out / 'overlays' / f'pose_overlay_{trial:03d}.png'), overlay)
            if trial in (0, args.trials - 1):
                cv2.imwrite(str(out / 'rgb' / f'rgb_{trial:03d}.png'), rgb)
                lo, hi = np.nanmin(depth), np.nanmax(depth)
                vis = np.nan_to_num(np.clip((depth - lo) / max(1e-6, hi - lo) * 255, 0, 255)).astype(np.uint8)
                cv2.imwrite(str(out / 'depth' / f'depth_{trial:03d}.png'), cv2.applyColorMap(vis, cv2.COLORMAP_TURBO))
                for name in COLORS:
                    cv2.imwrite(str(out / 'masks' / f'{name}_{trial:03d}.png'), masks[name])
            print(f'Trial {trial + 1:02d}/{args.trials}: '
                  + ', '.join(f'{r["color"]} {r["translation_error_mm"]:.2f} mm / '
                              f'{r["symmetry_rotation_error_deg"]:.2f} deg' if r['detected']
                              else f'{r["color"]} FAIL ({r["reason"]})' for r in trial_rows))
    finally:
        sim.stopSimulation()
    with (out / 'pose_errors.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['trial', 'color', 'detected', 'reason', 'translation_error_mm',
                                               'raw_rotation_error_deg', 'symmetry_rotation_error_deg',
                                               'yaw_deg_canonical'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (out / 'pose_details.json').write_text(json.dumps(detailed, indent=2))
    summary = {'n_trials': args.trials, 'calibration_file': str(calibration), 'camera': args.camera,
               'robot_base': args.robot_base, 'depth_mode': args.depth_mode,
               'method': 'RGB-D cuboid top plane (SVD) + minimum-area footprint rectangle',
               **summarize(rows),
               'assumptions': ['fixed recovered extrinsics', 'known cuboid dimensions', 'visible top face',
                               'upright blocks on the table', '180-degree yaw symmetry'],
               'evaluation_ground_truth_policy': 'Block poses read only after the estimation pass.'}
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2))
    plot_summary(rows, out)
    print('\n=== TASK 5 SUMMARY ===\n' + json.dumps(summary, indent=2))
    print('Results:', out)


if __name__ == '__main__':
    main()
