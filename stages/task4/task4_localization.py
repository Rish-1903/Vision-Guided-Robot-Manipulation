#!/usr/bin/env python3
"""Task 4: RGB-only object localization in the CoppeliaSim workcell.

Detects the three coloured blocks with HSV segmentation and contours using only
rendered RGB pixels. Simulator object poses are read after detection, for
evaluation only. 6D pose estimation is Task 5.
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
from vgm.metrics import describe, rate  # noqa: E402
from vgm.perception import (BLOCK_SIZE, COLORS, box_iou, detect_from_rgb,  # noqa: E402
                            load_calibration, overlay_image, project_box)
from vgm.scene import apply_layout, connect, object_pose_B, prepare_scene, read_bgr  # noqa: E402


def project_object_truth(sim, handle, base, K, dist, B_from_Ccv):
    """EVALUATION ONLY: true block pose projected with the ESTIMATED calibration."""
    return project_box(object_pose_B(sim, handle, base), BLOCK_SIZE, K, dist, B_from_Ccv)


def score_frame(detections, ground_truth, trial, extra=None):
    rows = []
    for name in COLORS:
        det, gt = detections[name], ground_truth[name]
        error = iou = None
        if det is not None and gt is not None:
            error = float(np.linalg.norm(np.asarray(det['centroid']) - np.asarray(gt['top_center_uv'])))
            iou = box_iou(det['bbox_xyxy'], gt['bbox_xyxy'])
        rows.append({'trial': trial, 'color': name, 'detected': det is not None,
                     'n_blobs': None if det is None else det['n_blobs'],
                     'centroid_u': None if det is None else det['centroid'][0],
                     'centroid_v': None if det is None else det['centroid'][1],
                     'gt_top_center_u': None if gt is None else gt['top_center_uv'][0],
                     'gt_top_center_v': None if gt is None else gt['top_center_uv'][1],
                     'center_error_px': error, 'bbox_iou': iou, **(extra or {})})
    return rows


def summarize(records):
    hits = [r for r in records if r['center_error_px'] is not None]
    out = {'detection': rate(sum(r['detected'] for r in records), len(records)),
           'center_error_px': describe(r['center_error_px'] for r in hits),
           'bbox_iou': describe(r['bbox_iou'] for r in hits), 'per_color': {}}
    for name in COLORS:
        group = [r for r in records if r['color'] == name]
        g_hits = [r for r in group if r['center_error_px'] is not None]
        out['per_color'][name] = {
            'detection': rate(sum(r['detected'] for r in group), len(group)),
            'center_error_px': describe(r['center_error_px'] for r in g_hits),
            'bbox_iou': describe(r['bbox_iou'] for r in g_hits)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', default='/Task2Camera')
    parser.add_argument('--robot-base', default='/Franka')
    parser.add_argument('--calibration', type=Path, default=None,
                        help='Task 3 .npz (default: latest calibration from this repo)')
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'task4_standalone')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    calibration = args.calibration or latest_calibration()
    K, dist, B_from_Ccv = load_calibration(calibration)
    out = args.output.resolve()
    for sub in ('images', 'overlays', 'masks'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    print('Using image-estimated camera matrix:\n', K)
    try:
        _, sim = connect()
        camera, base = sim.getObject(args.camera), sim.getObject(args.robot_base)
    except Exception as exc:
        sys.exit(f'Cannot find {args.camera} and {args.robot_base} in the open scene: {exc}')
    objects = prepare_scene(sim)
    rng = np.random.default_rng(args.seed)
    records = []
    sim.setStepping(True)
    sim.startSimulation()
    try:
        for trial in range(args.trials):
            commanded = apply_layout(sim, objects, rng, trial)
            sim.step()
            sim.step()
            frame = read_bgr(sim, camera)
            detections, masks = detect_from_rgb(frame)          # RGB pixels only
            ground_truth = {name: project_object_truth(sim, h, base, K, dist, B_from_Ccv)
                            for name, h in objects.items()}     # evaluation only
            cv2.imwrite(str(out / 'images' / f'frame_{trial:03d}.png'), frame)
            cv2.imwrite(str(out / 'overlays' / f'overlay_{trial:03d}.png'),
                        overlay_image(frame, detections, ground_truth))
            if trial == 0:
                for name, m in masks.items():
                    cv2.imwrite(str(out / 'masks' / f'{name}_trial000.png'), m)
            rows = score_frame(detections, ground_truth, trial)
            for row, pose in zip(rows, commanded):
                row['commanded_pose_for_dataset_generation_only'] = list(map(float, pose))
            records.extend(rows)
            print(f'Trial {trial + 1:02d}/{args.trials}: {sum(r["detected"] for r in rows)}/3 detected')
    finally:
        sim.stopSimulation()

    (out / 'detections.json').write_text(json.dumps(records, indent=2))
    with (out / 'detections.csv').open('w', newline='') as f:
        fields = ['trial', 'color', 'detected', 'n_blobs', 'centroid_u', 'centroid_v',
                  'gt_top_center_u', 'gt_top_center_v', 'center_error_px', 'bbox_iou']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    summary = {'camera': args.camera, 'robot_base': args.robot_base,
               'method': 'RGB-only HSV segmentation + largest connected contour',
               'calibration_file': str(Path(calibration).resolve()), 'n_trials': args.trials,
               **summarize(records),
               'ground_truth_policy': 'Simulator object poses read only after detection, for evaluation.',
               'metric_note': 'Centre error compares the mask centroid with the projected top-face '
                              'centre; visible side faces bias the centroid slightly.'}
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = [[r['center_error_px'] for r in records if r['color'] == c and r['center_error_px'] is not None]
              for c in COLORS]
    if all(groups):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(groups, labels=list(COLORS))
        ax.set_ylabel('Centroid vs projected top centre (px)')
        ax.set_title('Task 4: localization error by colour')
        fig.tight_layout()
        fig.savefig(out / 'centroid_error_by_color.png', dpi=160)
        plt.close(fig)
    d = summary['detection']
    print(f"\nDetection rate: {d['successes']}/{d['n']} = {100 * d['rate']:.1f}% "
          f"(95% CI {100 * d['ci95_low']:.1f}-{100 * d['ci95_high']:.1f}%)")
    print('Centre error (px):', summary['center_error_px'])
    print('Results:', out)


if __name__ == '__main__':
    main()
