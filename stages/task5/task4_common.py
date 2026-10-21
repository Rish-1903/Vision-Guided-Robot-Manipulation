#!/usr/bin/env python3
"""Task 4 — RGB-only object localization in a CoppeliaSim workcell.

Creates/reuses three static colored blocks inside the field of view of
/Task2Camera; detects boxes and centroids using *only* rendered RGB images.
Simulation object poses are read separately *after detection* for evaluation.

This stage does NOT attempt 6D pose estimation. That belongs to Task 5.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent

# RGB shape material colors. Rendered RGB hue can vary with scene lighting.
COLORS = {
    'red': {'rgb': (1.0, 0.06, 0.06), 'alias': '/T4_Red',
            'hue_ranges': [(0, 12), (170, 179)]},
    'green': {'rgb': (0.04, 1.0, 0.06), 'alias': '/T4_Green',
              'hue_ranges': [(40, 86)]},
    'blue': {'rgb': (0.06, 0.12, 1.0), 'alias': '/T4_Blue',
             'hue_ranges': [(95, 139)]},
}
BLOCK_SIZE = np.array([0.082, 0.060, 0.038], dtype=float)  # metres
TABLE_SIZE = [0.70, 0.58, 0.014]
TABLE_Z = 0.025
BLOCK_Z = TABLE_Z + TABLE_SIZE[2]/2 + BLOCK_SIZE[2]/2 + .001


def load_calibration(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing calibrated camera file: {path}\n'
            'Expected Task 3 recovery output: '
            'task3_results/recovered/camera_to_robot_transform_recovered.npz')
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


def read_bgr(sim, camera):
    raw, resolution = sim.getVisionSensorImg(camera)
    w, h = (int(resolution[0]), int(resolution[1]))
    buf = np.frombuffer(raw, dtype=np.uint8)
    if buf.size != w*h*3:
        raise ValueError(f'Expected RGB {w}x{h} image, got {buf.size} bytes')
    # CoppeliaSim's RGB image origin is bottom-left; OpenCV origin top-left.
    return cv2.cvtColor(np.flipud(buf.reshape(h, w, 3)).copy(), cv2.COLOR_RGB2BGR)


def detect_from_rgb(bgr: np.ndarray, min_area: int = 120):
    """Returns {(color): candidate} without reading any simulation state."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    detections, masks = {}, {}
    kernel = np.ones((3, 3), np.uint8)
    for name, spec in COLORS.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for low, high in spec['hue_ranges']:
            mask |= cv2.inRange(hsv, (low, 80, 45), (high, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masks[name] = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not contours:
            detections[name] = None
            continue
        # One instance of each color is intentionally present in this experiment.
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if abs(moments['m00']) < 1e-10:
            detections[name] = None
            continue
        x, y, w, h = cv2.boundingRect(contour)
        angle = cv2.minAreaRect(contour)[-1]  # only image orientation, not 6D pose
        detections[name] = {
            'centroid': [float(moments['m10']/moments['m00']),
                         float(moments['m01']/moments['m00'])],
            'bbox_xyxy': [int(x), int(y), int(x+w-1), int(y+h-1)],
            'area_pixels': float(cv2.contourArea(contour)),
            'image_rect_angle_deg': float(angle),
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
        cv2.rectangle(vis, tuple(box[:2]), tuple(box[2:]), palette[name], 2)
        cv2.circle(vis, (u, v), 4, (255, 255, 255), -1)
        cv2.putText(vis, name, (box[0], max(20, box[1]-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, palette[name], 2, cv2.LINE_AA)
    if ground_truth is not None:
        for name, gt in ground_truth.items():
            if gt is None:
                continue
            x1, y1, x2, y2 = gt['bbox_xyxy']
            cv2.rectangle(vis, (round(x1), round(y1)), (round(x2), round(y2)),
                          (245, 245, 245), 1)
            u, v = gt['top_center_uv']
            cv2.drawMarker(vis, (round(u), round(v)), (255, 255, 255),
                           cv2.MARKER_CROSS, 11, 1)
    return vis


def obj_pose_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.array(pose[3:], dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(pose[:3], float)
    return T


def local_corners(size):
    half = np.array(size, dtype=float)/2
    return np.asarray([[a*half[0], b*half[1], c*half[2]]
                       for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)


def project_object_truth(sim, handle, base, size, K, dist, B_from_Ccv):
    """FOR EVALUATION ONLY: simulator object pose -> calibrated pixel projection.

    Uses the recovered camera calibration, NOT CoppeliaSim's true camera pose.
    Must not be called by detect_from_rgb().
    """
    B_from_O = obj_pose_matrix(sim.getObjectPose(handle, base))
    C_from_B = np.linalg.inv(B_from_Ccv)
    verts_B = local_corners(size) @ B_from_O[:3, :3].T + B_from_O[:3, 3]
    top_B = (np.array([0.0, 0.0, size[2]/2]) @ B_from_O[:3, :3].T + B_from_O[:3, 3])
    xyz_B = np.concatenate([verts_B, top_B[None, :]], axis=0)
    xyz_C = xyz_B @ C_from_B[:3, :3].T + C_from_B[:3, 3]
    if np.any(xyz_C[:, 2] <= 0):
        return None
    uv, _ = cv2.projectPoints(np.ascontiguousarray(xyz_C), np.zeros(3),
                              np.zeros(3), K, dist)
    uv = uv.reshape(-1, 2)
    box_xyxy = [float(uv[:8, 0].min()), float(uv[:8, 1].min()),
                float(uv[:8, 0].max()), float(uv[:8, 1].max())]
    return {'bbox_xyxy': box_xyxy,
            'top_center_uv': uv[8].tolist()}


def box_iou(box_a, box_b):
    xa, ya, xb, yb = map(float, box_a)
    xc, yc, xd, yd = map(float, box_b)
    xx1, yy1 = max(xa, xc), max(ya, yc)
    xx2, yy2 = min(xb, xd), min(yb, yd)
    inter = max(0., xx2-xx1) * max(0., yy2-yy1)
    area1, area2 = max(0., xb-xa)*max(0., yb-ya), max(0., xd-xc)*max(0., yd-yc)
    return float(inter/(area1+area2-inter)) if area1+area2>inter else 0.


def connect():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client = RemoteAPIClient()
    sim = client.require('sim')
    return sim


def get_maybe(sim, alias):
    try:
        return sim.getObject(alias)
    except Exception:
        return None


def move_old_targets_outside_view(sim):
    # The Task 3 rig contains red/green/blue markers which would otherwise
    # create false detections. Moving parent moves all its child landmarks.
    for alias in ('/Task3CalibrationRig', '/Task2Checkerboard'):
        handle = get_maybe(sim, alias)
        if handle is not None:
            sim.setObjectPose(handle, [2.0, 2.0, 0.1, 0., 0., 0., 1.], -1)


def make_block(sim, alias, size, color):
    handle = get_maybe(sim, alias)
    if handle is None:
        handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, list(map(float, size)), 2)
        sim.setObjectAlias(handle, alias.lstrip('/'))
    sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)
    sim.setShapeColor(handle, '', sim.colorcomponent_ambient_diffuse, list(color))
    sim.setShapeColor(handle, '', sim.colorcomponent_emission,
                      [float(v)*0.30 for v in color])
    return handle


def prepare_scene(sim):
    move_old_targets_outside_view(sim)
    table = make_block(sim, '/T4_Table', TABLE_SIZE, (.25, .25, .25))
    sim.setObjectPose(table, [0., 0., TABLE_Z, 0., 0., 0., 1.], -1)
    objects = {}
    for name, info in COLORS.items():
        objects[name] = make_block(sim, info['alias'], BLOCK_SIZE, info['rgb'])
    return objects


def random_layout(rng, minimum_separation=0.17):
    positions = []
    for _ in range(3):
        for _try in range(300):
            x = float(rng.uniform(-0.215, 0.215))
            y = float(rng.uniform(-0.155, 0.155))
            if all(np.linalg.norm(np.array([x, y])-p) > minimum_separation
                   for p in positions):
                positions.append(np.array([x, y]))
                break
        else:
            raise RuntimeError('Could not sample distinct object locations')
    return positions


def apply_layout(sim, objects, rng, trial_idx):
    if trial_idx == 0:
        poses = [(-.19, -.11, 0.), (.19, -.10, 40.), (0., .135, -35.)]
    else:
        positions = random_layout(rng)
        poses = [(p[0], p[1], float(rng.uniform(-85, 85))) for p in positions]
    for (name, handle), (x, y, yaw) in zip(objects.items(), poses):
        quat = Rotation.from_euler('z', yaw, degrees=True).as_quat().tolist()
        sim.setObjectPose(handle, [float(x), float(y), float(BLOCK_Z), *quat], -1)
    return poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', default='/Task2Camera')
    parser.add_argument('--robot-base', default='/Franka')
    parser.add_argument('--calibration', type=Path, default=HERE.parent/'vision_calibration_task3'/
                        'task3_results'/'recovered'/'camera_to_robot_transform_recovered.npz')
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--output', type=Path, default=HERE/'task4_results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    K, dist, B_from_Ccv = load_calibration(args.calibration)
    out = args.output.resolve()
    (out/'images').mkdir(parents=True, exist_ok=True)
    (out/'overlays').mkdir(parents=True, exist_ok=True)
    (out/'masks').mkdir(parents=True, exist_ok=True)
    print('Using saved image-calibrated camera (not camera simulator GT):\n', K)
    try:
        sim = connect()
        camera = sim.getObject(args.camera)
        base = sim.getObject(args.robot_base)
    except Exception as exc:
        sys.exit(f'Cannot find {args.camera} and {args.robot_base} in the open scene: {exc}')
    objects = prepare_scene(sim)
    rng = np.random.default_rng(args.seed)
    records = []
    sim.setStepping(True)
    sim.startSimulation()
    try:
        for trial in range(args.trials):
            commanded_poses = apply_layout(sim, objects, rng, trial)
            sim.step()
            sim.step()
            frame = read_bgr(sim, camera)
            if abs(K[0, 2]-frame.shape[1]/2) > .1*frame.shape[1] or abs(K[1, 2]-frame.shape[0]/2) > .1*frame.shape[0]:
                print('WARNING: principal point is far from image center: verify calibration resolution')
            # This is the entire detection method: consumes ONLY RGB pixels.
            detections, masks = detect_from_rgb(frame)
            # Now, and only now, read 3D simulator object poses for EVALUATION.
            ground_truth = {
                name: project_object_truth(sim, handle, base, BLOCK_SIZE,
                                           K, dist, B_from_Ccv)
                for name, handle in objects.items()
            }
            overlay = overlay_image(frame, detections, ground_truth)
            cv2.imwrite(str(out/'images'/f'frame_{trial:03d}.png'), frame)
            cv2.imwrite(str(out/'overlays'/f'overlay_{trial:03d}.png'), overlay)
            if trial == 0:
                for name, m in masks.items():
                    cv2.imwrite(str(out/'masks'/f'{name}_trial000.png'), m)
            successes = 0
            for name in COLORS:
                det = detections[name]
                gt = ground_truth[name]
                if det is not None and gt is not None:
                    successes += 1
                    error = float(np.linalg.norm(np.asarray(det['centroid']) - np.asarray(gt['top_center_uv'])))
                    iou = box_iou(det['bbox_xyxy'], gt['bbox_xyxy'])
                else:
                    error, iou = None, None
                records.append({
                    'trial': trial, 'color': name,
                    'detected': det is not None,
                    'centroid_u': None if det is None else det['centroid'][0],
                    'centroid_v': None if det is None else det['centroid'][1],
                    'gt_top_center_u': None if gt is None else gt['top_center_uv'][0],
                    'gt_top_center_v': None if gt is None else gt['top_center_uv'][1],
                    'center_error_px': error, 'bbox_iou': iou,
                    'detector_bbox_xyxy': None if det is None else det['bbox_xyxy'],
                    'gt_bbox_xyxy': None if gt is None else gt['bbox_xyxy'],
                    'commanded_pose_for_dataset_generation_only':
                        list(map(float, commanded_poses[list(COLORS).index(name)])),
                })
            print(f'Trial {trial+1:02d}/{args.trials}: {successes}/3 detected')
    finally:
        sim.stopSimulation()

    with (out/'detections.json').open('w') as f:
        json.dump(records, f, indent=2)
    with (out/'detections.csv').open('w', newline='') as f:
        fields = ['trial', 'color', 'detected', 'centroid_u', 'centroid_v',
                  'gt_top_center_u', 'gt_top_center_v', 'center_error_px', 'bbox_iou']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)

    valid = [r for r in records if r['detected'] and r['center_error_px'] is not None]
    summary = {
        'camera': args.camera, 'robot_base': args.robot_base,
        'method': 'RGB-only HSV segmentation + connected contour localization',
        'calibration_file': str(args.calibration.resolve()),
        'n_trials': int(args.trials), 'n_objects_per_trial': len(COLORS),
        'n_expected': len(records), 'n_detected': sum(int(r['detected']) for r in records),
        'detection_rate': float(sum(int(r['detected']) for r in records)/len(records)),
        'mean_center_error_px': float(np.mean([r['center_error_px'] for r in valid])) if valid else None,
        'median_center_error_px': float(np.median([r['center_error_px'] for r in valid])) if valid else None,
        'mean_bbox_iou': float(np.mean([r['bbox_iou'] for r in valid])) if valid else None,
        'per_color': {},
        'ground_truth_policy': 'Ground-truth object poses from simulator only after RGB detection for evaluation; no simulator camera pose used',
        'metric_note': 'Center errors use projected 3D TOP-face center, while detector center is the color-mask image centroid; they need not coincide exactly for perspective/visible side faces.',
    }
    for name in COLORS:
        group = [r for r in records if r['color'] == name]
        hits = [r for r in group if r['center_error_px'] is not None]
        summary['per_color'][name] = {
            'detections': sum(int(r['detected']) for r in group),
            'trials': len(group),
            'mean_center_error_px': float(np.mean([r['center_error_px'] for r in hits])) if hits else None,
            'mean_bbox_iou': float(np.mean([r['bbox_iou'] for r in hits])) if hits else None,
        }
    (out/'metrics.json').write_text(json.dumps(summary, indent=2))

    # Optional chart; no seaborn or hardcoded plot colors.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [[r['center_error_px'] for r in records
               if r['color'] == c and r['center_error_px'] is not None] for c in COLORS]
    if all(groups):
        ax.boxplot(groups, labels=list(COLORS))
        ax.set_ylabel('Detected centroid vs projected top-center (pixels)')
        ax.set_title('Task 4 — localization errors by object color')
        fig.tight_layout()
        fig.savefig(out/'centroid_error_by_color.png', dpi=160)
    plt.close(fig)

    print('\n=== TASK 4 SUMMARY ===')
    print(f"RGB detection rate: {summary['n_detected']}/{summary['n_expected']} = {100*summary['detection_rate']:.1f}%")
    print(f"Mean center error: {summary['mean_center_error_px']} px")
    print(f"Mean box IoU: {summary['mean_bbox_iou']}")
    print('See images, overlays, masks, detections.csv, metrics.json in:', out)
    print('Save the CoppeliaSim scene manually to retain the workcell for Task 5.')


if __name__ == '__main__':
    main()
