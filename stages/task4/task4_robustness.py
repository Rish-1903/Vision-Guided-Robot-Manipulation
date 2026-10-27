#!/usr/bin/env python3
"""Tasks 4-5 robustness study: where does colour localization + RGB-D pose fail?

For each scenario the SAME seeded layouts are rendered, so scenarios are paired.

Image-space scenarios (applied to the rendered RGB / depth, see vgm/perturb.py):
  dim / dark / bright / gamma / warm colour cast / RGB noise / depth noise.
Scene scenarios (built in CoppeliaSim):
  distractor_small_red   a 3 cm red cube elsewhere on the table
  distractor_same_red    a second red block of identical size (ambiguous by design)
  occlusion_25/50/75     a grey plate hiding that fraction of one block's top face
  adjacent_red_green     green block touching the red block's long side

Outputs per-scenario detection rate, centroid error, pose rate, translation and
rotation error, and gross-error counts (>20 mm). Ground truth is read only after
inference, for scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vgm.config import latest_calibration  # noqa: E402
from vgm.geometry import translation_rotation_errors  # noqa: E402
from vgm.metrics import describe, rate  # noqa: E402
from vgm.perception import (BLOCK_SIZE, BLOCK_Z, COLORS, box_iou, detect_from_rgb,  # noqa: E402
                            estimate_pose_from_mask, load_calibration, overlay_image, project_box)
from vgm.perturb import IMAGE_SCENARIOS  # noqa: E402
from vgm.scene import (connect, make_block, object_pose_B, park, prepare_scene,  # noqa: E402
                       random_layout, read_bgr, read_depth_m, set_block_world)

GROSS_MM = 20.0
SCENE_SCENARIOS = ['distractor_small_red', 'distractor_same_red', 'occlusion_25', 'occlusion_50',
                   'occlusion_75', 'adjacent_red_green']
HELPERS = ('/T4_Distractor', '/T4_Occluder')


def stage_layout(sim, objects, rng):
    pts = random_layout(rng, minimum_separation=0.17)
    layout = {}
    for (name, h), xy in zip(objects.items(), pts):
        yaw = float(rng.uniform(-85, 85))
        set_block_world(sim, h, xy[0], xy[1], yaw)
        layout[name] = (float(xy[0]), float(xy[1]), yaw)
    return layout


def apply_scene_scenario(sim, objects, layout, scenario, rng, trial):
    """Modify the staged scene; returns the colour of interest for this trial."""
    for alias in HELPERS:
        park(sim, alias)
    if scenario.startswith('distractor'):
        size = [0.03, 0.03, 0.03] if scenario.endswith('small_red') else list(BLOCK_SIZE)
        d = make_block(sim, '/T4_Distractor', size, COLORS['red']['rgb'])
        taken = [np.array(v[:2]) for v in layout.values()]
        pos = random_layout(rng, n=1, minimum_separation=0.0,
                            accept=lambda xy: all(np.linalg.norm(xy - t) > 0.14 for t in taken))[0]
        z = BLOCK_Z - BLOCK_SIZE[2] / 2 + size[2] / 2
        set_block_world(sim, d, pos[0], pos[1], float(rng.uniform(-85, 85)), z=z)
        return 'red'
    if scenario.startswith('occlusion'):
        frac = int(scenario.split('_')[1]) / 100.0
        color = list(COLORS)[trial % 3]
        x, y, yaw = layout[color]
        L, W = float(BLOCK_SIZE[0]), float(BLOCK_SIZE[1])
        plate = make_block(sim, '/T4_Occluder', [frac * L, W + 0.02, 0.004], (0.5, 0.5, 0.5), emission=0.0)
        shift = L / 2 - frac * L / 2
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        set_block_world(sim, plate, x + c * shift, y + s * shift, yaw,
                        z=BLOCK_Z + BLOCK_SIZE[2] / 2 + 0.03)
        return color
    if scenario == 'adjacent_red_green':
        x, y, yaw = layout['red']
        off = float(BLOCK_SIZE[1]) + 0.002
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        set_block_world(sim, objects['green'], x - s * off, y + c * off, yaw)
        layout['green'] = (x - s * off, y + c * off, yaw)
        return 'red'
    return None


def infer_and_score(rgb, depth, K, dist, B_C, truth_B, scenario, trial, focus):
    detections, masks = detect_from_rgb(rgb)
    rows = []
    for name in COLORS:
        det = detections[name]
        pose, diag = estimate_pose_from_mask(depth, masks[name], K, dist, B_C)
        gt_px = project_box(truth_B[name], BLOCK_SIZE, K, dist, B_C)
        row = {'scenario': scenario, 'trial': trial, 'color': name, 'focus': name == focus if focus else True,
               'detected': det is not None, 'n_blobs': None if det is None else det['n_blobs'],
               'center_error_px': None, 'bbox_iou': None, 'pose_ok': pose is not None,
               'pose_reason': '' if pose is not None else diag['reason'],
               'translation_error_mm': None, 'rotation_error_deg': None}
        if det is not None and gt_px is not None:
            row['center_error_px'] = float(np.linalg.norm(np.asarray(det['centroid']) - gt_px['top_center_uv']))
            row['bbox_iou'] = box_iou(det['bbox_xyxy'], gt_px['bbox_xyxy'])
        if pose is not None:
            t, _, r = translation_rotation_errors(pose, truth_B[name])
            row['translation_error_mm'], row['rotation_error_deg'] = t, r
        rows.append(row)
    return rows, detections


def summarize(rows):
    out = {}
    for sc in dict.fromkeys(r['scenario'] for r in rows):
        g = [r for r in rows if r['scenario'] == sc and r['focus']]
        poses = [r for r in g if r['pose_ok']]
        gross = sum(1 for r in poses if r['translation_error_mm'] > GROSS_MM)
        out[sc] = {'n': len(g), 'detection': rate(sum(r['detected'] for r in g), len(g)),
                   'center_error_px': describe(r['center_error_px'] for r in g),
                   'pose': rate(len(poses), len(g)),
                   'translation_error_mm': describe(r['translation_error_mm'] for r in poses),
                   'rotation_error_deg': describe(r['rotation_error_deg'] for r in poses),
                   f'gross_errors_over_{int(GROSS_MM)}mm': gross,
                   'usable_pose_rate': rate(len(poses) - gross, len(g))}
    return out


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(summary)
    usable = [100 * (summary[n]['usable_pose_rate']['rate'] or 0) for n in names]
    med = [summary[n]['translation_error_mm']['median'] or 0 for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].barh(names, usable)
    ax[0].set_xlabel('Usable pose rate (%) - detected and < 20 mm')
    ax[0].invert_yaxis()
    ax[1].barh(names, med)
    ax[1].set_xlabel('Median translation error (mm)')
    ax[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / 'robustness.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--camera', default='/Task2Camera')
    p.add_argument('--robot-base', default='/Franka')
    p.add_argument('--calibration', type=Path, default=None)
    p.add_argument('--trials', type=int, default=15, help='Layouts per scenario')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--scenarios', nargs='+', default=list(IMAGE_SCENARIOS) + SCENE_SCENARIOS,
                   choices=list(IMAGE_SCENARIOS) + SCENE_SCENARIOS)
    p.add_argument('--output', type=Path, default=ROOT / 'runs' / 'task4_robustness')
    args = p.parse_args()
    out = args.output.resolve()
    (out / 'examples').mkdir(parents=True, exist_ok=True)
    K, dist, B_C = load_calibration(args.calibration or latest_calibration())
    _, sim = connect()
    camera, base = sim.getObject(args.camera), sim.getObject(args.robot_base)
    objects = prepare_scene(sim)
    rows = []
    sim.setStepping(True)
    sim.startSimulation()
    try:
        for scenario in args.scenarios:
            print(f'\n=== {scenario} ===')
            for trial in range(args.trials):
                rng = np.random.default_rng(args.seed + trial)      # paired layouts
                layout = stage_layout(sim, objects, rng)
                focus = None
                if scenario in SCENE_SCENARIOS:
                    focus = apply_scene_scenario(sim, objects, layout, scenario, rng, trial)
                else:
                    for alias in HELPERS:
                        park(sim, alias)
                sim.step()
                sim.step()
                rgb = read_bgr(sim, camera)
                depth = read_depth_m(sim, camera, rgb.shape[:2])
                if scenario in IMAGE_SCENARIOS:
                    f_rgb, f_depth = IMAGE_SCENARIOS[scenario]
                    noise = np.random.default_rng(1000 + trial)
                    rgb = f_rgb(rgb, noise) if f_rgb else rgb
                    depth = f_depth(depth, noise) if f_depth else depth
                truth_B = {n: object_pose_B(sim, h, base) for n, h in objects.items()}   # scoring only
                trial_rows, dets = infer_and_score(rgb, depth, K, dist, B_C, truth_B, scenario, trial, focus)
                rows.extend(trial_rows)
                if trial == 0:
                    cv2.imwrite(str(out / 'examples' / f'{scenario}.png'), overlay_image(rgb, dets))
                ok = sum(r['pose_ok'] for r in trial_rows if r['focus'])
                print(f'  trial {trial + 1:02d}: {ok}/{sum(r["focus"] for r in trial_rows)} poses')
    finally:
        for alias in HELPERS:
            park(sim, alias)
        sim.stopSimulation()
    with (out / 'robustness_rows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    (out / 'robustness_summary.json').write_text(json.dumps(summary, indent=2))
    plot(summary, out)
    print('\n{:<22} {:>8} {:>8} {:>10} {:>8}'.format('scenario', 'detect%', 'pose%', 'med t mm', 'gross'))
    for sc, s in summary.items():
        print('{:<22} {:>8.1f} {:>8.1f} {:>10} {:>8}'.format(
            sc, 100 * s['detection']['rate'], 100 * s['pose']['rate'],
            'n/a' if s['translation_error_mm']['median'] is None else f"{s['translation_error_mm']['median']:.2f}",
            s[f'gross_errors_over_{int(GROSS_MM)}mm']))
    print('Results:', out)


if __name__ == '__main__':
    main()
