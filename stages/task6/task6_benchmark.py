#!/usr/bin/env python3
"""Task 6 benchmark: multi-trial pick-and-place success rate, open vs closed loop.

Each trial samples a new random layout of the three blocks inside the reachable,
camera-visible part of the table, then picks every block with fresh RGB-D
perception and places it on its target. Placement is scored against simulator
ground truth AFTER the motion (evaluation only).

Open loop  : perceive -> plan -> execute once (the baseline).
Closed loop: after the open-loop placement the arm moves out of view, the camera
             re-measures the block and the robot regrasps and corrects it until
             the measured error is below --servo-tol-mm / --servo-tol-deg
             (position-based look-then-move visual servoing, bonus task).
Both are recorded for the same placement, so the comparison is paired.

Targets:
  --target-source world : fixed goals from config/destinations.json (robot frame).
  --target-source pads  : yellow pads at those goals; the goal is the pad pose
                          MEASURED BY THE CAMERA, so a calibration bias common
                          to block and pad cancels in closed loop.

Use --inject-rot-deg / --inject-trans-mm to perturb the camera extrinsics the
controller believes, to test robustness to calibration error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vgm.config import latest_calibration, load_destinations  # noqa: E402
from vgm.geometry import (make_tcp_goal, norm_yaw_pi, obj_pose_matrix,  # noqa: E402
                          perturb_transform, way_above, yaw_of)
from vgm.manipulation import (DEFAULT_XY_TOL_MM, DEFAULT_YAW_TOL_DEG, DEFAULT_Z_TOL_MM,  # noqa: E402
                              IKMotion, chain_joints, find_tip, is_success, live_poses,
                              placement_errors)
from vgm.metrics import describe, rate, success_at_thresholds  # noqa: E402
from vgm.perception import BLOCK_SIZE, COLORS, PAD_SIZE, detect_pads, load_calibration  # noqa: E402
from vgm.scene import (W_from_base, connect, object_pose_B, place_pads, prepare_scene,  # noqa: E402
                       random_layout, read_bgr, read_depth_m, remove_pads, set_block_world,
                       world_to_base, world_yaw_to_base)
from vgm.servo import look_then_move  # noqa: E402


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float))


class Cell:
    """Everything needed to run one placement in the live simulator."""

    def __init__(self, args, client, sim):
        self.args, self.client, self.sim = args, client, sim
        self.base = sim.getObject(args.robot_base)
        self.camera = sim.getObject(args.camera)
        self.K, self.dist, B_C = load_calibration(args.calibration or latest_calibration())
        self.B_C = perturb_transform(B_C, args.inject_rot_deg, args.inject_trans_mm)
        self.tip = find_tip(sim, self.base, args.tip)
        self.joints = chain_joints(sim, self.base, self.tip)
        self.objects = prepare_scene(sim)
        self.dests = load_destinations(args.destinations)
        self.pads = None
        self.motion = None
        self.home = None

    # ---------- motion primitives ----------
    def go_home(self):
        self.motion.move_joints(self.home, 'home', quiet=True)

    def pick_and_place(self, color, grasp_c, grasp_yaw, place_c, place_yaw, hop):
        """Approach, kinematic attach, carry, release, retreat, home."""
        tcp = self.args.tcp_height
        grasp = make_tcp_goal(grasp_c, grasp_yaw, tcp)
        place = make_tcp_goal(place_c, place_yaw, tcp)
        m, obj = self.motion, self.objects[color]
        m.move(way_above(grasp, hop), 'approach', quiet=True)
        m.move(grasp, 'grasp', quiet=True)
        tip_now = obj_pose_matrix(self.sim.getObjectPose(self.tip, self.base))
        if float(np.linalg.norm(tip_now[:3, 3] - grasp[:3, 3])) > 0.02:
            raise RuntimeError('TCP not at planned grasp; refusing to attach')
        self.sim.setObjectParent(obj, self.tip, True)
        self.sim.step()
        m.move(way_above(grasp, hop), 'lift', quiet=True)
        m.move(way_above(place, hop), 'transfer', quiet=True)
        m.move(place, 'lower', quiet=True)
        self.sim.setObjectParent(obj, -1, True)
        self.sim.step()
        m.move(way_above(place, hop), 'retreat', quiet=True)
        self.go_home()

    def preflight(self, grasp_c, grasp_yaw, place_c, place_yaw):
        tcp, hop = self.args.tcp_height, self.args.lift_height
        for T in (make_tcp_goal(grasp_c, grasp_yaw, tcp), make_tcp_goal(place_c, place_yaw, tcp)):
            self.motion.plan(T)
            self.motion.plan(way_above(T, hop))

    # ---------- perception ----------
    def perceive(self, out=None):
        poses, diag = live_poses(self.sim, self.camera, self.K, self.dist, self.B_C, out, verbose=False)
        return poses, diag

    def targets(self):
        """Measured targets for the controller + true targets for evaluation."""
        block_half = float(BLOCK_SIZE[2]) / 2
        nominal = {c: world_to_base(self.sim, self.base, d) for c, d in self.dests.items()}
        nominal_yaw = {c: (world_yaw_to_base(self.sim, self.base, math.radians(d[3])) if len(d) > 3 else None)
                       for c, d in self.dests.items()}
        if self.args.target_source == 'world':
            return nominal, nominal_yaw, nominal, nominal_yaw
        bgr = read_bgr(self.sim, self.camera)
        depth = read_depth_m(self.sim, self.camera, bgr.shape[:2])
        found = detect_pads(bgr, depth, self.K, self.dist, self.B_C)
        if len(found) < len(self.dests):
            raise RuntimeError(f'Only {len(found)} placement pads detected')
        ctrl, ctrl_yaw, true, true_yaw = {}, {}, {}, {}
        unused = list(found)
        for color in self.dests:   # match each pad to the nearest NOMINAL goal (design layout)
            k = int(np.argmin([np.linalg.norm(T[:2, 3] - nominal[color][:2]) for T, _ in unused]))
            T, _ = unused.pop(k)
            c = T[:3, 3].copy()
            c[2] += float(PAD_SIZE[2]) / 2 + block_half
            ctrl[color], ctrl_yaw[color] = c, yaw_of(T)
            gt = object_pose_B(self.sim, self.pads[color], self.base)      # evaluation only
            g = gt[:3, 3].copy()
            g[2] += float(PAD_SIZE[2]) / 2 + block_half
            true[color], true_yaw[color] = g, yaw_of(gt)
        return ctrl, ctrl_yaw, true, true_yaw


def sample_layout(cell, rng):
    args = cell.args
    W_B = W_from_base(cell.sim, cell.base)
    base_xy = W_B[:2, 3]
    dest_xy = [np.asarray(d[:2], float) for d in cell.dests.values()]

    def accept(xy):
        r = float(np.linalg.norm(xy - base_xy))
        return (args.min_base_dist <= r <= args.max_base_dist and
                all(np.linalg.norm(xy - d) >= args.target_clearance for d in dest_xy))

    pts = random_layout(rng, n=len(COLORS), minimum_separation=args.block_separation, accept=accept)
    layout = {}
    for (color, handle), xy in zip(cell.objects.items(), pts):
        yaw = float(rng.uniform(-85, 85))
        set_block_world(cell.sim, handle, xy[0], xy[1], yaw)
        layout[color] = [float(xy[0]), float(xy[1]), yaw]
    for _ in range(2):
        cell.sim.step()
    return layout


def place_yaw_for(est_yaw, target_yaw):
    """TCP yaw at placement: minimal wrist rotation that achieves the target yaw."""
    return est_yaw if target_yaw is None else est_yaw + norm_yaw_pi(target_yaw - est_yaw)


def run_trial(cell, trial, rng, out):
    args, sim = cell.args, cell.sim
    for obj in cell.objects.values():
        sim.setObjectParent(obj, -1, True)
    cell.go_home()
    rejected = 0
    for attempt in range(args.max_layout_attempts):
        layout = sample_layout(cell, rng)
        ctrl, ctrl_yaw, true_t, true_yaw = cell.targets()
        poses, diag = cell.perceive()
        problem = None
        for color in args.colors:
            est = poses[color]
            if est is None:
                problem = f'{color}: not detected ({diag[color].get("reason")})'
                break
            try:
                cell.preflight(est[:3, 3], yaw_of(est), ctrl[color],
                               place_yaw_for(yaw_of(est), ctrl_yaw[color]))
            except Exception as exc:
                problem = f'{color}: IK preflight failed ({exc})'
                break
        if problem is None:
            break
        rejected += 1
        print(f'  layout {attempt + 1} rejected: {problem}')
    else:
        return [], rejected, layout

    records = []
    for color in args.colors:
        rec = {'trial': trial, 'color': color, 'layout_world_xy_yaw': layout[color],
               'target_source': args.target_source, 'status': 'started'}
        records.append(rec)
        try:
            for _ in range(2):
                sim.step()
            poses, diag = cell.perceive(out / f'trial_{trial:03d}' / color if args.save_images else None)
            est = poses[color]
            if est is None:
                rec.update(status='not_detected', reason=diag[color].get('reason'))
                continue
            e_yaw = yaw_of(est)
            t0 = time.time()
            cell.pick_and_place(color, est[:3, 3].copy(), e_yaw, ctrl[color],
                                place_yaw_for(e_yaw, ctrl_yaw[color]), args.lift_height)
            truth = object_pose_B(sim, cell.objects[color], cell.base)             # evaluation only
            open_err = placement_errors(truth, true_t[color], true_yaw[color])
            open_err['success'] = is_success(open_err, args.xy_tol_mm, args.z_tol_mm, args.yaw_tol_deg)
            rec.update(status='executed', open_loop=open_err, open_loop_seconds=time.time() - t0)
            if args.closed_loop:
                gt_trace = []

                def observe():
                    p, _ = cell.perceive()
                    gt = object_pose_B(sim, cell.objects[color], cell.base)        # evaluation only
                    gt_trace.append(placement_errors(gt, true_t[color], true_yaw[color]))
                    return p[color]

                def regrasp(goal):
                    g_c, g_yaw, p_c, p_yaw = goal
                    cell.pick_and_place(color, g_c, g_yaw, p_c, p_yaw, args.servo_hop)

                t1 = time.time()
                history = look_then_move(observe, regrasp, ctrl[color], ctrl_yaw[color],
                                         tol_mm=args.servo_tol_mm, tol_deg=args.servo_tol_deg,
                                         gain=args.servo_gain, max_iters=args.max_servo_iters)
                for h, g in zip(history, gt_trace):
                    h['true_xy_error_mm'] = g['placement_xy_error_mm']
                    h['true_yaw_error_deg'] = g['placement_yaw_error_deg']
                truth = object_pose_B(sim, cell.objects[color], cell.base)         # evaluation only
                closed_err = placement_errors(truth, true_t[color], true_yaw[color])
                closed_err['success'] = is_success(closed_err, args.xy_tol_mm, args.z_tol_mm, args.yaw_tol_deg)
                rec.update(closed_loop=closed_err, servo_history=history,
                           servo_corrections=max(0, len(history) - 1),
                           closed_loop_extra_seconds=time.time() - t1)
            msg = f"open {open_err['placement_xy_error_mm']:.2f} mm"
            if args.closed_loop:
                msg += f" -> closed {rec['closed_loop']['placement_xy_error_mm']:.2f} mm " \
                       f"({rec['servo_corrections']} corrections)"
            print(f'  {color}: {msg}')
        except Exception as exc:
            rec.update(status='execution_error', reason=str(exc))
            print(f'  {color}: execution error: {exc}')
            for obj in cell.objects.values():
                sim.setObjectParent(obj, -1, True)
            try:
                cell.go_home()
            except Exception:
                pass
    return records, rejected, layout


def summarize(records, args, n_trials, rejected_layouts):
    def block(key):
        attempted = records
        errs = [r[key] if r.get('status') == 'executed' and key in r else None for r in attempted]
        ok = [e for e in errs if e is not None]
        return {'success': rate(sum(bool(e['success']) for e in ok), len(attempted)),
                'xy_error_mm': describe(e['placement_xy_error_mm'] for e in ok),
                'z_error_mm': describe(e['placement_z_error_mm'] for e in ok),
                'yaw_error_deg': describe(e['placement_yaw_error_deg'] for e in ok
                                          if e['placement_yaw_error_deg'] is not None),
                'xy_success_at_thresholds': success_at_thresholds(
                    [None if e is None else e['placement_xy_error_mm'] for e in errs])}

    failures = {}
    for r in records:
        if r.get('status') != 'executed':
            failures[r['status']] = failures.get(r['status'], 0) + 1
    out = {'n_trials': n_trials, 'n_placements_attempted': len(records),
           'layouts_rejected_by_ik_or_detection': rejected_layouts,
           'tolerances': {'xy_mm': args.xy_tol_mm, 'z_mm': args.z_tol_mm, 'yaw_deg': args.yaw_tol_deg},
           'target_source': args.target_source,
           'injected_extrinsic_error': {'rot_deg': list(args.inject_rot_deg),
                                        'trans_mm': list(args.inject_trans_mm)},
           'failure_counts': failures, 'open_loop': block('open_loop')}
    if args.closed_loop:
        out['closed_loop'] = block('closed_loop')
        corr = [r['servo_corrections'] for r in records if 'servo_corrections' in r]
        out['servo'] = {'corrections': describe(corr), 'tol_mm': args.servo_tol_mm,
                        'tol_deg': args.servo_tol_deg, 'gain': args.servo_gain,
                        'max_iterations': args.max_servo_iters}
    out['grasp_mode'] = 'kinematic_attach_proxy (physics disabled)'
    out['ground_truth_policy'] = 'Block/pad poses read only after motion, for scoring.'
    return out


def plot(records, args, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ex = [r for r in records if r.get('status') == 'executed']
    if not ex:
        return
    series = [('open loop', [r['open_loop']['placement_xy_error_mm'] for r in ex])]
    if args.closed_loop:
        series.append(('closed loop', [r['closed_loop']['placement_xy_error_mm'] for r in ex if 'closed_loop' in r]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].boxplot([s[1] for s in series], labels=[s[0] for s in series])
    ax[0].axhline(args.xy_tol_mm, ls='--', lw=1, color='grey')
    ax[0].set_ylabel('XY placement error (mm)')
    ax[0].set_title('Placement error')
    for name, vals in series:
        v = np.sort(vals)
        ax[1].step(v, np.arange(1, len(v) + 1) / len(v), where='post', label=name)
    ax[1].axvline(args.xy_tol_mm, ls='--', lw=1, color='grey')
    ax[1].set_xlabel('XY placement error (mm)')
    ax[1].set_ylabel('Fraction of placements')
    ax[1].set_title('Cumulative distribution')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / 'placement_error.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--camera', default='/Task2Camera')
    p.add_argument('--robot-base', default='/Franka')
    p.add_argument('--tip', default='')
    p.add_argument('--calibration', type=Path, default=None)
    p.add_argument('--destinations', type=Path, default=None)
    p.add_argument('--colors', nargs='+', choices=list(COLORS), default=list(COLORS))
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--tcp-height', type=float, default=.09)
    p.add_argument('--lift-height', type=float, default=.18)
    p.add_argument('--xy-tol-mm', type=float, default=DEFAULT_XY_TOL_MM)
    p.add_argument('--z-tol-mm', type=float, default=DEFAULT_Z_TOL_MM)
    p.add_argument('--yaw-tol-deg', type=float, default=DEFAULT_YAW_TOL_DEG)
    p.add_argument('--target-source', choices=['world', 'pads'], default='world')
    p.add_argument('--closed-loop', action='store_true', help='Add look-then-move correction (bonus)')
    p.add_argument('--servo-tol-mm', type=float, default=2.0)
    p.add_argument('--servo-tol-deg', type=float, default=1.5)
    p.add_argument('--servo-gain', type=float, default=1.0)
    p.add_argument('--max-servo-iters', type=int, default=3)
    p.add_argument('--servo-hop', type=float, default=.05, help='Lift height during corrections (m)')
    p.add_argument('--inject-rot-deg', type=float, nargs=3, default=[0., 0., 0.], metavar=('RX', 'RY', 'RZ'))
    p.add_argument('--inject-trans-mm', type=float, nargs=3, default=[0., 0., 0.], metavar=('TX', 'TY', 'TZ'))
    p.add_argument('--min-base-dist', type=float, default=0.40, help='Layout pre-filter (m from base)')
    p.add_argument('--max-base-dist', type=float, default=0.78)
    p.add_argument('--target-clearance', type=float, default=0.15, help='Min source-to-goal distance (m)')
    p.add_argument('--block-separation', type=float, default=0.12)
    p.add_argument('--max-layout-attempts', type=int, default=15)
    p.add_argument('--save-images', action='store_true')
    p.add_argument('--tcp-verified', action='store_true', required=True,
                   help='Required: confirm the TCP dummy was visually checked in this scene')
    p.add_argument('--output', type=Path, default=ROOT / 'runs')
    args = p.parse_args()
    out = args.output.resolve() / ('task6_benchmark_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'config.json', {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})

    client, sim = connect()
    if sim.getSimulationState() != sim.simulation_stopped:
        raise SystemExit('Stop the simulation first.')
    cell = Cell(args, client, sim)
    if args.target_source == 'pads':
        cell.pads = place_pads(sim, cell.dests)
    else:
        remove_pads(sim)
    rng = np.random.default_rng(args.seed)
    records, rejected_total = [], 0
    sim.setStepping(True)
    sim.startSimulation()
    try:
        for _ in range(3):
            sim.step()
        cell.motion = IKMotion(client, sim, cell.base, cell.tip, cell.joints)
        cell.home = cell.motion.joint_positions()
        for trial in range(args.trials):
            print(f'\n=== Trial {trial + 1}/{args.trials} ===')
            recs, rejected, _ = run_trial(cell, trial, rng, out)
            rejected_total += rejected
            if not recs:
                print('  no reachable layout found; trial skipped')
            records.extend(recs)
            dump(out / 'placements.json', records)
    finally:
        if cell.motion is not None:
            cell.motion.cleanup()
        sim.stopSimulation()
        summary = summarize(records, args, args.trials, rejected_total)
        dump(out / 'summary.json', summary)
        with (out / 'placements.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['trial', 'color', 'status', 'open_xy_mm', 'open_z_mm', 'open_yaw_deg', 'open_success',
                        'closed_xy_mm', 'closed_yaw_deg', 'closed_success', 'corrections'])
            for r in records:
                o, c = r.get('open_loop', {}), r.get('closed_loop', {})
                w.writerow([r['trial'], r['color'], r['status'], o.get('placement_xy_error_mm'),
                            o.get('placement_z_error_mm'), o.get('placement_yaw_error_deg'), o.get('success'),
                            c.get('placement_xy_error_mm'), c.get('placement_yaw_error_deg'), c.get('success'),
                            r.get('servo_corrections')])
        plot(records, args, out)
    print('\n' + json.dumps(summary, indent=2, default=float))
    print('Results:', out)


if __name__ == '__main__':
    main()
