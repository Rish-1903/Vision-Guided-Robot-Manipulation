#!/usr/bin/env python3
"""Task 6 (single pick): open-loop vision-guided pick-and-place of one block.

Live RGB-D -> Task 4 HSV mask -> Task 5 cuboid pose -> simIK waypoints ->
kinematic attach -> place at a world-frame goal. The block's true pose is read
only after release, to score the placement. For the multi-trial success-rate
benchmark and the closed-loop comparison use ``task6_benchmark.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vgm.config import latest_calibration, load_destinations  # noqa: E402
from vgm.geometry import make_tcp_goal, obj_pose_matrix, way_above, yaw_of  # noqa: E402
from vgm.manipulation import (DEFAULT_XY_TOL_MM, DEFAULT_YAW_TOL_DEG, DEFAULT_Z_TOL_MM,  # noqa: E402
                              IKMotion, chain_joints, evaluate_object, find_tip, live_poses)
from vgm.perception import COLORS, load_calibration  # noqa: E402
from vgm.scene import apply_layout, connect, get_maybe, prepare_scene, world_to_base  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera', default='/Task2Camera')
    p.add_argument('--robot-base', default='/Franka')
    p.add_argument('--tip', default='', help='Full path of the TCP dummy (auto-discovered if unique)')
    p.add_argument('--color', choices=list(COLORS), default='red')
    p.add_argument('--calibration', type=Path, default=None)
    p.add_argument('--destinations', type=Path, default=None,
                   help='JSON colour -> world [x, y, z(, yaw_deg)] (default: config/destinations.json)')
    p.add_argument('--tcp-height', type=float, default=.09)
    p.add_argument('--lift-height', type=float, default=.18)
    p.add_argument('--xy-tol-mm', type=float, default=DEFAULT_XY_TOL_MM)
    p.add_argument('--z-tol-mm', type=float, default=DEFAULT_Z_TOL_MM)
    p.add_argument('--yaw-tol-deg', type=float, default=DEFAULT_YAW_TOL_DEG)
    p.add_argument('--output', type=Path, default=ROOT / 'runs' / 'task6_single')
    p.add_argument('--setup-demo-layout', action='store_true',
                   help='Before starting, stage the blocks at the fixed Task 4 layout (scene setup only)')
    p.add_argument('--execute', action='store_true', help='Move the robot (default: dry run)')
    args = p.parse_args()
    if args.tcp_height < .025 or args.lift_height < .05:
        p.error('Specify a plausible TCP offset and lift height')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    K, dist, B_from_C = load_calibration(args.calibration or latest_calibration())
    dest = load_destinations(args.destinations)[args.color]
    client, sim = connect()
    base, camera = sim.getObject(args.robot_base), sim.getObject(args.camera)
    if args.setup_demo_layout:
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError('Stop the simulation before --setup-demo-layout.')
        apply_layout(sim, prepare_scene(sim), np.random.default_rng(42), 0)
    obj = get_maybe(sim, COLORS[args.color]['alias'])
    if obj is None:
        raise SystemExit(f"Missing {COLORS[args.color]['alias']}; open the saved workcell scene.")
    tip = find_tip(sim, base, args.tip)
    joints = chain_joints(sim, base, tip)
    sim.setStepping(True)
    sim.startSimulation()
    motion = None
    result = {'color': args.color, 'grasp_mode': 'kinematic_attach_proxy',
              'uses_object_true_pose_for_planning': False, 'collision_free_planning': False,
              'success': False}
    try:
        for _ in range(3):
            sim.step()
        poses, diag = live_poses(sim, camera, K, dist, B_from_C, out / 'perception')
        est = poses[args.color]
        if est is None:
            raise RuntimeError(f'Live RGB-D pose for {args.color} unavailable: {diag[args.color]}')
        yaw = yaw_of(est)
        pick_c = est[:3, 3].copy()
        place_c = world_to_base(sim, base, dest)
        grasp = make_tcp_goal(pick_c, yaw, args.tcp_height)
        place = make_tcp_goal(place_c, yaw, args.tcp_height)
        result.update(estimated_pick_center_B_m=pick_c.tolist(), goal_center_B_m=place_c.tolist(),
                      estimated_yaw_deg=math.degrees(yaw), perception_diagnostics=diag[args.color])
        print('Vision-estimated centre [base m]:', np.round(pick_c, 4), '| goal:', np.round(place_c, 4))
        if not args.execute:
            result['status'] = 'dry_run'
            print('DRY RUN: no joint motion. Re-run with --execute after checking the scene.')
            return
        motion = IKMotion(client, sim, base, tip, joints)
        motion.move(way_above(grasp, args.lift_height), 'approach')
        motion.move(grasp, 'descend')
        if np.linalg.norm(obj_pose_matrix(sim.getObjectPose(tip, base))[:3, 3] - grasp[:3, 3]) > .02:
            raise RuntimeError('TCP not at the planned grasp; refusing to attach')
        sim.setObjectParent(obj, tip, True)
        for _ in range(4):
            sim.step()
        motion.move(way_above(grasp, args.lift_height), 'lift')
        motion.move(way_above(place, args.lift_height), 'transfer')
        motion.move(place, 'lower')
        sim.setObjectParent(obj, -1, True)
        for _ in range(4):
            sim.step()
        result.update(evaluate_object(sim, obj, base, place_c, None, xy_tol_mm=args.xy_tol_mm,
                                      z_tol_mm=args.z_tol_mm, yaw_tol_deg=args.yaw_tol_deg))
        result['status'] = 'executed'
        motion.move(way_above(place, args.lift_height), 'retreat')
        print(f"Placement error XY {result['placement_xy_error_mm']:.2f} mm, "
              f"Z {result['placement_z_error_mm']:.2f} mm -> success={result['success']}")
    except Exception as exc:
        result.update(status='failed', error=str(exc))
        raise
    finally:
        if motion is not None:
            motion.cleanup()
        sim.stopSimulation()
        (out / 'trial_result.json').write_text(json.dumps(result, indent=2))
        print('Saved:', out / 'trial_result.json')


if __name__ == '__main__':
    main()
