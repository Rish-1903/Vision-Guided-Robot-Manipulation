#!/usr/bin/env python3
"""One entry point for project Tasks 2–6: new-scene creation + all experiments.

Task 1 is a documented engineering choice; Tasks 2/3 are one-time calibrations.
Tasks 4/5 perform repeatable offline-in-simulator evaluations and fresh inference
is repeated for each pick in Task 6. The Task 6 stage keeps ONE CoppeliaSim
simulation active for the complete selected color sequence.

No object's simulation ground-truth pose is used for picking or motion planning.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent
S2 = ROOT / 'stages' / 'task2'
S3 = ROOT / 'stages' / 'task3'
S4 = ROOT / 'stages' / 'task4'
S5 = ROOT / 'stages' / 'task5'
S6 = ROOT / 'stages' / 'task6'
sys.path.insert(0, str(ROOT))

from vgm.config import (latest_calibration, load_destinations,  # noqa: E402
                        load_start_layout, validate_layout)
from vgm.geometry import obj_pose_matrix, yaw_of  # noqa: E402
from vgm.manipulation import (DEFAULT_XY_TOL_MM, DEFAULT_YAW_TOL_DEG,  # noqa: E402
                              DEFAULT_Z_TOL_MM)


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))


def call_stage(name: str, argv: list[str], output_dir: Path, pipeline):
    """Run an existing stage and preserve exact console output and exit status."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logfile = output_dir / 'stage.log'
    print(f'\n========== {name} ==========\n$ ' + ' '.join(map(str, argv)), flush=True)
    with logfile.open('w') as log:
        process = subprocess.Popen([sys.executable, *map(str, argv)], cwd=ROOT,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        exitcode = process.wait()
    pipeline['stages'].append({'name': name, 'exit_code': exitcode, 'log': str(logfile)})
    dump_json(pipeline['manifest_path'], {k:v for k,v in pipeline.items() if k != 'manifest_path'})
    if exitcode:
        raise RuntimeError(f'{name} failed with exit code {exitcode}. See {logfile}')


def default_existing_calibration() -> Path:
    """Reuse only calibration produced by this project."""
    return latest_calibration()


def validate_calibration(file: Path):
    with np.load(file) as arr:
        K = np.asarray(arr['K'], float)
        B_C = np.asarray(arr['B_from_Ccv'], float)
    if K.shape != (3,3) or B_C.shape != (4,4):
        raise RuntimeError('Calibration must contain K (3×3) and B_from_Ccv (4×4)')
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(B_C)):
        raise RuntimeError('Calibration contains NaN/Inf')
    if not np.allclose(B_C[3], [0,0,0,1], atol=1e-7) or np.linalg.det(B_C[:3,:3]) < 0.95:
        raise RuntimeError('Invalid SE(3) transformation')
    if not 0.01 < K[0,0] < 100000 or not 0.01 < K[1,1] < 100000:
        raise RuntimeError('Invalid focal length')
    print('Validated estimated K:\n', K)
    print('Validated robot_base_T_OpenCV_camera:\n', B_C)


def assert_calibration_matches_scene(sim, base, camera, calibration: Path):
    """VALIDATION ONLY: ensure calibration belongs to CURRENT scene.

    Simulator truth is not used for pose estimation, grasp target generation,
    or controller operation; only to reject an incorrect scene/calibration.
    """
    from scipy.spatial.transform import Rotation as R
    with np.load(calibration) as npz:
        est=np.asarray(npz['B_from_Ccv'],float)
    sim_pose=obj_pose_matrix(sim.getObjectPose(camera,base))
    sim_to_cv=np.eye(4)
    sim_to_cv[:3,:3]=np.diag([-1.,-1.,1.])
    gt=sim_pose@sim_to_cv
    error_m=float(np.linalg.norm(gt[:3,3]-est[:3,3]))
    error_deg=float(R.from_matrix(gt[:3,:3].T@est[:3,:3]).magnitude()*180/math.pi)
    print('Calibration CURRENT-scene validation: '
          f'translation={error_m*1000:.3f}mm, rotation={error_deg:.4f}deg')
    if error_m>.05 or error_deg>5.:
        raise RuntimeError('The current camera/Franka geometry does not match '
            'the saved calibration! Re-run --mode fresh or --mode full before execution.')


def check_recovery_quality(recovery_json: Path, *, strict: bool = True):
    """Gate fresh calibration on withheld images and ground truth *validation only*."""
    data = json.loads(recovery_json.read_text())
    holdout = data.get('validation_projection_mean_rms_px')
    truth = data.get('simulator_ground_truth_validation_only', {})
    problems = []
    if holdout is None or not math.isfinite(float(holdout)) or holdout > 3.0:
        problems.append(f'held-out landmark reprojection RMS={holdout!r} px (limit: 3)')
    te = truth.get('translation_error_mm')
    re = truth.get('rotation_error_deg')
    if te is None or not math.isfinite(float(te)) or te > 50.0:
        problems.append(f'camera-to-robot translation validation={te!r} mm (limit: 50)')
    if re is None or not math.isfinite(float(re)) or re > 5.0:
        problems.append(f'camera-to-robot rotation validation={re!r} deg (limit: 5)')
    if problems:
        message = 'Calibration quality gate failed: ' + '; '.join(problems)
        if strict: raise RuntimeError(message)
        print('WARNING:',message)
    else:
        print(f'Calibration quality PASS: held-out {holdout:.3f} px; '
              f'translation {te:.3f} mm; rotation {re:.4f} deg')
    return data


def run_calibration(args, output, pipeline):
    data = output/'task2'/'dataset'
    call_stage('Task 2A — Checkerboard image capture', [
        S2/'01_capture_coppeliasim.py', '--views', args.calibration_views,
        '--output', data], output/'task2', pipeline)
    call_stage('Task 2B — Intrinsic calibration', [
        S2/'02_calibrate_camera.py', '--dataset', data], output/'task2', pipeline)
    initial_K_file = data/'calibration_results'/'camera_calibration.npz'
    with np.load(initial_K_file) as f:
        estimated = np.asarray(f['K_constrained'] if 'K_constrained' in f else f['K_pinhole'],float)
        reference = np.asarray(f['K_sim_reference_for_validation_only'],float)
    rel = abs(estimated[0,0]/reference[0,0]-1)
    pipeline['task2_constrained_focal_error_percent_validation_only'] = float(rel*100)
    if rel > .15:
        print('WARNING: Checkerboard-only calibration is not independently validated: fitted focal length differs by >15% from camera reference.')
        print('Task 2 requires a checkerboard rendering/corner-geometry audit for the final report. Task 3 will jointly recover')
        print('K and eye-to-hand extrinsics from measured 3D-2D landmarks (as in your earlier successful run).')
    s3out = output/'task3'
    call_stage('Task 3A — Eye-to-hand landmark collection', [
        S3/'03_calibrate_camera_robot.py', '--robot-base', args.robot_base,
        '--camera', args.camera, '--intrinsics', initial_K_file,
        '--views', args.extrinsic_views, '--collect-only', '--output', s3out], s3out, pipeline)
    call_stage('Task 3B — Joint intrinsic/extrinsic recovery', [
        S3/'04_recover_intrinsics_and_extrinsics.py',
        '--results', s3out/'camera_robot_calibration.json',
        '--output', s3out/'recovered'], s3out/'recovered', pipeline)
    recovered = s3out/'recovered'/'camera_to_robot_transform_recovered.npz'
    validate_calibration(recovered)
    check_recovery_quality(s3out/'recovered'/'recovery_results.json')
    return recovered


def run_offline_evaluations(args, output, calibration, pipeline):
    call_stage('Task 4 — RGB localization and metrics', [
        S4/'task4_localization.py', '--camera', args.camera,
        '--robot-base', args.robot_base, '--calibration', calibration,
        '--trials', args.localization_trials, '--output', output/'task4'],
        output/'task4', pipeline)
    call_stage('Task 5 — RGB-D pose and metrics', [
        S5/'task5_pose_estimation.py', '--camera', args.camera,
        '--robot-base', args.robot_base, '--calibration', calibration,
        '--trials', args.pose_trials, '--output', output/'task5'],
        output/'task5', pipeline)


def run_live_sequence(args, output: Path, calibration: Path, pipeline):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from vgm.manipulation import (IKMotion, chain_joints, evaluate_object, find_tip,
                                  live_poses)
    from vgm.geometry import make_tcp_goal, way_above
    from vgm.perception import detect_from_rgb, load_calibration
    from vgm.scene import prepare_scene, read_bgr, set_block_world, world_to_base

    k, dist, B_C = load_calibration(calibration)
    client = RemoteAPIClient()
    sim = client.require('sim')
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError('Stop CoppeliaSim simulation before starting the unified pipeline.')
    base = sim.getObject(args.robot_base)
    camera = sim.getObject(args.camera)
    assert_calibration_matches_scene(sim,base,camera,calibration)
    # Safety: the kinematic-attachment demonstration must NEVER run with
    # the physics engine active on dynamically configured Panda links.
    try:
        physics_active = bool(sim.getBoolProperty(sim.handle_scene,'dynamicsEnabled'))
    except Exception:
        physics_active = bool(sim.getBoolParam(sim.boolparam_dynamics_handling_enabled))
    if physics_active:
        raise RuntimeError('Scene physics is enabled in a kinematic-only workcell. '
                           'Do not execute; rerun robot_stability.py --check with simulation stopped.')
    tip = find_tip(sim, base, args.tip)
    joints = chain_joints(sim,base,tip)
    if len(joints) != 7:
        raise RuntimeError('Expected seven Franka arm joints for this validated configuration.')
    print('TCP BASE xyz:', sim.getObjectPosition(tip,base))
    print('NOTE: TCP placement and orientation must be VISUALLY verified in your scene.')
    print('GRASP MODE: kinematic attachment proxy. No contact/force validation.')
    objects = prepare_scene(sim)
    destinations = load_destinations(args.destinations)
    start_layout = load_start_layout()
    colors = args.colors
    print('SEQUENTIAL PICK ORDER:', ' -> '.join(colors))
    print('INDIVIDUAL DESTINATIONS (WORLD m):', {c:destinations[c] for c in colors})
    if not args.keep_layout:
        validate_layout(start_layout, destinations, colors)
    else:
        print('WARNING: --keep-layout skips known-layout clearance check; inspect scene for occupied destinations.')

    # The scene INITIALIZATION uses known pose commands, but these positions
    # are not ever read/used as pick estimates: each pick is fresh RGB-D.
    if not args.keep_layout:
        for name, h in objects.items():
            x,y,z,yaw = start_layout[name]
            set_block_world(sim, h, x, y, yaw, z=z)
        print('Initialized camera-visible source layout (scene setup ONLY).')
    else:
        print('Preserving object positions from existing stopped scene.')

    if args.mode=='fresh':
        scene_after=output/'workcell_ready_for_runtime.ttt'
        sim.saveScene(str(scene_after))
        pipeline['scene_ready_for_runtime']=str(scene_after)
        print('Saved scene with initialized objects:', scene_after)

    result_root = output/'runtime'
    result_root.mkdir(exist_ok=True,parents=True)
    sim.setStepping(True)
    sim.startSimulation()
    motion=None
    records=[]
    started=True
    initial_plans={}
    try:
        for _ in range(3): sim.step()
        # Task 4 and Task 5 are performed LIVE before IK, independent of any
        # previously saved image/pose, in addition to optional evaluation stages.
        stage4_5_dir = result_root/'initial_perception'
        stage4_5_dir.mkdir(exist_ok=True)
        initial, diagnostics = live_poses(sim,camera,k,dist,B_C,stage4_5_dir)
        initial_rgb = read_bgr(sim,camera)
        boxes, _ = detect_from_rgb(initial_rgb)
        dump_json(stage4_5_dir/'localization.json',boxes)
        dump_json(stage4_5_dir/'pose_estimates.json',
            {name: (v.tolist() if v is not None else None) for name,v in initial.items()})
        print('Task 4 live RGB detection:', {c:boxes[c] is not None for c in colors})
        print('Task 5 live RGB-D poses:', {c:initial[c] is not None for c in colors})
        for color in colors:
            if boxes[color] is None or initial[color] is None:
                raise RuntimeError(f'{color}: no live 2D/3D detection, see {stage4_5_dir}')
        motion=IKMotion(client,sim,base,tip,joints)
        home_configuration = [float(sim.getJointPosition(j)) for j in joints]

        # FIRST preflight ALL requested colors, before any joint motion.
        for color in colors:
            pose=initial[color]
            yaw=yaw_of(pose)
            center=pose[:3,3]
            place_B=world_to_base(sim,base,destinations[color])
            grasp=make_tcp_goal(center,yaw,args.tcp_height)
            above_pick=way_above(grasp,args.lift_height)
            place=make_tcp_goal(place_B,yaw,args.tcp_height)
            above_place=way_above(place,args.lift_height)
            sequence=[('approach',above_pick),('grasp',grasp),('lift',above_pick),
                      ('transfer',above_place),('release',place),('retreat',above_place)]
            print(f'\n{color.upper()} preflight: image-estimated pick {np.round(center,4)} '
                  f'world-target {destinations[color]}')
            for label,T in sequence:
                try:
                    joints_candidate=motion.plan(T)
                    print(f'   PASS {label:9s} | joint 1 = {joints_candidate[0]:+.3f} rad')
                except Exception as exc:
                    raise RuntimeError(f'{color} {label} IK PRECHECK failed: {exc}') from exc
            initial_plans[color]={'estimated_pick_center_B_m':center.tolist(),
                                  'goal_center_B_m':place_B.tolist(),
                                  'all_six_waypoint_IK_solutions':True}
        dump_json(result_root/'ik_preflight.json',initial_plans)
        print('\nAll requested colors passed independent waypoint IK preflight.')
        if not args.execute:
            print('DRY RUN: NO joint targets sent, no objects reparented.')
            pipeline['runtime_status']='dry_run_ik_passed'
            return

        # Each subsequent pick estimates pose AGAIN, after the prior placement.
        for color in colors:
            trialdir=result_root/color
            trialdir.mkdir(exist_ok=True,parents=True)
            record={'color':color, 'uses_live_RGB_D_for_pick':True,
                    'uses_simulator_object_pose_for_planning':False,
                    'grasp_mode':'kinematic_parenting_proxy',
                    'collision_free_planning':False,
                    'status':'started','success':False}
            records.append(record)
            dump_json(trialdir/'trial_result.json',record)
            print(f'\n========== Live pick-and-place: {color} ==========')
            for _ in range(3): sim.step()
            measured,diag=live_poses(sim,camera,k,dist,B_C,trialdir)
            est=measured[color]
            if est is None:
                raise RuntimeError(f'{color} RGB-D perception failed: {diag[color]}')
            yaw=yaw_of(est)
            pick_center=est[:3,3].copy()
            place_center=world_to_base(sim,base,destinations[color])
            grasp=make_tcp_goal(pick_center,yaw,args.tcp_height)
            above_pick=way_above(grasp,args.lift_height)
            place=make_tcp_goal(place_center,yaw,args.tcp_height)
            above_place=way_above(place,args.lift_height)
            record.update({'estimated_pick_B_m':pick_center.tolist(),
                           'requested_place_B_m':place_center.tolist(),
                           'yaw_deg':math.degrees(yaw), 'perception_diagnostics':diag[color]})
            motion.move(above_pick,'approach '+color)
            motion.move(grasp,'grasp approach '+color)
            tip_actual=obj_pose_matrix(sim.getObjectPose(tip,base))
            if float(np.linalg.norm(tip_actual[:3,3]-grasp[:3,3])) > 0.02:
                raise RuntimeError('TCP not at planned grasp pose; no attachment performed')
            sim.setObjectParent(objects[color],tip,True)
            for _ in range(4): sim.step()
            motion.move(above_pick,'lift '+color)
            motion.move(above_place,'transfer '+color)
            motion.move(place,'release '+color)
            sim.setObjectParent(objects[color],-1,True)
            for _ in range(4): sim.step()
            record.update(evaluate_object(sim,objects[color],base,place_center,None,
                                          xy_tol_mm=args.xy_tol_mm,z_tol_mm=args.z_tol_mm,
                                          yaw_tol_deg=args.yaw_tol_deg))
            record['status']='executed'
            dump_json(trialdir/'trial_result.json',record)
            motion.move(above_place,'retreat '+color)
            if color != colors[-1]:
                # Clear the field of view for the next LIVE observation.
                # This is a joint-space return to the initial home pose,
                # not a proof of collision-free motion.
                motion.move_joints(home_configuration,'return home after '+color)
            print(f"{color} final XY placement error {record['placement_xy_error_mm']:.2f} mm; success={record['success']}")
        pipeline['runtime_status']='executed'
    except Exception as exc:
        pipeline['runtime_status']='failed'
        pipeline['runtime_error']=str(exc)
        print('RUNTIME ERROR:', exc, file=sys.stderr)
        raise
    finally:
        if motion is not None:
            try: motion.cleanup()
            except Exception as exc: print('IK cleanup warning:',exc)
        if started:
            try: sim.stopSimulation()
            except Exception as exc: print('Simulation stop warning:',exc)
        if records:
            with (result_root/'trials.csv').open('w',newline='') as fp:
                cols=['color','status','success','placement_error_mm','placement_xy_error_mm','placement_z_error_mm']
                writer=csv.DictWriter(fp,fieldnames=cols,extrasaction='ignore')
                writer.writeheader()
                writer.writerows(records)
            dump_json(result_root/'summary.json',{
                'trials':records,
                'n_attempted':len(records),
                'n_successful':sum(bool(r.get('success')) for r in records),
                'success_rate':sum(bool(r.get('success')) for r in records)/len(records),
                'grasp_mode':'kinematic_parenting_proxy',
                'tolerances':{'xy_mm':args.xy_tol_mm,'z_mm':args.z_tol_mm,'yaw_deg':args.yaw_tol_deg},
                'note':'One sequence is a demonstration, not a success rate. '
                       'Use --benchmark-trials or stages/task6/task6_benchmark.py.',
            })
        pipeline['stages'].append({'name':'Task 6 — continuous simulated pick/place',
                                   'status':pipeline.get('runtime_status'),
                                   'results':str(result_root)})
        dump_json(pipeline['manifest_path'],{k:v for k,v in pipeline.items() if k!='manifest_path'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['fresh','full','reuse','runtime'], default='fresh',
                   help='fresh: build a complete workcell from an empty scene, recalibrate and evaluate; full: recalibrate Task 2+3 in an existing scene; '
                        'reuse: use previous calibration and evaluate Task 4+5; '
                        'runtime: load previous calibration and go directly to fresh live vision + manipulation')
    p.add_argument('--franka-model',default='',
                   help='For fresh mode: path to installed Franka .ttm model (auto-search if omitted)')
    p.add_argument('--reset-scene',action='store_true',
                   help='For fresh mode: explicitly discard current scene, then rebuild; otherwise require a blank scene')
    p.add_argument('--tcp-offset',nargs=3,type=float,default=[0.,0.,.10],
                   metavar=('X','Y','Z'),help='Provisional TCP offset relative to joint 7 in m')
    p.add_argument('--tcp-verified',action='store_true',
                   help='Explicitly confirm the loaded TCP was visually aligned before --execute')
    p.add_argument('--robot-base',default='/Franka')
    p.add_argument('--camera',default='/Task2Camera')
    p.add_argument('--tip',default='',help='Full path of an already visually verified TCP dummy')
    p.add_argument('--calibration',type=Path,default=None,
                   help='Recovered .npz containing K and B_from_Ccv, used for reuse/runtime modes')
    p.add_argument('--calibration-views',type=int,default=30)
    p.add_argument('--extrinsic-views',type=int,default=24)
    p.add_argument('--localization-trials',type=int,default=30)
    p.add_argument('--pose-trials',type=int,default=30)
    p.add_argument('--colors',nargs='+',choices=['red','green','blue'],default=['red'],
                   help='Color order, e.g. --colors red green blue performs sequential picks')
    p.add_argument('--destinations',type=Path,default=None,
                   help='JSON mapping colors to WORLD XYZ placement centres (default: config/destinations.json)')
    p.add_argument('--keep-layout',action='store_true',
                   help='Keep current stopped-scene block poses, instead of staging known initial locations')
    p.add_argument('--tcp-height',type=float,default=.09)
    p.add_argument('--lift-height',type=float,default=.18)
    p.add_argument('--execute',action='store_true',
                   help='Explicitly enable joint motion and kinematic grasp proxy; default only checks IK')
    p.add_argument('--xy-tol-mm',type=float,default=DEFAULT_XY_TOL_MM,help='Placement success: XY tolerance')
    p.add_argument('--z-tol-mm',type=float,default=DEFAULT_Z_TOL_MM,help='Placement success: Z tolerance')
    p.add_argument('--yaw-tol-deg',type=float,default=DEFAULT_YAW_TOL_DEG,
                   help='Placement success: yaw tolerance (only when a target yaw is given)')
    p.add_argument('--benchmark-trials',type=int,default=0,
                   help='After the live sequence, run N randomized Task 6 benchmark trials (requires --execute)')
    p.add_argument('--benchmark-closed-loop',action='store_true',
                   help='Benchmark also runs the look-then-move visual-servo correction (bonus)')
    p.add_argument('--output',type=Path,default=ROOT/'runs')
    args=p.parse_args()
    if len(set(args.colors))!=len(args.colors): p.error('Do not repeat colors')
    if args.tcp_height < .025 or args.lift_height < .05: p.error('Unreasonable TCP offset or lift height')
    if args.execute and not args.tcp_verified:
        p.error('--execute requires --tcp-verified. First build/inspect the new scene and visually verify T6_TCP before moving the arm.')
    if args.reset_scene and args.mode!='fresh':
        p.error('--reset-scene is only supported with --mode fresh')
    if args.calibration_views<15 or args.extrinsic_views<12:
        p.error('Need >=15 checkerboard views and >=12 extrinsic views')
    runid=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=args.output.resolve()/runid
    out.mkdir(parents=True,exist_ok=False)
    pipeline={'pipeline_version':'4.0-physics-disabled-kinematic-proxy-tricolor',
              'mode':args.mode, 'execution_requested':bool(args.execute),
              'task1':'Literature review and technical selection in LiteratureReview.md',
              'grasp_mode':'kinematic_parenting_proxy',
              'all_objects_true_poses_used_for_planning':False,
              'calibration_policy':'estimated matrices from images/landmarks; simulator ground truth for evaluation only',
              'stages':[], 'manifest_path':out/'pipeline_manifest.json'}
    pipeline['stages'].append({'name':'Task 1 — design and literature','status':'documented','file':str(ROOT/'LiteratureReview.md')})
    dump_json(pipeline['manifest_path'],{k:v for k,v in pipeline.items() if k!='manifest_path'})
    try:
        if args.mode == 'fresh':
            # Build the SCENE, not only file-based calibration. A prior .ttt scene
            # is never needed in this mode, and previously estimated extrinsics
            # are NEVER reused for the newly positioned camera and robot.
            from prepare_fresh_scene import build_new_scene
            scene_path = out/'new_workcell_before_calibration.ttt'
            meta = build_new_scene(model_path=args.franka_model,
                                   reset_scene=args.reset_scene,
                                   save_path=scene_path,
                                   tcp_offset=args.tcp_offset)
            dump_json(out/'scene_setup.json',meta)
            if not args.tip:
                args.tip=meta['tcp_alias']
            pipeline['stages'].append({'name':'New scene — Franka, camera, checkerboard and TCP',
                                       'status':'created', 'scene':str(scene_path),
                                       'robot_model':meta['model_file']})
            pipeline['scene_file']=str(scene_path)
            dump_json(pipeline['manifest_path'],
                      {k:v for k,v in pipeline.items() if k!='manifest_path'})
        if args.mode != 'fresh':
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            from robot_stability import hold_arm_kinematically, preflight_idle_hold
            sim_hold = RemoteAPIClient().require('sim')
            base_hold = sim_hold.getObject(args.robot_base)
            joints_hold, q_hold = hold_arm_kinematically(sim_hold,base_hold)
            preflight_idle_hold(sim_hold,joints_hold,q_hold)
        if args.mode in ('full','fresh'):
            cal=run_calibration(args,out,pipeline)
            dump_json(ROOT/'latest_calibration.json',{
                'calibration_file':str(cal.resolve()),
                'scene_setup_file':str(out/'scene_setup.json') if args.mode=='fresh' else None,
                'run_dir':str(out)})
        else:
            cal=args.calibration.resolve() if args.calibration else default_existing_calibration()
            validate_calibration(cal)
            quality_file = cal.parent/'recovery_results.json'
            if quality_file.is_file():
                check_recovery_quality(quality_file, strict=True)
            else:
                print('NOTE: No recovery_results.json beside reused calibration; numeric shape validated, '
                      'but re-check accuracy against your saved Task 3 results before execution.')
            pipeline['stages'].append({'name':'Tasks 2–3 — validated calibration loaded', 'file':str(cal),
                                       'note':'No calibration refit in reuse/runtime mode'})
        pipeline['calibration_used']=str(cal)
        if args.mode in ('fresh','full','reuse'):
            run_offline_evaluations(args,out,cal,pipeline)
        run_live_sequence(args,out,cal,pipeline)
        if args.benchmark_trials and args.execute:
            argv=[S6/'task6_benchmark.py','--camera',args.camera,'--robot-base',args.robot_base,
                  '--calibration',cal,'--trials',args.benchmark_trials,'--tcp-verified',
                  '--xy-tol-mm',args.xy_tol_mm,'--z-tol-mm',args.z_tol_mm,'--yaw-tol-deg',args.yaw_tol_deg,
                  '--output',out/'task6_benchmark']
            if args.tip: argv+=['--tip',args.tip]
            if args.destinations: argv+=['--destinations',args.destinations]
            if args.benchmark_closed_loop: argv+=['--closed-loop']
            call_stage('Task 6 — randomized multi-trial benchmark',argv,out/'task6_benchmark',pipeline)
        elif args.benchmark_trials:
            print('NOTE: --benchmark-trials ignored without --execute.')
        print(f'\nPIPELINE COMPLETE. Results: {out}')
    except Exception as exc:
        pipeline['pipeline_error']=str(exc)
        print(f'PIPELINE STOPPED: {exc}\nResults: {out}',file=sys.stderr)
        raise
    finally:
        dump_json(pipeline['manifest_path'],{k:v for k,v in pipeline.items() if k!='manifest_path'})


if __name__=='__main__':
    main()
