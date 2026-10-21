"""Task 3: eye-to-hand calibration from known 3-D colored landmarks (v2).

Estimate robot_base_T_opencv_camera using images and known robot_base_T_target.
Camera pose from sim.getObjectPose is used ONLY in final validation.

Requirements: Task2 /Task2Camera and Task2 calibration .npz (same intrinsics);
real robot base loaded as an object /Panda (or adjust --robot-base).

Example:
 python 03_calibrate_camera_robot.py --robot-base /Panda \
   --intrinsics ../vision_calibration_task2/dataset/calibration_results/camera_calibration.npz

For a pipeline smoke test without the Franka model only:
 python 03_calibrate_camera_robot.py --demo-base ...
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

# In CoppeliaSim vision-sensor coordinates x=left, y=up, z=forward;
# in OpenCV optical coordinates x=right, y=down, z=forward.
# Rotation is 180 degrees about the common optical Z axis.
SIM_CAMERA_FROM_CV = np.eye(4)
SIM_CAMERA_FROM_CV[:3, :3] = np.diag([-1., -1., 1.])

# 8 uniquely colored, NONCOPLANAR landmarks relative to the target rig (metres).
# Hues are OpenCV units (0...179). Distinct marker IDs avoid checkerboard
# 180-degree corner-order ambiguity and depth points improve PnP conditioning.
LANDMARKS = [
    ('red',     (1.0, 0.04, 0.04),   0, (-.145, -.085, .028)),
    ('orange',  (1.0, 0.46, 0.01),  14, ( .135, -.083, .042)),
    ('yellow',  (1.0, 1.00, 0.02),  30, ( .125,  .090, .038)),
    ('green',   (0.02, 1.0, 0.02),  60, (-.140,  .087, .034)),
    ('cyan',    (0.02, 1.0, 1.00),  90, (-.050, -.035, .125)),
    ('blue',    (0.03, 0.05, 1.0), 120, ( .048,  .044, .115)),
    ('violet',  (0.55, 0.04, 1.0), 137, (-.052,  .054, .072)),
    ('magenta', (1.0, 0.03, 0.76), 157, ( .085, -.035, .077)),
]


def mat_from_pose(pose):
    """CoppeliaSim pose: [x,y,z,qx,qy,qz,qw]."""
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(np.asarray(pose[3:], float)).as_matrix()
    result[:3, 3] = np.asarray(pose[:3], float)
    return result


def target_pose(rng, attempt, center=(0., 0., 0.2)):
    """Known target pose in WORLD frame, centered in Task-2 field of view.

    This deliberately does not depend on the robot-base position. The robot
    base pose in world is known calibration-fixture geometry (not camera GT).
    """
    if attempt == 0:
        return [*map(float, center), 0., 0., 0., 1.]
    xyz = [float(center[0] + rng.uniform(-.07, .07)),
           float(center[1] + rng.uniform(-.07, .07)),
           float(center[2] + rng.uniform(-.035, .035))]
    angles = [rng.uniform(-12, 12), rng.uniform(-12, 12), rng.uniform(-18, 18)]
    quat = Rotation.from_euler('xyz', angles, degrees=True).as_quat()
    return xyz + quat.tolist()


def pose_from_mat(mat):
    return [*mat[:3, 3].tolist(),
            *Rotation.from_matrix(mat[:3, :3]).as_quat().tolist()]


def scout_target_visibility(sim, camera, rig, out, center_override=None):
    """Try known WORLD coordinates from Task 2; no camera true pose queried.

    Save raw images with detection overlays for diagnosis. Use a visible pose
    before calibrating, instead of generating hundreds of invisible poses.
    """
    if center_override is not None:
        candidates = [tuple(center_override)]
    else:
        candidates = [(x, y, z)
                      for z in (0.20, 0.36)
                      for x, y in ((0., 0.), (.22, 0.), (-.22, 0.),
                                   (0., .20), (0., -.20), (.22, .20),
                                   (-.22, -.20), (.22, -.20), (-.22, .20))]
    best_count, best_center, best_raw = -1, None, None
    for idx, center in enumerate(candidates):
        sim.setObjectPose(rig, target_pose(None, 0, center), -1)
        sim.step()
        sim.step()
        raw = read_bgr(sim, camera)
        hits, annotated = find_landmarks(raw)
        print(f'Scout {idx + 1:02d}/{len(candidates)}: world center {center}, '
              f'colored landmarks {len(hits)}/8')
        if len(hits) > best_count:
            best_count, best_center, best_raw = len(hits), center, raw.copy()
            cv2.imwrite(str(out/'first_frame_debug.png'), annotated)
            cv2.imwrite(str(out/'first_frame_raw.png'), raw)
        if len(hits) >= 7:
            break
    if best_count < 6:
        raise RuntimeError(
            f'No visible target pose; maximum {best_count}/8 colored markers. '
            f'Inspect {out / "first_frame_raw.png"} and first_frame_debug.png. '
            'Either the camera sees no target, Franka blocks it, or HSV segmentation '
            'does not match the rendered colors. Reposition Franka or pass '
            '--target-center-world X Y Z to a clearly visible area. '
            'Do not lower the 6-marker PnP requirement.')
    print(f'Visibility scout selected WORLD center {best_center} '
          f'with {best_count}/8 visible landmarks')
    return best_center


def read_bgr(sim, camera):
    raw, wh = sim.getVisionSensorImg(camera)
    w, h = map(int, wh)
    buff = np.frombuffer(raw, dtype=np.uint8)
    if buff.size != 3*w*h:
        raise RuntimeError(f'Camera buffer size {buff.size} != {3*w*h}.')
    return cv2.cvtColor(cv2.flip(buff.reshape(h, w, 3), 0), cv2.COLOR_RGB2BGR)


def create_color_target(sim):
    # Re-running --mode full after a failed calibration should reuse the
    # existing target, not create duplicate colored markers in the camera view.
    try:
        existing = sim.getObject('/Task3CalibrationRig')
        n_shapes = len(sim.getObjectsInTree(existing, sim.object_shape_type, 0))
        if n_shapes != len(LANDMARKS):
            raise RuntimeError(f'Existing Task3CalibrationRig has {n_shapes} shapes, '
                               f'expected {len(LANDMARKS)}. Reopen a clean scene.')
        print(f'Reusing existing /Task3CalibrationRig ({n_shapes} landmarks).')
        return existing
    except Exception as exc:
        # A missing object is normal in a fresh scene. A malformed partial rig
        # must fail explicitly rather than silently creating duplicate objects.
        if 'Existing Task3CalibrationRig has' in str(exc):
            raise
    rig = sim.createDummy(.02)
    sim.setObjectAlias(rig, 'Task3CalibrationRig')
    for name, color, _, xyz in LANDMARKS:
        obj = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [.030, .030, .030], 0)
        sim.setObjectAlias(obj, 'T3_' + name)
        sim.setShapeColor(obj, '', sim.colorcomponent_ambient_diffuse, list(color))
        # Pure colors under different lighting should remain recognizable.
        sim.setShapeColor(obj, '', sim.colorcomponent_emission,
                          [float(x)*.45 for x in color])
        sim.setObjectParent(obj, rig, False)
        sim.setObjectPose(obj, [*xyz, 0., 0., 0., 1.], rig)
    return rig


def circular_hue_mask(h, hue, tolerance=8):
    delta = np.abs(h.astype(np.int16) - int(hue))
    return np.minimum(delta, 180 - delta) <= tolerance


def find_landmarks(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hits = {}
    marks = image.copy()
    for name, _, hue, _ in LANDMARKS:
        binary = (circular_hue_mask(h, hue, tolerance=8) &
                  (s > 90) & (v > 65)).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  np.ones((3, 3), np.uint8))
        components, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
        candidates = [(stats[i, cv2.CC_STAT_AREA], centers[i]) for i in range(1, components)
                      if 18 <= stats[i, cv2.CC_STAT_AREA] <= 18000]
        if not candidates:
            continue
        area, (u, vv) = max(candidates, key=lambda t: t[0])
        hits[name] = [float(u), float(vv)]
        cv2.circle(marks, (round(u), round(vv)), 5, (255, 255, 255), 1)
        cv2.putText(marks, name, (round(u)+6, round(vv)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return hits, marks


def fit_camera_from_rig(hits, K, dist):
    xyz, uv = [], []
    for name, _, _, point in LANDMARKS:
        if name in hits:
            xyz.append(point)
            uv.append(hits[name])
    if len(xyz) < 6:
        raise ValueError(f'Only {len(xyz)} visible markers; need at least 6.')
    xyz = np.ascontiguousarray(np.array(xyz, np.float64))
    uv = np.ascontiguousarray(np.array(uv, np.float64))
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        xyz, uv, K, dist, iterationsCount=350,
        reprojectionError=6.0, confidence=.999, flags=cv2.SOLVEPNP_EPNP
    )
    if not ok or inliers is None or len(inliers) < 6:
        raise ValueError('PnP RANSAC failed to find 6 consistent landmarks.')
    idx = inliers.ravel()
    ok, rvec, tvec = cv2.solvePnP(
        xyz[idx], uv[idx], K, dist, rvec, tvec, True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise ValueError('PnP refinement failed.')
    projected, _ = cv2.projectPoints(xyz[idx], rvec, tvec, K, dist)
    residuals = projected.reshape(-1, 2) - uv[idx]
    rms = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    C_from_Q = np.eye(4)
    C_from_Q[:3, :3] = cv2.Rodrigues(rvec)[0]
    C_from_Q[:3, 3] = tvec.ravel()
    return C_from_Q, rms, len(idx)


def angle_deg(R1, R2):
    return float(np.rad2deg((Rotation.from_matrix(R1).inv() *
                             Rotation.from_matrix(R2)).magnitude()))


def fuse_robot_from_cv_camera(transforms):
    """Robust fixed-camera calibration: medoid -> reject outliers -> Lie rotation mean."""
    if len(transforms) < 5:
        raise ValueError('Need >= 5 successful training poses.')
    n = len(transforms)
    d = np.zeros((n, n), float)
    for i in range(n):
        for j in range(i + 1, n):
            t = np.linalg.norm(transforms[i][:3, 3] - transforms[j][:3, 3])
            r = np.deg2rad(angle_deg(transforms[i][:3, :3], transforms[j][:3, :3]))
            d[i, j] = d[j, i] = t + .20*r
    medoid = int(np.argmin(np.median(d, axis=1)))
    anchor = transforms[medoid]
    trans_err = np.array([np.linalg.norm(m[:3, 3]-anchor[:3, 3]) for m in transforms])
    rot_err = np.array([angle_deg(m[:3, :3],anchor[:3, :3]) for m in transforms])
    threshold_t = max(.025, 3 * float(np.median(trans_err)))
    threshold_r = max(5.0, 3 * float(np.median(rot_err)))
    keep = np.flatnonzero((trans_err <= threshold_t) & (rot_err <= threshold_r))
    if len(keep) < 5:
        keep = np.argsort(d[medoid])[:max(5, n//2)]
    merged = np.eye(4)
    merged[:3, 3] = np.mean([transforms[i][:3, 3] for i in keep], axis=0)
    merged[:3, :3] = Rotation.from_matrix(
        np.stack([transforms[i][:3, :3] for i in keep])
    ).mean().as_matrix()
    return merged, keep.tolist()


def holdout_error(B_from_C, frames, K, dist):
    per_view = []
    C_from_B = np.linalg.inv(B_from_C)
    for f in frames:
        obj_pts = []
        measured = []
        B_from_Q = mat_from_pose(f['target_pose_in_robot_base'])
        for name, _, _, pos in LANDMARKS:
            if name not in f['detections']:
                continue
            p = B_from_Q @ np.array([*pos, 1.])
            xyz_C = C_from_B @ p
            obj_pts.append(xyz_C[:3])
            measured.append(f['detections'][name])
        if len(obj_pts) < 6:
            continue
        projected, _ = cv2.projectPoints(
            np.asarray(obj_pts, float), np.zeros(3), np.zeros(3), K, dist
        )
        uv = projected.reshape(-1, 2)
        measured = np.asarray(measured, float)
        rms = float(np.sqrt(np.mean(np.sum((uv - measured)**2, axis=1))))
        per_view.append(rms)
    return per_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-base', default='/Panda',
                        help='Actual manipulator BASE alias, e.g. /Panda')
    parser.add_argument('--camera', default='/Task2Camera')
    parser.add_argument('--intrinsics', type=Path, required=True,
                        help='Task2 camera_calibration.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'task3_results')
    parser.add_argument('--views', type=int, default=24,
                        help='Total accepted poses; >=12 recommended')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--target-center-world', nargs=3, type=float, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='Optional target center in WORLD metres; default scans around Task2 board area')
    parser.add_argument('--collect-only', action='store_true',
                        help='Save 3D-to-2D landmark correspondences without rejecting views via PnP. '
                             'The following recovery stage fits intrinsics and camera-to-robot pose jointly.')
    parser.add_argument('--demo-base', action='store_true',
                        help='Create reference dummy when Panda not loaded (not real robot calibration)')
    args = parser.parse_args()
    if args.views < 12:
        parser.error('Capture >=12 accepted poses to enable useful held-out testing.')
    if not args.intrinsics.is_file():
        parser.error(f'Intrinsics file missing: {args.intrinsics}')
    data = np.load(args.intrinsics)
    K = np.array(data['K_constrained'] if 'K_constrained' in data else
                 data['K_pinhole'] if 'K_pinhole' in data else data['K'], float)
    dist = np.array(data['dist_constrained'] if 'dist_constrained' in data else
                    data['dist_pinhole'] if 'dist_pinhole' in data else data['dist'], float)
    width, height = int(data['width']), int(data['height'])
    if not np.all(np.isfinite(K)) or K.shape != (3, 3):
        parser.error('Invalid Task2 K matrix.')
    out = args.output.resolve()
    images = out/'images'
    images.mkdir(parents=True, exist_ok=True)

    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client = RemoteAPIClient()
        sim = client.require('sim')
        camera = sim.getObject(args.camera)
        if args.demo_base:
            base = sim.createDummy(.035)
            sim.setObjectAlias(base, 'Task3_Demo_Robot_Base')
            sim.setObjectPose(base, [-.18, 0., 0., 0., 0., 0., 1.])
        else:
            base = sim.getObject(args.robot_base)
    except Exception as exc:
        sys.exit('Could not connect or find camera/robot base. Open saved Task2 scene, '
                 'load a Panda model, and pass its root alias with --robot-base. '
                 f'Error: {exc}')

    if args.demo_base:
        print('WARNING: --demo-base is a pipeline check, not calibration against a real robot model.')
    print('Using Task2 IMAGE-estimated intrinsic matrix:\n', K)
    print('Optical-frame convention: OpenCV x-right, y-down, z-forward.')
    f_nominal = max(width, height) / (2. * np.tan(np.deg2rad(60.) / 2.))
    if (abs(K[0, 0] / f_nominal - 1.) > .25 or
            abs(K[1, 1] / f_nominal - 1.) > .25 or
            abs(K[0, 2] - width/2) > .10*width or
            abs(K[1, 2] - height/2) > .10*height):
        print('WARNING: Task2 fitted intrinsics are far from the 60-degree ideal camera reference. ' 
              'This does not cause missing colored blobs, but PnP could be wrong. ' 
              'Check/recalibrate Task2 intrinsics before trusting Task3 results.')
    # Board from Task2 would otherwise hide the colored target.
    try:
        old_board = sim.getObject('/Task2Checkerboard')
        sim.setObjectPose(old_board, [2., 2., 0., 0., 0., 0., 1.])
    except Exception:
        pass
    # Known fixed robot-base placement in WORLD; NOT camera ground truth.
    W_from_B = mat_from_pose(sim.getObjectPose(base, -1))
    B_from_W = np.linalg.inv(W_from_B)
    rig = create_color_target(sim)
    rng = np.random.default_rng(args.seed)
    records = []
    sim.setStepping(True)
    sim.startSimulation()
    try:
        center = scout_target_visibility(sim, camera, rig, out, args.target_center_world)
        for attempt in range(max(args.views*7, 100)):
            if len(records) >= args.views:
                break
            # Known mechanical/commanded target pose in robot base coordinates:
            # sim.getObjectPose(camera,...) is NEVER called inside this loop.
            pose_W_Q = target_pose(rng, attempt, center)
            pose_B_Q = pose_from_mat(B_from_W @ mat_from_pose(pose_W_Q))
            sim.setObjectPose(rig, pose_W_Q, -1)
            sim.step()
            sim.step()
            img = read_bgr(sim, camera)
            if (img.shape[1], img.shape[0]) != (width, height):
                raise ValueError('Camera resolution changed since Task2. Recalibrate intrinsics.')
            hits, annotated = find_landmarks(img)
            if attempt == 0:
                cv2.imwrite(str(out/'first_frame_debug.png'), annotated)
            if len(hits) < 6:
                print(f'Attempt {attempt+1}: only {len(hits)} colored landmarks detected')
                if attempt == 0:
                    cv2.imwrite(str(out/'first_frame_debug.png'), img)
                continue
            if args.collect_only:
                # IMPORTANT: collect measured correspondences independently from the
                # faulty Task 2 K. The subsequent joint fitting stage recovers both
                # K and B_T_Ccv directly from these images and known target poses.
                # Do not fabricate a PnP solution or use simulator-camera ground truth.
                idx = len(records)
                cv2.imwrite(str(images/f'view_{idx:03d}.png'), annotated)
                records.append({'view_index': idx,
                                'target_pose_in_robot_base': pose_B_Q,
                                'detections': hits,
                                'pnp_inlier_count': None,
                                'pnp_reprojection_rms_px': None,
                                'B_from_C_estimate': None})
                print(f'Recorded {idx+1:02d}/{args.views} | '
                      f'visible image landmarks {len(hits)}/8 | '
                      'PnP intentionally deferred')
                continue
            try:
                C_from_Q, pnp_err, n_inliers = fit_camera_from_rig(hits, K, dist)
            except (ValueError, cv2.error) as exc:
                print(f'Attempt {attempt+1}: {exc}')
                if attempt == 0:
                    cv2.imwrite(str(out/'first_frame_debug.png'), annotated)
                continue
            # B_T_Ccv from commanded B_T_Q and image-estimated Ccv_T_Q.
            B_from_C = mat_from_pose(pose_B_Q) @ np.linalg.inv(C_from_Q)
            idx = len(records)
            cv2.imwrite(str(images/f'view_{idx:03d}.png'), annotated)
            records.append({'view_index': idx,
                            'target_pose_in_robot_base': pose_B_Q,
                            'detections': hits,
                            'pnp_inlier_count': n_inliers,
                            'pnp_reprojection_rms_px': pnp_err,
                            'B_from_C_estimate': B_from_C.tolist()})
            print(f'Accepted {idx+1:02d}/{args.views} | landmarks {len(hits)} '
                  f'| PnP reprojection RMS {pnp_err:.2f} px')
        # Freeze target in a good accessible visible configuration at the end.
        sim.setObjectPose(rig, target_pose(np.random.default_rng(args.seed), 0, center), -1)
        sim.step()
    finally:
        sim.stopSimulation()
    if len(records) < 12:
        raise RuntimeError('Fewer than 12 usable landmark views. Check first_frame_debug.png, '
                           'camera field of view, colored target visibility, base pose and lighting.')

    if args.collect_only:
        # Ground truth is retained only for post-fit validation in stage 3B;
        # it plays no role in stage 3B residuals/optimization or grasp planning.
        sim_truth = mat_from_pose(sim.getObjectPose(camera, base))
        summary = {
            'status': 'Image-only landmark observations for independent joint calibration',
            'robot_base': args.robot_base,
            'calibration_target': '8 uniquely colored non-coplanar marker centers',
            'camera_alias': args.camera,
            'intrinsics_file': str(args.intrinsics.resolve()),
            'views_accepted': len(records),
            'optimization_deferred_to': '04_recover_intrinsics_and_extrinsics.py',
            'simulator_ground_truth_USED_ONLY_FOR_VALIDATION': {
                'B_from_Csim_ground_truth': sim_truth.tolist(),
            },
            'records': records,
        }
        with (out/'camera_robot_calibration.json').open('w') as f:
            json.dump(summary, f, indent=2)
        print(f'Collected {len(records)} usable target views without PnP filtering.')
        print('Next: run 04_recover_intrinsics_and_extrinsics.py to fit K and eye-to-hand pose.')
        return

    # Split by pose; never use held-out detections in the fitted transform.
    shuffle = np.random.default_rng(12345).permutation(len(records))
    n_val = max(3, round(len(records) * .25))
    val_ids, train_ids = shuffle[:n_val], shuffle[n_val:]
    views = [np.array(records[i]['B_from_C_estimate']) for i in train_ids]
    B_from_Ccv, kept_local_ids = fuse_robot_from_cv_camera(views)
    kept_ids = [int(train_ids[i]) for i in kept_local_ids]
    B_from_Csim = B_from_Ccv @ SIM_CAMERA_FROM_CV
    errors = holdout_error(B_from_Ccv, [records[int(i)] for i in val_ids], K, dist)

    # STRICTLY VALIDATION-ONLY: query the actual CoppeliaSim camera pose now,
    # AFTER camera-to-base estimation/fusion and heldout reprojection metrics.
    sim_truth = mat_from_pose(sim.getObjectPose(camera, base))
    translation_error_mm = 1000*float(np.linalg.norm(
        B_from_Csim[:3, 3] - sim_truth[:3, 3]))
    rotation_error_deg = angle_deg(B_from_Csim[:3, :3], sim_truth[:3, :3])
    summary = {
        'status': 'CoppeliaSim experiment; all estimation uses image detections and commanded target poses',
        'robot_base': 'demo dummy (NOT real robot)' if args.demo_base else args.robot_base,
        'calibration_target': '8 colored non-coplanar landmark centers with measured design coordinates (metres)',
        'camera_alias': args.camera,
        'intrinsics_file': str(args.intrinsics.resolve()),
        'optical_frame': 'OpenCV x right, y down, z forward',
        'coppeliasim_sensor_frame': 'x left, y up, z forward',
        'views_accepted': len(records),
        'training_view_ids': train_ids.tolist(),
        'training_inlier_view_ids': kept_ids,
        'heldout_view_ids': val_ids.tolist(),
        'B_from_Ccv_estimated': B_from_Ccv.tolist(),
        'B_from_Csim_estimated': B_from_Csim.tolist(),
        'heldout_projection_rms_px_per_view': errors,
        'heldout_projection_mean_rms_px': float(np.mean(errors)),
        'simulator_ground_truth_USED_ONLY_FOR_VALIDATION': {
            'B_from_Csim_ground_truth': sim_truth.tolist(),
            'translation_error_mm': translation_error_mm,
            'rotation_error_deg': rotation_error_deg,
        },
        'records': records,
    }
    with (out/'camera_robot_calibration.json').open('w') as f:
        json.dump(summary, f, indent=2)
    np.savez(out/'camera_to_robot_transform.npz', B_from_Ccv=B_from_Ccv,
             B_from_Csim=B_from_Csim, K=K, dist=dist)
    plt.figure(figsize=(8,4))
    plt.bar(range(len(errors)), errors)
    plt.xticks(range(len(errors)))
    plt.xlabel('Independent validation pose')
    plt.ylabel('Pixel reprojection RMS')
    plt.title('Task 3 — holdout 3D-to-2D consistency')
    plt.tight_layout()
    plt.savefig(out/'holdout_projection_rms.png', dpi=180)
    plt.close()

    print('\n=== Estimated robot_base_T_OpenCV_camera ===\n', B_from_Ccv)
    print('\n=== Estimated robot_base_T_CoppeliaSim_sensor ===\n', B_from_Csim)
    print('\nHeld-out projection mean RMS:', round(float(np.mean(errors)), 3), 'px')
    print('GROUND TRUTH VALIDATION ONLY — Translation:', round(translation_error_mm, 3), 'mm')
    print('GROUND TRUTH VALIDATION ONLY — Rotation:', round(rotation_error_deg, 4), 'degrees')
    print('Results saved to:', out)
    print('Save your CoppeliaSim scene (File > Save scene) for Task 4.')


if __name__ == '__main__':
    main()
