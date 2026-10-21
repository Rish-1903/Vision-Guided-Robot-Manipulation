"""Robot-side helpers: TCP discovery, simIK planning and placement evaluation.

Kinematic demonstration: the scene's physics engine is disabled and a grasp is
a kinematic re-parenting of the block to the TCP dummy.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import T_to_pose, obj_pose_matrix, symmetric_yaw_error_deg, yaw_of
from .perception import COLORS, detect_from_rgb, draw_axes, estimate_pose_from_mask
from .scene import read_bgr, read_depth_m

# Default placement acceptance: tight enough to be meaningful for an
# 82 x 60 x 38 mm block. Override from the command line if needed.
DEFAULT_XY_TOL_MM = 10.0
DEFAULT_Z_TOL_MM = 5.0
DEFAULT_YAW_TOL_DEG = 5.0


def chain_joints(sim, base, tip):
    """Joint handles from base to tip; refuse disconnected paths."""
    found, h, n = [], tip, 0
    while h != -1 and h != base and n < 100:
        if sim.getObjectType(h) == sim.object_joint_type:
            found.append(h)
        h = sim.getObjectParent(h)
        n += 1
    if h != base:
        raise RuntimeError('TCP dummy is not a descendant of the robot base; fix its parent.')
    found.reverse()
    if len(found) < 7:
        raise RuntimeError(f'TCP has only {len(found)} ancestor joints (need >= 7).')
    if len(found) > 8:
        raise RuntimeError(f'TCP has {len(found)} joints in its chain; a finger tip was probably '
                           'selected. Put the TCP dummy on the hand, before the finger joints.')
    return found


def find_tip(sim, base, explicit=None):
    if explicit:
        return sim.getObject(explicit)
    candidates = []
    for h in sim.getObjectsInTree(base, sim.handle_all, 0):
        if sim.getObjectType(h) != sim.object_dummy_type:
            continue
        alias = str(sim.getObjectAlias(h)).lower()
        if not any(x in alias for x in ('tcp', 'iktip', 'grasp_tip', 'tool_tip', 'flange_tip')):
            continue
        try:
            candidates.append((h, str(sim.getObjectAlias(h)), len(chain_joints(sim, base, h))))
        except RuntimeError:
            pass
    if len(candidates) != 1:
        print('Possible tip candidates:', candidates)
        raise RuntimeError('Pass --tip /FULL/PATH to the TCP dummy; no unique tip was found.')
    print('Selected TCP dummy:', candidates[0])
    return candidates[0][0]


def index_map(mapping, sim_handle):
    """ZeroMQ maps may contain numeric or stringified object-handle keys."""
    if sim_handle in mapping:
        return mapping[sim_handle]
    if str(sim_handle) in mapping:
        return mapping[str(sim_handle)]
    raise KeyError(f'Could not map scene handle {sim_handle} to an IK-world handle')


class IKMotion:
    """simIK configuration search + deterministic joint-space interpolation."""

    def __init__(self, client, sim, base, tip, joints, precision_m: float = 0.006,
                 precision_deg: float = 6.0):
        self.sim, self.base, self.tip, self.joints = sim, base, tip, joints
        self.ik = client.require('simIK')
        self.target = sim.createDummy(0.012)
        sim.setObjectAlias(self.target, 'Task6IKTarget')
        sim.setObjectPose(self.target, sim.getObjectPose(tip, base), base)
        self.env = self.ik.createEnvironment()
        self.group = self.ik.createGroup(self.env)
        el, m, _ = self.ik.addElementFromScene(self.env, self.group, base, tip, self.target,
                                               self.ik.constraint_pose)
        self.ik.setElementPrecision(self.env, self.group, el, [precision_m, math.radians(precision_deg)])
        self.ik.setGroupCalculation(self.env, self.group, self.ik.method_damped_least_squares, 0.06, 80)
        self.ik_joints = [index_map(m, j) for j in joints]
        self.ik_target = index_map(m, self.target)
        self.ik_base = index_map(m, self.base)

    def joint_positions(self):
        return np.asarray([self.sim.getJointPosition(j) for j in self.joints], float)

    def plan(self, goal):
        self.ik.syncFromSim(self.env, [self.group])
        self.ik.setObjectPose(self.env, self.ik_target, T_to_pose(goal), self.ik_base)
        configs = self.ik.findConfigs(self.env, self.group, self.ik_joints,
                                      {'maxDist': 2.0, 'maxTime': 2.0, 'findMultiple': False})
        if not configs:
            raise RuntimeError('No IK solution: waypoint outside the workspace, or TCP axes/offset wrong.')
        return np.asarray(configs[0], float)

    def move_joints(self, q_goal, label='joint target', resolution=.012, quiet=False):
        """Smoothstep joint interpolation. No collision checking."""
        q0, q1 = self.joint_positions(), np.asarray(q_goal, float)
        if q1.shape != q0.shape:
            raise ValueError('Expected one target per arm joint')
        n = max(30, int(np.max(np.abs(q1 - q0)) / resolution) + 1)
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)
            for j, angle in zip(self.joints, (1 - a) * q0 + a * q1):
                self.sim.setJointPosition(j, float(angle))
                self.sim.setJointTargetPosition(j, float(angle))
            self.sim.step()
        final = self.joint_positions()
        if float(np.max(np.abs(final - q1))) > .03:
            raise RuntimeError(f'{label}: joint tracking failed; a model script may override joints.')
        if not quiet:
            print(f'  {label} reached: max joint error {np.max(np.abs(final - q1)):.5f} rad')
        return final

    def move(self, goal, label='waypoint', resolution=0.012, quiet=False):
        self.move_joints(self.plan(goal), label, resolution=min(.012, resolution), quiet=True)
        tpose = obj_pose_matrix(self.sim.getObjectPose(self.tip, self.base))
        err = float(np.linalg.norm(tpose[:3, 3] - goal[:3, 3]))
        angle = math.degrees(Rotation.from_matrix(tpose[:3, :3].T @ goal[:3, :3]).magnitude())
        if err >= 0.018 or angle >= 12:
            raise RuntimeError(f'{label}: IK pose tolerance not reached: {1000 * err:.1f} mm, {angle:.1f} deg')
        if not quiet:
            print(f'  {label}: TCP error {1000 * err:.1f} mm, {angle:.1f} deg')
        return {'tip_error_mm': 1000 * err, 'tip_angle_deg': angle}

    def cleanup(self):
        try:
            self.ik.eraseEnvironment(self.env)
        except Exception:
            pass
        try:
            self.sim.removeObject(self.target)
        except Exception:
            pass


def live_poses(sim, camera, K, dist, B_from_C, output=None, verbose=True):
    """One live RGB-D frame -> per-colour pose estimates (no simulator object poses)."""
    bgr = read_bgr(sim, camera)
    depth = read_depth_m(sim, camera, bgr.shape[:2])
    _, masks = detect_from_rgb(bgr)
    valid_depth = np.isfinite(depth) & (depth > 0.08) & (depth < 5.0)
    poses, diag, overlay = {}, {}, bgr.copy()
    for color in COLORS:
        T, info = estimate_pose_from_mask(depth, masks[color], K, dist, B_from_C)
        poses[color], diag[color] = T, info
        if T is not None:
            draw_axes(overlay, T, B_from_C, K, dist)
        elif verbose:
            print(f'  {color} pose unavailable:', info.get('reason', 'unknown'))
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output / 'rgb.png'), bgr)
        cv2.imwrite(str(output / 'perception.png'), overlay)
        finite = depth[np.isfinite(depth)]
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99])
            hi = hi if hi - lo > 1e-6 else lo + 1.0
            vis = np.nan_to_num(np.clip((depth - lo) / (hi - lo), 0, 1), nan=0.0)
            cv2.imwrite(str(output / 'depth.png'),
                        cv2.applyColorMap((255 * vis).astype(np.uint8), cv2.COLORMAP_TURBO))
        for color in COLORS:
            cv2.imwrite(str(output / f'mask_{color}.png'), masks[color])
    if verbose:
        print('  valid depth pixels:', int(valid_depth.sum()), '/', int(depth.size))
    return poses, diag


def placement_errors(final_B, target_B_xyz, target_yaw=None) -> dict:
    """Placement error of a block (ground-truth pose) w.r.t. its target. EVALUATION."""
    pos = final_B[:3, 3]
    target = np.asarray(target_B_xyz, float)
    out = {'final_position_B_m': pos.tolist(),
           'placement_error_mm': 1000 * float(np.linalg.norm(pos - target)),
           'placement_xy_error_mm': 1000 * float(np.linalg.norm(pos[:2] - target[:2])),
           'placement_z_error_mm': 1000 * float(abs(pos[2] - target[2])),
           'placement_yaw_error_deg': None}
    if target_yaw is not None:
        out['placement_yaw_error_deg'] = symmetric_yaw_error_deg(yaw_of(final_B), target_yaw)
    return out


def is_success(err: dict, xy_tol_mm=DEFAULT_XY_TOL_MM, z_tol_mm=DEFAULT_Z_TOL_MM,
               yaw_tol_deg=DEFAULT_YAW_TOL_DEG) -> bool:
    ok = err['placement_xy_error_mm'] < xy_tol_mm and err['placement_z_error_mm'] < z_tol_mm
    if err.get('placement_yaw_error_deg') is not None:
        ok = ok and err['placement_yaw_error_deg'] < yaw_tol_deg
    return bool(ok)


def evaluate_object(sim, obj, base, target_B_xyz, target_yaw=None, **tolerances) -> dict:
    """Read the block's TRUE pose after the motion and score it (evaluation only)."""
    final_B = obj_pose_matrix(sim.getObjectPose(obj, base))
    out = placement_errors(final_B, target_B_xyz, target_yaw)
    out['success'] = is_success(out, **tolerances)
    out['tolerances'] = {'xy_mm': tolerances.get('xy_tol_mm', DEFAULT_XY_TOL_MM),
                         'z_mm': tolerances.get('z_tol_mm', DEFAULT_Z_TOL_MM),
                         'yaw_deg': tolerances.get('yaw_tol_deg', DEFAULT_YAW_TOL_DEG)}
    out['note'] = 'Outcome uses simulator ground truth for evaluation only.'
    return out
